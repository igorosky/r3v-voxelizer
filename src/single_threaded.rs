use rustc_hash::FxBuildHasher;

use crate::{
    ALL_POINTS_GROUP_EDGE_BITS, CompressedVoxelGroup, Multipliers, Voxelizer, VoxelizerParams,
    geometry::{PointF, PointI, ZOrder},
};

const BITS_FOR_ZORDER: u32 = 2;
const BITS_FOR_ZORDER_MASK: i32 = 2_i32.pow(BITS_FOR_ZORDER) - 1;
const PAGE_SIZE: u32 = 4096;
const BITS_FOR_COMPRESSED_ZORDER: u32 = PAGE_SIZE.highest_one().unwrap() / 3;
const BITS_FOR_COMPRESSED_ZORDER_MASK: i32 = 2_i32.pow(BITS_FOR_COMPRESSED_ZORDER) - 1;

type HashMap<K, V> = hashbrown::HashMap<K, V, FxBuildHasher>;
type HashSet<K> = hashbrown::HashSet<K, FxBuildHasher>;
type ThreeDim = (i32, i32, i32);
type PagedIndex = (ThreeDim, usize);

static_assertions::const_assert!(BITS_FOR_COMPRESSED_ZORDER == ALL_POINTS_GROUP_EDGE_BITS);

#[derive(Debug, Clone)]
pub struct SingleThreaded {
    voxels: HashMap<ThreeDim, [u32; 2_usize.pow(3 * BITS_FOR_ZORDER)]>,
    compressed_voxels:
        Option<HashMap<ThreeDim, [u64; 2_usize.pow(3 * BITS_FOR_COMPRESSED_ZORDER)]>>,
    result: HashMap<PagedIndex, u32>,
    // Pages a voxel of which got zeroed or stayed 0 during the current call
    maybe_empty: HashSet<ThreeDim>,
    resolution: f32,
    multipliers: Multipliers,
    destroy_threshold: u32,
    construct_threshold: u32,
    max_value: u32,
    zorder: ZOrder,
}

impl SingleThreaded {
    #[inline]
    pub fn new(params: VoxelizerParams) -> Self {
        let VoxelizerParams {
            resolution,
            multipliers,
            destroy_threshold,
            construct_threshold,
            max_value,
            compress,
        } = params;
        Self {
            voxels: HashMap::default(),
            compressed_voxels: if compress {
                Some(HashMap::default())
            } else {
                None
            },
            result: HashMap::default(),
            maybe_empty: HashSet::default(),
            resolution,
            multipliers,
            destroy_threshold,
            construct_threshold,
            max_value,
            zorder: ZOrder::default(),
        }
    }
}

impl Voxelizer for SingleThreaded {
    fn add_points(&mut self, mut sensor_pos: PointF, points: &[PointF]) -> Vec<PointI> {
        sensor_pos /= self.resolution;
        let sensor_pos_i = PointI {
            x: sensor_pos.x.floor() as i32,
            y: sensor_pos.y.floor() as i32,
            z: sensor_pos.z.floor() as i32,
            i: sensor_pos.i as u32,
        };
        points
            .iter()
            .for_each(|v| self.ray_march(sensor_pos_i.clone(), &sensor_pos, v.clone()));
        let resolution = self.resolution;
        let intensity_multiplier = self.multipliers.intensity;
        points
            .iter()
            .map(move |v| v / resolution)
            .map(|v| PointI {
                x: v.x.floor() as i32,
                y: v.y.floor() as i32,
                z: v.z.floor() as i32,
                i: (v.i * intensity_multiplier) as u32,
            })
            .for_each(|point| self.add_voxel(&point));
        let mut result = Vec::new();
        // Compressed pages a word of which got zeroed
        let mut emptied_compressed = Vec::new();
        let mut map = HashMap::default();
        std::mem::swap(&mut map, &mut self.result);
        let iter = map
            .drain()
            .map(|(idx, prev)| {
                (
                    idx,
                    prev,
                    self.voxels
                        .get(&idx.0)
                        .expect("This voxel was touched - it must exist")[idx.1],
                )
            })
            .filter(|(_, prev, voxel)| voxel != prev);
        if let Some(compressed_voxels) = self.compressed_voxels.as_mut() {
            let mut compressed_dedup = HashMap::default();
            for (idx, prev, voxel) in iter {
                let [x, y, z] = self.zorder.coord_of(idx.1 as u64);
                let idx2 = || {
                    self.zorder.index_of([
                        (idx.0.0 & BITS_FOR_COMPRESSED_ZORDER_MASK) as u16,
                        (idx.0.1 & BITS_FOR_COMPRESSED_ZORDER_MASK) as u16,
                        (idx.0.2 & BITS_FOR_COMPRESSED_ZORDER_MASK) as u16,
                    ]) as usize
                };
                let mut changed = |v| {
                    let upper_bits = z as i32 & 2;
                    debug_assert!(upper_bits & !2 == 0);
                    compressed_dedup.insert(
                        (
                            idx.0.0 << BITS_FOR_ZORDER,
                            idx.0.1 << BITS_FOR_ZORDER,
                            (idx.0.2 << BITS_FOR_ZORDER) | upper_bits,
                        ),
                        if upper_bits != 0 {
                            (v >> 32) as u32
                        } else {
                            v as u32
                        },
                    );
                };
                let idx = || {
                    (
                        idx.0.0 >> BITS_FOR_COMPRESSED_ZORDER,
                        idx.0.1 >> BITS_FOR_COMPRESSED_ZORDER,
                        idx.0.2 >> BITS_FOR_COMPRESSED_ZORDER,
                    )
                };
                let bit: u64 = 1 << x << (y << 2) << (z << 4);
                // A voxel between the thresholds keeps its state, so a
                // crossing changes it only when it is in the other one
                if prev < self.construct_threshold && voxel >= self.construct_threshold {
                    let v = &mut compressed_voxels.entry(idx()).or_insert_with(|| [0; _])[idx2()];
                    if *v & bit == 0 {
                        *v |= bit;
                        changed(*v);
                    }
                } else if prev > self.destroy_threshold
                    && voxel <= self.destroy_threshold
                    && let Some(words) = compressed_voxels.get_mut(&idx())
                    && words[idx2()] & bit != 0
                {
                    let v = &mut words[idx2()];
                    *v &= !bit;
                    changed(*v);
                    if *v == 0 {
                        emptied_compressed.push(idx());
                    }
                }
            }
            result.extend(
                compressed_dedup
                    .into_iter()
                    .map(|((x, y, z), i)| PointI { x, y, z, i }),
            );
        } else {
            result.extend(iter.map(|(idx, _, voxel)| self.point_from_index(idx, voxel)));
        }
        std::mem::swap(&mut map, &mut self.result);
        self.remove_emptied(emptied_compressed);
        result
    }
    #[inline]
    fn set_multipliers(&mut self, multipliers: Multipliers) {
        self.multipliers = multipliers;
    }
    fn get_all_points(&self) -> Vec<PointI> {
        self.voxels
            .iter()
            .flat_map(|(p, v)| {
                v.iter()
                    .enumerate()
                    .filter(|(_, v)| **v != 0)
                    .map(|(i, v)| self.point_from_index((*p, i), *v))
            })
            .collect()
    }
    fn get_all_points_compressed(&self) -> Vec<CompressedVoxelGroup> {
        if let Some(compressed) = self.compressed_voxels.as_ref() {
            compressed
                .into_iter()
                .map(|((x, y, z), v)| CompressedVoxelGroup {
                    point: PointI {
                        x: *x,
                        y: *y,
                        z: *z,
                        i: 0,
                    },
                    values: (*v).into(),
                })
                .collect()
        } else {
            todo!()
        }
    }
    #[inline]
    fn multipliers(&self) -> Multipliers {
        self.multipliers
    }
}

impl SingleThreaded {
    #[inline]
    fn get_index(&self, point: &PointI) -> PagedIndex {
        (
            (
                point.x >> BITS_FOR_ZORDER,
                point.y >> BITS_FOR_ZORDER,
                point.z >> BITS_FOR_ZORDER,
            ),
            self.zorder.index_of([
                (point.x & BITS_FOR_ZORDER_MASK) as u16,
                (point.y & BITS_FOR_ZORDER_MASK) as u16,
                (point.z & BITS_FOR_ZORDER_MASK) as u16,
            ]) as usize,
        )
    }
    #[inline]
    fn point_from_index(&self, idx: PagedIndex, i: u32) -> PointI {
        let [x, y, z] = self.zorder.coord_of(idx.1 as u64);
        PointI {
            x: (idx.0.0 << BITS_FOR_ZORDER) | x as i32,
            y: (idx.0.1 << BITS_FOR_ZORDER) | y as i32,
            z: (idx.0.2 << BITS_FOR_ZORDER) | z as i32,
            i,
        }
    }
    #[inline]
    fn closest_boundary(upper: bool, position: f32, len: f32) -> f32 {
        let boundary = position.floor() + upper as i32 as f32;
        let result = (boundary - position) / len;
        debug_assert!(result >= 0.0);
        result
    }
    /// `max_[xyz]` of an axis after a step along it. An axis that reached the
    /// target is never stepped again: when the point lies on boundaries of
    /// several axes (or due to float errors) its next crossing can tie with
    /// the last one of another axis, and stepping it would walk past the
    /// target.
    #[inline]
    fn next_boundary(max: f32, delta: f32, reached: bool) -> f32 {
        if reached { f32::INFINITY } else { max + delta }
    }
    #[inline]
    fn hit_voxel(&mut self, point: &PointI) {
        let idx = self.get_index(point);
        // If voxel is not hit it does not exists anyway
        if let Some(v) = self.voxels.get_mut(&idx.0) {
            self.result.entry(idx).or_insert(v[idx.1]);
            let prev = v[idx.1];
            v[idx.1] = prev.saturating_sub(point.i);
            if prev != 0 && v[idx.1] == 0 {
                self.maybe_empty.insert(idx.0);
            }
        }
    }
    /// Key of the compressed page holding the bits of `page` and index of
    /// their word in it.
    #[inline]
    fn compressed_index(&self, page: ThreeDim) -> PagedIndex {
        (
            (
                page.0 >> BITS_FOR_COMPRESSED_ZORDER,
                page.1 >> BITS_FOR_COMPRESSED_ZORDER,
                page.2 >> BITS_FOR_COMPRESSED_ZORDER,
            ),
            self.zorder.index_of([
                (page.0 & BITS_FOR_COMPRESSED_ZORDER_MASK) as u16,
                (page.1 & BITS_FOR_COMPRESSED_ZORDER_MASK) as u16,
                (page.2 & BITS_FOR_COMPRESSED_ZORDER_MASK) as u16,
            ]) as usize,
        )
    }
    /// Removes pages zeroed during the call that are empty (all voxels 0 and
    /// none constructed) and compressed pages from `emptied_compressed`
    /// that are empty (all words 0).
    fn remove_emptied(&mut self, emptied_compressed: Vec<ThreeDim>) {
        for page in std::mem::take(&mut self.maybe_empty) {
            let (key, inner) = self.compressed_index(page);
            let constructed = self
                .compressed_voxels
                .as_ref()
                .and_then(|compressed_voxels| compressed_voxels.get(&key))
                .is_some_and(|words| words[inner] != 0);
            if !constructed && self.voxels[&page].iter().all(|&v| v == 0) {
                self.voxels.remove(&page);
            }
        }
        if let Some(compressed_voxels) = self.compressed_voxels.as_mut() {
            for key in emptied_compressed {
                if compressed_voxels
                    .get(&key)
                    .is_some_and(|words| words.iter().all(|&w| w == 0))
                {
                    compressed_voxels.remove(&key);
                }
            }
        }
    }
    #[inline]
    fn add_voxel(&mut self, point: &PointI) {
        let idx = self.get_index(point);
        let entry = self.voxels.entry(idx.0).or_insert_with(|| [0; _]);
        self.result.entry(idx).or_insert(entry[idx.1]);
        entry[idx.1] = entry[idx.1]
            .saturating_add(point.i * self.multipliers.target)
            .min(self.max_value);
        // A target adding nothing (or capped at 0) can leave a new page all 0
        if entry[idx.1] == 0 {
            self.maybe_empty.insert(idx.0);
        }
    }
    fn ray_march(
        &mut self,
        mut sensor_pos: PointI,
        sensor_pos_divided: &PointF,
        mut point: PointF,
    ) {
        // http://www.cse.yorku.ca/~amana/research/grid.pdf
        point /= self.resolution;
        // dir_[xyz] cannot be calculated after `point -= sensor_pos_divided;`
        // due to desired rounding before subtraction
        let target_point = PointI {
            x: point.x.floor() as i32,
            y: point.y.floor() as i32,
            z: point.z.floor() as i32,
            i: (point.i * self.multipliers.intensity) as u32 * self.multipliers.hit,
        };
        let dir_x = (target_point.x - sensor_pos.x).signum();
        let dir_y = (target_point.y - sensor_pos.y).signum();
        let dir_z = (target_point.z - sensor_pos.z).signum();
        sensor_pos.i = target_point.i;
        point -= sensor_pos_divided;
        // Every step moves exactly one voxel towards the target, so this is
        // the exact step count. Used instead of `target_point != sensor_pos`
        // so float errors can't cause an infinite loop
        let steps = (target_point.x - sensor_pos.x).abs()
            + (target_point.y - sensor_pos.y).abs()
            + (target_point.z - sensor_pos.z).abs();
        use std::cmp::Ordering;
        match (dir_x.cmp(&0), dir_y.cmp(&0), dir_z.cmp(&0)) {
            (Ordering::Equal, Ordering::Equal, Ordering::Equal) => (),
            (_, Ordering::Equal, Ordering::Equal) => {
                while target_point.x != sensor_pos.x {
                    self.hit_voxel(&sensor_pos);
                    sensor_pos.x += dir_x;
                }
            }
            (Ordering::Equal, _, Ordering::Equal) => {
                while target_point.y != sensor_pos.y {
                    self.hit_voxel(&sensor_pos);
                    sensor_pos.y += dir_y;
                }
            }
            (Ordering::Equal, Ordering::Equal, _) => {
                while target_point.z != sensor_pos.z {
                    self.hit_voxel(&sensor_pos);
                    sensor_pos.z += dir_z;
                }
            }
            (x, y, Ordering::Equal) => {
                let dx = 1.0 / point.x.abs();
                let dy = 1.0 / point.y.abs();
                let mut max_x = Self::closest_boundary(x.is_gt(), sensor_pos_divided.x, point.x);
                let mut max_y = Self::closest_boundary(y.is_gt(), sensor_pos_divided.y, point.y);
                for _ in 0..steps {
                    debug_assert!(sensor_pos.x == target_point.x || max_x < 1.0 + dx);
                    debug_assert!(sensor_pos.y == target_point.y || max_y < 1.0 + dy);
                    self.hit_voxel(&sensor_pos);
                    if max_x < max_y {
                        sensor_pos.x += dir_x;
                        max_x = Self::next_boundary(max_x, dx, sensor_pos.x == target_point.x);
                    } else {
                        sensor_pos.y += dir_y;
                        max_y = Self::next_boundary(max_y, dy, sensor_pos.y == target_point.y);
                    }
                }
            }
            (x, Ordering::Equal, z) => {
                let dx = 1.0 / point.x.abs();
                let dz = 1.0 / point.z.abs();
                let mut max_x = Self::closest_boundary(x.is_gt(), sensor_pos_divided.x, point.x);
                let mut max_z = Self::closest_boundary(z.is_gt(), sensor_pos_divided.z, point.z);
                for _ in 0..steps {
                    debug_assert!(sensor_pos.x == target_point.x || max_x < 1.0 + dx);
                    debug_assert!(sensor_pos.z == target_point.z || max_z < 1.0 + dz);
                    self.hit_voxel(&sensor_pos);
                    if max_x < max_z {
                        sensor_pos.x += dir_x;
                        max_x = Self::next_boundary(max_x, dx, sensor_pos.x == target_point.x);
                    } else {
                        sensor_pos.z += dir_z;
                        max_z = Self::next_boundary(max_z, dz, sensor_pos.z == target_point.z);
                    }
                }
            }
            (Ordering::Equal, y, z) => {
                let dy = 1.0 / point.y.abs();
                let dz = 1.0 / point.z.abs();
                let mut max_y = Self::closest_boundary(y.is_gt(), sensor_pos_divided.y, point.y);
                let mut max_z = Self::closest_boundary(z.is_gt(), sensor_pos_divided.z, point.z);
                for _ in 0..steps {
                    debug_assert!(sensor_pos.y == target_point.y || max_y < 1.0 + dy);
                    debug_assert!(sensor_pos.z == target_point.z || max_z < 1.0 + dz);
                    self.hit_voxel(&sensor_pos);
                    if max_y < max_z {
                        sensor_pos.y += dir_y;
                        max_y = Self::next_boundary(max_y, dy, sensor_pos.y == target_point.y);
                    } else {
                        sensor_pos.z += dir_z;
                        max_z = Self::next_boundary(max_z, dz, sensor_pos.z == target_point.z);
                    }
                }
            }
            (x, y, z) => {
                let dx = 1.0 / point.x.abs();
                let dy = 1.0 / point.y.abs();
                let dz = 1.0 / point.z.abs();
                let mut max_x = Self::closest_boundary(x.is_gt(), sensor_pos_divided.x, point.x);
                let mut max_y = Self::closest_boundary(y.is_gt(), sensor_pos_divided.y, point.y);
                let mut max_z = Self::closest_boundary(z.is_gt(), sensor_pos_divided.z, point.z);
                for _ in 0..steps {
                    debug_assert!(sensor_pos.x == target_point.x || max_x < 1.0 + dx);
                    debug_assert!(sensor_pos.y == target_point.y || max_y < 1.0 + dy);
                    debug_assert!(sensor_pos.z == target_point.z || max_z < 1.0 + dz);
                    self.hit_voxel(&sensor_pos);
                    if max_x < max_z {
                        if max_x < max_y {
                            sensor_pos.x += dir_x;
                            max_x = Self::next_boundary(max_x, dx, sensor_pos.x == target_point.x);
                        } else {
                            sensor_pos.y += dir_y;
                            max_y = Self::next_boundary(max_y, dy, sensor_pos.y == target_point.y);
                        }
                    } else {
                        if max_y < max_z {
                            sensor_pos.y += dir_y;
                            max_y = Self::next_boundary(max_y, dy, sensor_pos.y == target_point.y);
                        } else {
                            sensor_pos.z += dir_z;
                            max_z = Self::next_boundary(max_z, dz, sensor_pos.z == target_point.z);
                        }
                    }
                }
            }
        };
        // `steps` bounds the loops, so a wrong walk would end silently elsewhere
        debug_assert_eq!(sensor_pos, target_point);
    }
}

#[cfg(test)]
mod test {
    use std::collections::BTreeMap;

    use super::{
        BITS_FOR_COMPRESSED_ZORDER, BITS_FOR_COMPRESSED_ZORDER_MASK, BITS_FOR_ZORDER, PAGE_SIZE,
        SingleThreaded,
    };
    use crate::{
        ALL_POINTS_GROUP_EDGE_BITS, CompressedVoxelGroup, Multipliers, POINTS_IN_VOXEL_GROUP,
        Voxelizer, VoxelizerParams,
        geometry::{PointF, PointI, ZOrder},
        test::{
            COMPRESSED_PAGE_LEN, Coord, PAGE_LEN, PLAIN, PagedIndex, SCENARIO_PARAMS,
            TestVoxelizer, along_x, block_bit, block_of, compressed, compressed_bits, cube_faces,
            from_above, nonzero_voxels, page_bit, pf, pi, plain, sorted,
        },
    };

    impl TestVoxelizer for SingleThreaded {
        const BITS_FOR_ZORDER: u32 = BITS_FOR_ZORDER;
        const BITS_FOR_COMPRESSED_ZORDER: u32 = BITS_FOR_COMPRESSED_ZORDER;
        const BITS_FOR_COMPRESSED_ZORDER_MASK: i32 = BITS_FOR_COMPRESSED_ZORDER_MASK;
        const PAGE_SIZE: u32 = PAGE_SIZE;
        fn with_params(params: VoxelizerParams) -> Self {
            Self::new(params)
        }
        fn get_index(&self, point: &PointI) -> PagedIndex {
            SingleThreaded::get_index(self, point)
        }
        fn point_from_index(&self, idx: PagedIndex, i: u32) -> PointI {
            SingleThreaded::point_from_index(self, idx, i)
        }
        fn closest_boundary(upper: bool, position: f32, len: f32) -> f32 {
            SingleThreaded::closest_boundary(upper, position, len)
        }
        fn set_voxel(&mut self, x: i32, y: i32, z: i32, value: u32) {
            let idx = SingleThreaded::get_index(self, &pi(x, y, z, 0));
            self.voxels.entry(idx.0).or_insert_with(|| [0; PAGE_LEN])[idx.1] = value;
        }
        fn pages(&self) -> Vec<(Coord, [u32; PAGE_LEN])> {
            let mut out: Vec<_> = self.voxels.iter().map(|(k, v)| (*k, *v)).collect();
            out.sort_by_key(|(k, _)| *k);
            out
        }
        fn compressed_pages(&self) -> Option<Vec<(Coord, Vec<u64>)>> {
            let mut out: Vec<_> = self
                .compressed_voxels
                .as_ref()?
                .iter()
                .map(|(k, v)| (*k, v.to_vec()))
                .collect();
            out.sort_by_key(|(k, _)| *k);
            Some(out)
        }
        fn pending_results(&self) -> usize {
            self.result.len() + self.maybe_empty.len()
        }
    }

    crate::test::shared_tests!(SingleThreaded);

    // ------------------------------------------------------------ constructor

    #[test]
    fn new_stores_parameters() {
        let v = SingleThreaded::new(VoxelizerParams {
            resolution: 0.5,
            multipliers: Multipliers {
                intensity: 0.25,
                hit: 2,
                target: 3,
            },
            destroy_threshold: 4,
            construct_threshold: 5,
            max_value: 6,
            compress: false,
        });
        assert_eq!(v.resolution, 0.5);
        assert_eq!(
            v.multipliers,
            Multipliers {
                intensity: 0.25,
                hit: 2,
                target: 3,
            }
        );
        assert_eq!(v.destroy_threshold, 4);
        assert_eq!(v.construct_threshold, 5);
        assert_eq!(v.max_value, 6);
        assert!(v.voxels.is_empty());
        assert!(v.result.is_empty());
    }

    #[test]
    fn new_compress_flag() {
        assert!(plain::<SingleThreaded>().compressed_voxels.is_none());
        let v = compressed::<SingleThreaded>();
        assert!(v.compressed_voxels.as_ref().is_some_and(|m| m.is_empty()));
    }

    // ------------------------------------------------------------------- dump

    /// Two calls from the origin: call 1 targets the faces of a cube of size
    /// 50 (intensities 1..=5), call 2 a cube of size 100 (intensities 1..=3)
    /// whose rays cross call 1 targets. Returns the result of every call.
    fn workload(v: &mut SingleThreaded) -> Vec<Vec<PointI>> {
        let sensor = pf(0.5, 0.5, 0.5, 1.0);
        [(50, 20, 4, 5), (100, 40, 8, 3)]
            .into_iter()
            .map(|(dist, half, step, max_i)| {
                let points: Vec<PointF> = cube_faces(dist, half, step)
                    .into_iter()
                    .enumerate()
                    .map(|(k, (x, y, z))| {
                        let i = 1 + k as u32 % max_i;
                        pf(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5, i as f32)
                    })
                    .collect();
                v.add_points(sensor.clone(), &points)
            })
            .collect()
    }
    /// Points of `points` with a non-zero value, sorted.
    fn nonzero(points: Vec<PointI>) -> Vec<PointI> {
        sorted(points.into_iter().filter(|p| p.i != 0).collect())
    }
    /// Last value every voxel got in `results`, without the zeroed ones.
    fn replay(results: Vec<Vec<PointI>>) -> BTreeMap<Coord, u32> {
        let mut out = BTreeMap::new();
        for result in results {
            out.extend(result.into_iter().map(|p| ((p.x, p.y, p.z), p.i)));
        }
        out.retain(|_, i| *i != 0);
        out
    }
    /// Every voxel set in `groups`, with `i = 1`, sorted. A group holds one
    /// u64 word per fine page in Z-order, the bit of voxel `(x, y, z)` of the
    /// page is `x + 4 * y + 16 * z`.
    fn decode_groups(groups: &[CompressedVoxelGroup]) -> Vec<PointI> {
        let zorder = ZOrder::default();
        let mut out = Vec::new();
        for group in groups {
            for (inner, word) in group.values.iter().enumerate() {
                let [px, py, pz] = zorder.coord_of(inner as u64);
                let page = (
                    (group.point.x << ALL_POINTS_GROUP_EDGE_BITS) | px as i32,
                    (group.point.y << ALL_POINTS_GROUP_EDGE_BITS) | py as i32,
                    (group.point.z << ALL_POINTS_GROUP_EDGE_BITS) | pz as i32,
                );
                for bit in (0..64i32).filter(|bit| word & (1u64 << bit) != 0) {
                    out.push(pi(
                        (page.0 << BITS_FOR_ZORDER) | (bit & 3),
                        (page.1 << BITS_FOR_ZORDER) | ((bit >> 2) & 3),
                        (page.2 << BITS_FOR_ZORDER) | (bit >> 4),
                        1,
                    ));
                }
            }
        }
        sorted(out)
    }

    #[test]
    fn get_all_points_empty() {
        assert!(plain::<SingleThreaded>().get_all_points().is_empty());
        assert!(compressed::<SingleThreaded>().get_all_points().is_empty());
    }

    #[test]
    fn get_all_points_single_target() {
        let mut v = plain::<SingleThreaded>();
        from_above(&mut v, (5, 6, 7), 3.0);
        assert_eq!(v.get_all_points(), vec![pi(5, 6, 7, 3)]);
    }

    #[test]
    fn get_all_points_matches_voxels() {
        for params in SCENARIO_PARAMS.into_iter().filter(|p| !p.compress) {
            let mut v = SingleThreaded::new(params.clone());
            workload(&mut v);
            let dump = v.get_all_points();
            let zeros = dump.iter().filter(|p| p.i == 0).count();
            assert_eq!(zeros, 0, "{params:?}: zero voxels dumped");
            assert_eq!(sorted(dump), nonzero_voxels(&v), "{params:?}");
        }
    }

    #[test]
    fn get_all_points_zero_intensity_target() {
        let mut v = plain::<SingleThreaded>();
        assert!(from_above(&mut v, (5, 6, 7), 0.0).is_empty());
        assert!(v.get_all_points().is_empty());
    }

    #[test]
    fn zero_target_leaves_no_page() {
        // Intensity 0, or a cap of 0, leaves the target at 0
        for (params, i) in [
            (PLAIN, 0.0),
            (
                VoxelizerParams {
                    max_value: 0,
                    ..PLAIN
                },
                5.0,
            ),
        ] {
            let mut v = SingleThreaded::new(params.clone());
            assert!(from_above(&mut v, (5, 6, 7), i).is_empty(), "{params:?}");
            assert!(v.pages().is_empty(), "{params:?}");
            assert_eq!(v.pending_results(), 0, "{params:?}");
        }
    }

    #[test]
    fn get_all_points_matches_replayed_results() {
        for params in SCENARIO_PARAMS.into_iter().filter(|p| !p.compress) {
            let mut v = SingleThreaded::new(params.clone());
            let replayed: Vec<_> = replay(workload(&mut v))
                .into_iter()
                .map(|((x, y, z), i)| pi(x, y, z, i))
                .collect();
            assert!(!replayed.is_empty());
            assert_eq!(nonzero(v.get_all_points()), replayed, "{params:?}");
        }
    }

    #[test]
    fn get_all_points_negative_and_large_coords() {
        let voxels = [
            (-1, -1, -1, 1),
            (-4, 3, -5, 2),
            (3, 4, -4, 3),
            (63, -64, 64, 4),
            (i32::MIN, i32::MAX, 0, 5),
            (i32::MAX, i32::MIN, -1, 6),
        ];
        let mut v = plain::<SingleThreaded>();
        v.set_voxels(&voxels);
        let expected = sorted(voxels.map(|(x, y, z, i)| pi(x, y, z, i)).to_vec());
        assert_eq!(nonzero(v.get_all_points()), expected);
    }

    #[test]
    fn get_all_points_emptied_page_gone() {
        let mut v = plain::<SingleThreaded>();
        from_above(&mut v, (0, 0, 0), 2.0);
        // Drops (0, 0, 0) to 0, which empties its page
        along_x(&mut v, 0, 5, 0, 0, 2.0);
        let dump = v.get_all_points();
        assert!(dump.iter().all(|p| v.get_index(p).0 != (0, 0, 0)));
        assert_eq!(nonzero(dump), vec![pi(5, 0, 0, 2)]);
    }

    #[test]
    fn get_all_points_does_not_change_state() {
        for params in SCENARIO_PARAMS {
            let mut v = SingleThreaded::new(params.clone());
            workload(&mut v);
            let mut untouched = v.clone();
            let first = sorted(v.get_all_points());
            assert_eq!(sorted(v.get_all_points()), first, "{params:?}");
            if params.compress {
                assert_eq!(
                    v.get_all_points_compressed(),
                    v.get_all_points_compressed(),
                    "{params:?}"
                );
            }
            assert_eq!(v.pending_results(), 0, "{params:?}");
            assert_eq!(workload(&mut v), workload(&mut untouched), "{params:?}");
            assert_eq!(v.pages(), untouched.pages(), "{params:?}");
            assert_eq!(
                v.compressed_pages(),
                untouched.compressed_pages(),
                "{params:?}"
            );
        }
    }

    #[test]
    fn get_all_points_with_compress_returns_values() {
        for params in SCENARIO_PARAMS.into_iter().filter(|p| p.compress) {
            let mut v = SingleThreaded::new(params.clone());
            workload(&mut v);
            assert_eq!(
                nonzero(v.get_all_points()),
                nonzero_voxels(&v),
                "{params:?}"
            );
        }
    }

    // -------------------------------------------------------- compressed dump

    #[test]
    fn compressed_voxel_group_size() {
        assert_eq!(POINTS_IN_VOXEL_GROUP, COMPRESSED_PAGE_LEN);
        assert_eq!(ALL_POINTS_GROUP_EDGE_BITS, BITS_FOR_COMPRESSED_ZORDER);
    }

    #[test]
    fn get_all_points_compressed_empty() {
        assert!(
            compressed::<SingleThreaded>()
                .get_all_points_compressed()
                .is_empty()
        );
    }

    #[test]
    fn get_all_points_compressed_single_voxel() {
        let mut v = compressed::<SingleThreaded>();
        from_above(&mut v, (5, 6, 7), 1.0);
        let groups = v.get_all_points_compressed();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].point, pi(0, 0, 0, 0));
        let words: Vec<_> = groups[0]
            .values
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, word)| *word != 0)
            .collect();
        let inner = ZOrder::default().index_of([1u16, 1, 1]) as usize;
        assert_eq!(words, vec![(inner, page_bit(5, 6, 7))]);
    }

    #[test]
    fn get_all_points_compressed_group_keys() {
        // A group spans 16 fine pages, 64 voxels, along every axis
        for (voxel, key) in [
            ((-1, -1, -1), (-1, -1, -1)),
            ((64, 0, 0), (1, 0, 0)),
            ((63, 63, 63), (0, 0, 0)),
            ((-65, 0, 0), (-2, 0, 0)),
            ((0, -64, 128), (0, -1, 2)),
        ] {
            let mut v = compressed::<SingleThreaded>();
            from_above(&mut v, voxel, 1.0);
            let groups = v.get_all_points_compressed();
            assert_eq!(groups.len(), 1, "voxel {voxel:?}");
            assert_eq!(
                groups[0].point,
                pi(key.0, key.1, key.2, 0),
                "voxel {voxel:?}"
            );
            assert_eq!(
                decode_groups(&groups),
                vec![pi(voxel.0, voxel.1, voxel.2, 1)],
                "voxel {voxel:?}"
            );
        }
    }

    #[test]
    fn get_all_points_compressed_decodes_to_bits() {
        for params in SCENARIO_PARAMS.into_iter().filter(|p| p.compress) {
            let mut v = SingleThreaded::new(params.clone());
            workload(&mut v);
            let bits = compressed_bits(&v);
            assert!(!bits.is_empty(), "{params:?}");
            assert_eq!(
                decode_groups(&v.get_all_points_compressed()),
                bits,
                "{params:?}"
            );
        }
    }

    #[test]
    fn get_all_points_compressed_matches_replayed_results() {
        for params in SCENARIO_PARAMS.into_iter().filter(|p| p.compress) {
            let mut v = SingleThreaded::new(params.clone());
            let replayed = replay(workload(&mut v));
            let mut dumped = BTreeMap::new();
            for p in decode_groups(&v.get_all_points_compressed()) {
                *dumped.entry(block_of(p.x, p.y, p.z)).or_insert(0) |= block_bit(p.x, p.y, p.z);
            }
            assert_eq!(dumped, replayed, "{params:?}");
        }
    }

    #[test]
    fn get_all_points_compressed_destroyed_group_removed() {
        let mut v = compressed::<SingleThreaded>();
        from_above(&mut v, (5, 6, 7), 1.0);
        // Destroys (5, 6, 7) and constructs (70, 6, 7) in the next group
        along_x(&mut v, 5, 70, 6, 7, 1.0);
        let groups = v.get_all_points_compressed();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].point, pi(1, 0, 0, 0));
        assert_eq!(decode_groups(&groups), vec![pi(70, 6, 7, 1)]);
    }

    #[test]
    fn get_all_points_compressed_groups_unique_non_empty() {
        for params in SCENARIO_PARAMS.into_iter().filter(|p| p.compress) {
            let mut v = SingleThreaded::new(params.clone());
            workload(&mut v);
            let groups = v.get_all_points_compressed();
            assert!(groups.len() > 1, "{params:?}");
            let mut points: Vec<_> = groups.iter().map(|g| g.point.clone()).collect();
            points.sort();
            points.dedup();
            assert_eq!(points.len(), groups.len(), "{params:?}");
            assert!(points.iter().all(|p| p.i == 0), "{params:?}");
            assert!(
                groups.iter().all(|g| g.values.iter().any(|&w| w != 0)),
                "{params:?}"
            );
        }
    }
}
