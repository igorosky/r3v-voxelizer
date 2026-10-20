use rustc_hash::FxBuildHasher;

use crate::{
    Voxelizer, VoxelizerParams,
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

#[derive(Debug, Clone)]
pub struct SingleThreaded {
    voxels: HashMap<ThreeDim, [u32; 2_usize.pow(3 * BITS_FOR_ZORDER)]>,
    compressed_voxels:
        Option<HashMap<ThreeDim, [u64; 2_usize.pow(3 * BITS_FOR_COMPRESSED_ZORDER)]>>,
    result: HashMap<PagedIndex, u32>,
    // Pages a voxel of which got zeroed during the current call
    maybe_empty: HashSet<ThreeDim>,
    resolution: f32,
    intensity_multiplier: f32,
    hit_multiplier: u32,
    target_multiplier: u32,
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
            intensity_multiplier,
            hit_multiplier,
            target_multiplier,
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
            intensity_multiplier,
            hit_multiplier,
            target_multiplier,
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
        let intensity_multiplier = self.intensity_multiplier;
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
                if prev < self.construct_threshold && voxel >= self.construct_threshold {
                    let v = &mut compressed_voxels.entry(idx()).or_insert_with(|| [0; _])[idx2()];
                    *v |= 1 << x << (y << 2) << (z << 4);
                    changed(*v);
                } else if prev > self.destroy_threshold && voxel <= self.destroy_threshold {
                    debug_assert!(
                        compressed_voxels
                            .get_mut(&idx())
                            .expect("It must have existed if it is destroyed")[idx2()]
                            & (1 << x << (y << 2) << (z << 4))
                            != 0
                    );
                    let v = &mut compressed_voxels
                        .get_mut(&idx())
                        .expect("It must have existed if it is destroyed")[idx2()];
                    *v ^= 1 << x << (y << 2) << (z << 4);
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
            .saturating_add(point.i * self.target_multiplier)
            .min(self.max_value);
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
            i: (point.i * self.intensity_multiplier) as u32 * self.hit_multiplier,
        };
        let dir_x = (target_point.x - sensor_pos.x).signum();
        let dir_y = (target_point.y - sensor_pos.y).signum();
        let dir_z = (target_point.z - sensor_pos.z).signum();
        sensor_pos.i = target_point.i;
        point -= sensor_pos_divided;
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
                while target_point != sensor_pos {
                    debug_assert!(max_x < 1.0 + dx);
                    debug_assert!(max_y < 1.0 + dy);
                    self.hit_voxel(&sensor_pos);
                    if max_x < max_y {
                        sensor_pos.x += dir_x;
                        max_x += dx;
                    } else {
                        sensor_pos.y += dir_y;
                        max_y += dy;
                    }
                }
            }
            (x, Ordering::Equal, z) => {
                let dx = 1.0 / point.x.abs();
                let dz = 1.0 / point.z.abs();
                let mut max_x = Self::closest_boundary(x.is_gt(), sensor_pos_divided.x, point.x);
                let mut max_z = Self::closest_boundary(z.is_gt(), sensor_pos_divided.z, point.z);
                while target_point != sensor_pos {
                    debug_assert!(max_x < 1.0 + dx);
                    debug_assert!(max_z < 1.0 + dz);
                    self.hit_voxel(&sensor_pos);
                    if max_x < max_z {
                        sensor_pos.x += dir_x;
                        max_x += dx;
                    } else {
                        sensor_pos.z += dir_z;
                        max_z += dz;
                    }
                }
            }
            (Ordering::Equal, y, z) => {
                let dy = 1.0 / point.y.abs();
                let dz = 1.0 / point.z.abs();
                let mut max_y = Self::closest_boundary(y.is_gt(), sensor_pos_divided.y, point.y);
                let mut max_z = Self::closest_boundary(z.is_gt(), sensor_pos_divided.z, point.z);
                while target_point != sensor_pos {
                    debug_assert!(max_y < 1.0 + dy);
                    debug_assert!(max_z < 1.0 + dz);
                    self.hit_voxel(&sensor_pos);
                    if max_y < max_z {
                        sensor_pos.y += dir_y;
                        max_y += dy;
                    } else {
                        sensor_pos.z += dir_z;
                        max_z += dz;
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
                while target_point != sensor_pos {
                    debug_assert!(max_x < 1.0 + dx);
                    debug_assert!(max_y < 1.0 + dy);
                    debug_assert!(max_z < 1.0 + dz);
                    self.hit_voxel(&sensor_pos);
                    if max_x < max_z {
                        if max_x < max_y {
                            sensor_pos.x += dir_x;
                            max_x += dx;
                        } else {
                            sensor_pos.y += dir_y;
                            max_y += dy;
                        }
                    } else {
                        if max_y < max_z {
                            sensor_pos.y += dir_y;
                            max_y += dy;
                        } else {
                            sensor_pos.z += dir_z;
                            max_z += dz;
                        }
                    }
                }
            }
        };
    }
}

#[cfg(test)]
mod test {
    use super::{
        BITS_FOR_COMPRESSED_ZORDER, BITS_FOR_COMPRESSED_ZORDER_MASK, BITS_FOR_ZORDER, PAGE_SIZE,
        SingleThreaded,
    };
    use crate::{
        VoxelizerParams,
        geometry::PointI,
        test::{Coord, PAGE_LEN, PagedIndex, TestVoxelizer, compressed, pi, plain},
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
            intensity_multiplier: 0.25,
            hit_multiplier: 2,
            target_multiplier: 3,
            destroy_threshold: 4,
            construct_threshold: 5,
            max_value: 6,
            compress: false,
        });
        assert_eq!(v.resolution, 0.5);
        assert_eq!(v.intensity_multiplier, 0.25);
        assert_eq!(v.hit_multiplier, 2);
        assert_eq!(v.target_multiplier, 3);
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
}
