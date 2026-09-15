use hashbrown::HashMap;

use crate::{
    Voxelizer,
    geometry::{PointF, PointI, ZOrder},
};

const BITS_FOR_ZORDER: u32 = 2;
const BITS_FOR_ZORDER_MASK: i32 = 2_i32.pow(BITS_FOR_ZORDER) - 1;
const PAGE_SIZE: u32 = 4096;
const BITS_FOR_COMPRESSED_ZORDER: u32 = PAGE_SIZE.highest_one().unwrap() / 3;
const BITS_FOR_COMPRESSED_ZORDER_MASK: i32 = 2_i32.pow(BITS_FOR_COMPRESSED_ZORDER) - 1;

type ThreeDim = (i32, i32, i32);
type PagedIndex = (ThreeDim, usize);

#[derive(Debug, Clone)]
pub struct SingleThreaded {
    voxels: HashMap<ThreeDim, [u32; 2_usize.pow(3 * BITS_FOR_ZORDER)]>,
    compressed_voxels:
        Option<HashMap<ThreeDim, [u64; 2_usize.pow(3 * BITS_FOR_COMPRESSED_ZORDER)]>>,
    result: HashMap<PagedIndex, u32>,
    resolution: f32,
    hit_multiplayer: u32,
    target_multiplayer: u32,
    destroy_threshold: u32,
    construct_threshold: u32,
    zorder: ZOrder,
}

impl SingleThreaded {
    #[inline]
    pub fn new(
        resolution: f32,
        hit_multiplayer: u32,
        target_multiplayer: u32,
        destroy_threshold: u32,
        construct_threshold: u32,
        compress: bool,
    ) -> Self {
        Self {
            voxels: HashMap::new(),
            compressed_voxels: if compress { Some(HashMap::new()) } else { None },
            result: HashMap::new(),
            resolution,
            hit_multiplayer,
            target_multiplayer,
            destroy_threshold,
            construct_threshold,
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
        let mut result = Vec::new();
        let mut map = hashbrown::HashMap::new();
        let mut compressed_dedup = HashMap::new();
        std::mem::swap(&mut map, &mut self.result);
        for (idx, prev) in map.drain() {
            let voxel = self
                .voxels
                .get(&idx.0)
                .expect("This voxel was touched - it must exist")[idx.1];
            if voxel != prev {
                if let Some(compressed_voxels) = self.compressed_voxels.as_mut() {
                    let [x, y, z] = self.zorder.coord_of(idx.1 as u64);
                    let idx2 = || {
                        self.zorder.index_of([
                            (idx.0.0 & BITS_FOR_COMPRESSED_ZORDER_MASK) as u16,
                            (idx.0.1 & BITS_FOR_COMPRESSED_ZORDER_MASK) as u16,
                            (idx.0.2 & BITS_FOR_COMPRESSED_ZORDER_MASK) as u16,
                        ]) as usize
                    };
                    let mut changed = |v: u64| {
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
                        let v =
                            &mut compressed_voxels.entry(idx()).or_insert_with(|| [0; _])[idx2()];
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
                    }
                } else {
                    result.push(self.point_from_index(idx, voxel));
                }
            }
        }
        std::mem::swap(&mut map, &mut self.result);
        if self.compressed_voxels.is_some() {
            result.extend(
                compressed_dedup
                    .into_iter()
                    .map(|((x, y, z), i)| PointI { x, y, z, i }),
            );
        }
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
            v[idx.1] = v[idx.1].saturating_sub(point.i * self.hit_multiplayer);
        }
    }
    #[inline]
    fn add_voxel(&mut self, point: &PointI) {
        let idx = self.get_index(point);
        let entry = self.voxels.entry(idx.0).or_insert_with(|| [0; _]);
        self.result.entry(idx).or_insert(entry[idx.1]);
        entry[idx.1] += point.i * self.target_multiplayer;
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
            i: point.i as u32,
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
        self.add_voxel(&sensor_pos);
    }
}

#[cfg(test)]
mod test {
    use super::{
        BITS_FOR_COMPRESSED_ZORDER, BITS_FOR_COMPRESSED_ZORDER_MASK, BITS_FOR_ZORDER, PAGE_SIZE,
    };
    use crate::{
        Voxelizer,
        geometry::{PointF, PointI},
        single_threaded::SingleThreaded,
    };

    const PAGE_LEN: usize = 2_usize.pow(3 * BITS_FOR_ZORDER);
    const COMPRESSED_PAGE_LEN: usize = 2_usize.pow(3 * BITS_FOR_COMPRESSED_ZORDER);
    /// Hits saturate at zero, so a voxel must hold something before a hit on
    /// it becomes observable. Hit voxels are seeded with this value.
    const SEED: u32 = 10;

    type Coord = (i32, i32, i32);

    #[inline]
    fn pf(x: f32, y: f32, z: f32, i: f32) -> PointF {
        PointF { x, y, z, i }
    }
    #[inline]
    fn pi(x: i32, y: i32, z: i32, i: u32) -> PointI {
        PointI { x, y, z, i }
    }
    #[inline]
    fn sorted<T: Ord>(mut v: Vec<T>) -> Vec<T> {
        v.sort();
        v
    }
    /// Resolution 1, multipliers 1, destroy at 0, construct at 1, no compression.
    fn plain() -> SingleThreaded {
        SingleThreaded::new(1.0, 1, 1, 0, 1, false)
    }
    /// Same as [`plain`] but with compression on.
    fn compressed() -> SingleThreaded {
        SingleThreaded::new(1.0, 1, 1, 0, 1, true)
    }
    /// Builds points from coordinate triples sharing one intensity.
    fn with_i(coords: &[Coord], i: u32) -> Vec<PointI> {
        coords.iter().map(|&(x, y, z)| pi(x, y, z, i)).collect()
    }
    /// Writes a voxel directly, creating its (zero filled) page when missing.
    /// Used to make pages exist so that hits become observable.
    fn set_voxel(v: &mut SingleThreaded, x: i32, y: i32, z: i32, value: u32) {
        let idx = v.get_index(&pi(x, y, z, 0));
        v.voxels.entry(idx.0).or_insert_with(|| [0; PAGE_LEN])[idx.1] = value;
    }
    fn seed(v: &mut SingleThreaded, coords: &[Coord], value: u32) {
        for &(x, y, z) in coords {
            set_voxel(v, x, y, z, value);
        }
    }
    fn get_voxel(v: &SingleThreaded, x: i32, y: i32, z: i32) -> Option<u32> {
        let idx = v.get_index(&pi(x, y, z, 0));
        v.voxels.get(&idx.0).map(|page| page[idx.1])
    }
    /// Every voxel with a non-zero value, sorted.
    fn nonzero_voxels(v: &SingleThreaded) -> Vec<PointI> {
        let out = v
            .voxels
            .iter()
            .flat_map(|(page, arr)| {
                arr.iter()
                    .enumerate()
                    .filter(|(_, val)| **val != 0)
                    .map(|(inner, val)| v.point_from_index((*page, inner), *val))
            })
            .collect();
        sorted(out)
    }
    /// Seeds every `hit` voxel with `SEED`, fires one ray of intensity 1 and
    /// checks that exactly those voxels dropped to `SEED - 1` and that
    /// `target` went to 1.
    fn check_ray(sensor: PointF, point: PointF, hit: &[Coord], target: Coord) {
        let mut v = plain();
        seed(&mut v, hit, SEED);
        let result = v.add_points(sensor, &[point]);
        let mut expected = with_i(hit, SEED - 1);
        expected.push(pi(target.0, target.1, target.2, 1));
        let expected = sorted(expected);
        assert_eq!(sorted(result), expected);
        assert_eq!(nonzero_voxels(&v), expected);
    }

    // ------------------------------------------------------ compress helpers
    //
    // A fine page is 4x4x4 voxels stored in one u64. The output of compress
    // mode is one PointI per changed 4x4x2 block (half of a fine page): x, y, z
    // are the voxel origin of the block, i is its 32 bit occupancy pack.

    /// Bit of voxel `(x, y, z)` inside the u64 word of its fine page.
    fn page_bit(x: i32, y: i32, z: i32) -> u64 {
        1 << ((x & 3) + 4 * (y & 3) + 16 * (z & 3))
    }
    /// Voxel origin of the 4x4x2 block that `(x, y, z)` belongs to.
    fn block_of(x: i32, y: i32, z: i32) -> Coord {
        (x & !3, y & !3, z & !1)
    }
    /// Bit of voxel `(x, y, z)` inside the 32 bit pack of its block.
    fn block_bit(x: i32, y: i32, z: i32) -> u32 {
        1 << ((x & 3) + 4 * (y & 3) + 16 * (z & 1))
    }
    /// Expected compress mode output for the block containing `(x, y, z)`.
    fn block(x: i32, y: i32, z: i32, mask: u32) -> PointI {
        let (bx, by, bz) = block_of(x, y, z);
        pi(bx, by, bz, mask)
    }
    /// The u64 word of the fine page containing voxel `(x, y, z)`, if its
    /// compressed page exists.
    fn compressed_word(v: &SingleThreaded, x: i32, y: i32, z: i32) -> Option<u64> {
        let page = v.get_index(&pi(x, y, z, 0)).0;
        let key = (
            page.0 >> BITS_FOR_COMPRESSED_ZORDER,
            page.1 >> BITS_FOR_COMPRESSED_ZORDER,
            page.2 >> BITS_FOR_COMPRESSED_ZORDER,
        );
        let inner = v.zorder.index_of([
            (page.0 & BITS_FOR_COMPRESSED_ZORDER_MASK) as u16,
            (page.1 & BITS_FOR_COMPRESSED_ZORDER_MASK) as u16,
            (page.2 & BITS_FOR_COMPRESSED_ZORDER_MASK) as u16,
        ]) as usize;
        v.compressed_voxels
            .as_ref()?
            .get(&key)
            .map(|words| words[inner])
    }
    /// Every voxel whose bit is set in `compressed_voxels`, with `i = 1`, sorted.
    fn compressed_bits(v: &SingleThreaded) -> Vec<PointI> {
        let Some(map) = v.compressed_voxels.as_ref() else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for (key, words) in map {
            for (inner, word) in words.iter().enumerate() {
                if *word == 0 {
                    continue;
                }
                let [px, py, pz] = v.zorder.coord_of(inner as u64);
                let page = (
                    (key.0 << BITS_FOR_COMPRESSED_ZORDER) | px as i32,
                    (key.1 << BITS_FOR_COMPRESSED_ZORDER) | py as i32,
                    (key.2 << BITS_FOR_COMPRESSED_ZORDER) | pz as i32,
                );
                for bit in 0..64i32 {
                    if word & (1u64 << bit) != 0 {
                        out.push(pi(
                            (page.0 << BITS_FOR_ZORDER) | (bit & 3),
                            (page.1 << BITS_FOR_ZORDER) | ((bit >> 2) & 3),
                            (page.2 << BITS_FOR_ZORDER) | (bit >> 4),
                            1,
                        ));
                    }
                }
            }
        }
        sorted(out)
    }
    /// Fires one ray of intensity `i` into voxel `(x, y, z)` from the voxel
    /// right above it, so the only voxel hit on the way is `(x, y, z + 1)`.
    fn from_above(v: &mut SingleThreaded, (x, y, z): Coord, i: f32) -> Vec<PointI> {
        let (fx, fy, fz) = (x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5);
        v.add_points(pf(fx, fy, fz + 1.0, 1.0), &[pf(fx, fy, fz, i)])
    }
    /// Fires one ray of intensity `i` along x from voxel `from_x` to voxel
    /// `to_x`, hitting every voxel in between (including `from_x`).
    fn along_x(
        v: &mut SingleThreaded,
        from_x: i32,
        to_x: i32,
        y: i32,
        z: i32,
        i: f32,
    ) -> Vec<PointI> {
        let (fy, fz) = (y as f32 + 0.5, z as f32 + 0.5);
        v.add_points(
            pf(from_x as f32 + 0.5, fy, fz, 1.0),
            &[pf(to_x as f32 + 0.5, fy, fz, i)],
        )
    }

    // ---------------------------------------------------------------- indexing

    #[test]
    fn constants() {
        assert_eq!(BITS_FOR_ZORDER, 2);
        assert_eq!(PAGE_LEN, 64);
        // One u64 word holds exactly one fine page.
        assert_eq!(PAGE_LEN, u64::BITS as usize);
        assert_eq!(BITS_FOR_COMPRESSED_ZORDER, 4);
        assert_eq!(BITS_FOR_COMPRESSED_ZORDER_MASK, 15);
        assert_eq!(COMPRESSED_PAGE_LEN, PAGE_SIZE as usize);
    }

    #[test]
    fn get_index_page_and_inner() {
        let v = plain();
        assert_eq!(v.get_index(&pi(0, 0, 0, 0)), ((0, 0, 0), 0));
        assert_eq!(v.get_index(&pi(17, 33, -1, 0)).0, (4, 8, -1));
        assert_eq!(v.get_index(&pi(3, 3, 3, 0)).0, (0, 0, 0));
        assert_eq!(v.get_index(&pi(4, 4, 4, 0)).0, (1, 1, 1));
        assert_eq!(v.get_index(&pi(-4, -4, -4, 0)).0, (-1, -1, -1));
        assert_eq!(v.get_index(&pi(-5, -5, -5, 0)).0, (-2, -2, -2));
        for c in [-1000, -5, -4, -1, 0, 3, 4, 1000] {
            assert!(v.get_index(&pi(c, c, c, 0)).1 < PAGE_LEN);
        }
        // Same offset inside a page gives the same inner index, whatever the page.
        let inner = v.get_index(&pi(3, 5, 7, 0)).1;
        assert_eq!(v.get_index(&pi(7, 9, 11, 0)).1, inner);
        assert_eq!(v.get_index(&pi(-1, -3, -5, 0)).1, inner);
        // Corners of a page.
        assert_eq!(v.get_index(&pi(4, 8, 12, 0)).1, 0);
        assert_eq!(v.get_index(&pi(7, 11, 15, 0)).1, PAGE_LEN - 1);
    }

    #[test]
    fn index_roundtrip() {
        let v = plain();
        let coords = [-1000, -5, -4, -3, -1, 0, 1, 3, 4, 5, 1000];
        for &x in &coords {
            for &y in &coords {
                for &z in &coords {
                    let p = pi(x, y, z, 42);
                    assert_eq!(v.point_from_index(v.get_index(&p), 42), p);
                }
            }
        }
    }

    #[test]
    fn inner_index_bijective_within_page() {
        let v = plain();
        let mut seen = vec![false; PAGE_LEN];
        for x in 0..4 {
            for y in 0..4 {
                for z in 0..4 {
                    let (page, inner) = v.get_index(&pi(x, y, z, 0));
                    assert_eq!(page, (0, 0, 0));
                    assert!(!seen[inner], "inner index {inner} used twice");
                    seen[inner] = true;
                }
            }
        }
        assert!(seen.iter().all(|s| *s));
    }

    // -------------------------------------------------------- closest_boundary

    #[test]
    fn closest_boundary_fractional() {
        assert_eq!(SingleThreaded::closest_boundary(true, 0.5, 2.0), 0.25);
        assert_eq!(SingleThreaded::closest_boundary(false, 0.5, -2.0), 0.25);
        assert_eq!(SingleThreaded::closest_boundary(true, 2.25, 4.0), 0.1875);
        assert_eq!(SingleThreaded::closest_boundary(false, 2.25, -1.0), 0.25);
        assert_eq!(SingleThreaded::closest_boundary(true, -0.5, 2.0), 0.25);
        assert_eq!(SingleThreaded::closest_boundary(false, -0.5, -2.0), 0.25);
    }

    #[test]
    fn closest_boundary_on_integer_lower() {
        // Voxel k spans [k, k+1); standing on k and going down crosses at once.
        assert_eq!(SingleThreaded::closest_boundary(false, 3.0, -2.0), 0.0);
    }

    #[test]
    fn closest_boundary_on_integer_upper() {
        // Standing on 3.0 means voxel [3, 4); going up the next face is 4.0.
        assert_eq!(SingleThreaded::closest_boundary(true, 3.0, 2.0), 0.5);
    }

    // ------------------------------------------------------------ constructor

    #[test]
    fn new_stores_parameters() {
        let v = SingleThreaded::new(0.5, 2, 3, 4, 5, false);
        assert_eq!(v.resolution, 0.5);
        assert_eq!(v.hit_multiplayer, 2);
        assert_eq!(v.target_multiplayer, 3);
        assert_eq!(v.destroy_threshold, 4);
        assert_eq!(v.construct_threshold, 5);
        assert!(v.voxels.is_empty());
        assert!(v.result.is_empty());
    }

    #[test]
    fn new_compress_flag() {
        assert!(plain().compressed_voxels.is_none());
        let v = compressed();
        assert!(v.compressed_voxels.as_ref().is_some_and(|m| m.is_empty()));
    }

    // -------------------------------------------------------- add_points basics

    #[test]
    fn basic_test_1d() {
        let mut x = plain();
        let result = x.add_points(pf(0.0, 0.0, 0.0, 1.0), &[pf(5.0, 0.0, 0.0, 1.0)]);
        assert_eq!(sorted(result), sorted(vec![pi(5, 0, 0, 1)]));
    }

    #[test]
    fn empty_points_no_change() {
        let mut v = plain();
        assert!(v.add_points(pf(0.5, 0.5, 0.5, 1.0), &[]).is_empty());
        assert!(v.voxels.is_empty());
        assert!(v.result.is_empty());
    }

    #[test]
    fn result_map_cleared_after_call() {
        let mut v = plain();
        let sensor = pf(0.5, 0.5, 0.5, 1.0);
        let points = [pf(3.5, 0.5, 0.5, 1.0)];
        v.add_points(sensor.clone(), &points);
        assert!(v.result.is_empty());
        v.add_points(sensor, &points);
        assert!(v.result.is_empty());
    }

    #[test]
    fn returns_absolute_value_not_delta() {
        let mut v = plain();
        let hits = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0)];
        seed(&mut v, &hits, SEED);
        set_voxel(&mut v, 5, 0, 0, 3);
        let result = v.add_points(pf(0.5, 0.5, 0.5, 1.0), &[pf(5.5, 0.5, 0.5, 2.0)]);
        let mut expected = with_i(&hits, SEED - 2);
        expected.push(pi(5, 0, 0, 5));
        let expected = sorted(expected);
        assert_eq!(sorted(result), expected);
        assert_eq!(nonzero_voxels(&v), expected);
    }

    #[test]
    fn sensor_intensity_ignored() {
        let mut v = plain();
        set_voxel(&mut v, 0, 0, 0, SEED);
        let result = v.add_points(pf(0.5, 0.5, 0.5, 99.0), &[pf(3.5, 0.5, 0.5, 1.0)]);
        // Hits use the point's intensity, not the sensor's.
        assert_eq!(sorted(result), vec![pi(0, 0, 0, SEED - 1), pi(3, 0, 0, 1)]);
        assert_eq!(get_voxel(&v, 0, 0, 0), Some(SEED - 1));
        assert_eq!(get_voxel(&v, 3, 0, 0), Some(1));
    }

    #[test]
    fn intensity_truncated_toward_zero() {
        let mut v = plain();
        let result = v.add_points(pf(0.5, 0.5, 0.5, 1.0), &[pf(3.5, 0.5, 0.5, 2.9)]);
        assert_eq!(result, vec![pi(3, 0, 0, 2)]);
    }

    #[test]
    fn zero_intensity_reports_nothing() {
        let mut v = plain();
        set_voxel(&mut v, 0, 0, 0, SEED);
        let result = v.add_points(pf(0.5, 0.5, 0.5, 1.0), &[pf(3.5, 0.5, 0.5, 0.0)]);
        assert!(result.is_empty());
        assert_eq!(get_voxel(&v, 0, 0, 0), Some(SEED));
        assert_eq!(get_voxel(&v, 3, 0, 0), Some(0));
    }

    #[test]
    fn resolution_scales_coords_not_intensity() {
        let mut v = SingleThreaded::new(0.25, 1, 1, 0, 1, false);
        let hits = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)];
        seed(&mut v, &hits, SEED);
        let result = v.add_points(pf(0.1, 0.1, 0.1, 1.0), &[pf(1.0, 0.1, 0.1, 7.0)]);
        let mut expected = with_i(&hits, SEED - 7);
        expected.push(pi(4, 0, 0, 7));
        assert_eq!(sorted(result), sorted(expected));

        let mut v = SingleThreaded::new(2.0, 1, 1, 0, 1, false);
        let result = v.add_points(pf(0.0, 0.0, 0.0, 1.0), &[pf(5.0, 0.0, 0.0, 7.0)]);
        assert_eq!(result, vec![pi(2, 0, 0, 7)]);
    }

    #[test]
    fn resolution_scales_sensor_too() {
        let mut v = SingleThreaded::new(2.0, 1, 1, 0, 1, false);
        seed(&mut v, &[(1, 0, 0), (2, 0, 0)], SEED);
        // sensor 2.0 -> 1.0 -> voxel 1, point 7.0 -> 3.5 -> voxel 3
        let result = v.add_points(pf(2.0, 0.0, 0.0, 1.0), &[pf(7.0, 0.0, 0.0, 1.0)]);
        assert_eq!(
            sorted(result),
            vec![pi(1, 0, 0, SEED - 1), pi(2, 0, 0, SEED - 1), pi(3, 0, 0, 1)]
        );
    }

    #[test]
    fn large_coords() {
        let mut v = plain();
        let result = v.add_points(
            pf(1000.5, -2000.5, 29999.5, 1.0),
            &[pf(1000.5, -2000.5, 30000.5, 4.0)],
        );
        assert_eq!(result, vec![pi(1000, -2001, 30000, 4)]);
        assert_eq!(get_voxel(&v, 1000, -2001, 30000), Some(4));
    }

    #[test]
    fn negative_coords_use_floor() {
        for (x, voxel) in [(-0.5, -1), (-1.0, -1), (-1.5, -2)] {
            let mut v = plain();
            let result = v.add_points(pf(-3.5, 0.5, 0.5, 1.0), &[pf(x, 0.5, 0.5, 1.0)]);
            assert_eq!(result, vec![pi(voxel, 0, 0, 1)], "point x = {x}");
        }
    }

    // ------------------------------------------------ multipliers / saturation

    #[test]
    fn hit_multiplayer_scales_hits() {
        let mut v = SingleThreaded::new(1.0, 3, 1, 0, 1, false);
        set_voxel(&mut v, 0, 0, 0, SEED);
        let result = v.add_points(pf(0.5, 0.5, 0.5, 1.0), &[pf(3.5, 0.5, 0.5, 2.0)]);
        assert_eq!(
            sorted(result),
            vec![pi(0, 0, 0, SEED - 2 * 3), pi(3, 0, 0, 2)]
        );
    }

    #[test]
    fn target_multiplayer_scales_target() {
        let mut v = SingleThreaded::new(1.0, 1, 5, 0, 1, false);
        set_voxel(&mut v, 0, 0, 0, SEED);
        let result = v.add_points(pf(0.5, 0.5, 0.5, 1.0), &[pf(3.5, 0.5, 0.5, 2.0)]);
        assert_eq!(
            sorted(result),
            vec![pi(0, 0, 0, SEED - 2), pi(3, 0, 0, 2 * 5)]
        );
    }

    #[test]
    fn both_multiplayers() {
        let mut v = SingleThreaded::new(1.0, 2, 3, 0, 1, false);
        set_voxel(&mut v, 0, 0, 0, SEED);
        let result = v.add_points(pf(0.5, 0.5, 0.5, 1.0), &[pf(3.5, 0.5, 0.5, 2.0)]);
        assert_eq!(
            sorted(result),
            vec![pi(0, 0, 0, SEED - 2 * 2), pi(3, 0, 0, 2 * 3)]
        );
        assert_eq!(get_voxel(&v, 0, 0, 0), Some(SEED - 4));
        assert_eq!(get_voxel(&v, 3, 0, 0), Some(6));
    }

    #[test]
    fn hit_saturates_at_zero() {
        let mut v = plain();
        set_voxel(&mut v, 0, 0, 0, 1);
        // (1,0,0) and (2,0,0) exist (same page) and hold 0.
        let result = v.add_points(pf(0.5, 0.5, 0.5, 1.0), &[pf(3.5, 0.5, 0.5, 5.0)]);
        // 1 -> 0 is a change and gets reported as 0; 0 -> 0 is not.
        assert_eq!(sorted(result), vec![pi(0, 0, 0, 0), pi(3, 0, 0, 5)]);
        assert_eq!(get_voxel(&v, 0, 0, 0), Some(0));
        assert_eq!(get_voxel(&v, 1, 0, 0), Some(0));
        assert_eq!(get_voxel(&v, 2, 0, 0), Some(0));

        // The multiplier saturates too.
        let mut v = SingleThreaded::new(1.0, 100, 1, 0, 1, false);
        set_voxel(&mut v, 0, 0, 0, SEED);
        let result = v.add_points(pf(0.5, 0.5, 0.5, 1.0), &[pf(3.5, 0.5, 0.5, 1.0)]);
        assert_eq!(sorted(result), vec![pi(0, 0, 0, 0), pi(3, 0, 0, 1)]);
    }

    #[test]
    fn hit_on_zero_voxel_not_reported() {
        let mut v = plain();
        set_voxel(&mut v, 0, 0, 0, 0);
        let result = v.add_points(pf(0.5, 0.5, 0.5, 1.0), &[pf(3.5, 0.5, 0.5, 1.0)]);
        assert_eq!(result, vec![pi(3, 0, 0, 1)]);
        assert_eq!(nonzero_voxels(&v), vec![pi(3, 0, 0, 1)]);
    }

    #[test]
    fn thresholds_ignored_without_compress() {
        let mut v = SingleThreaded::new(1.0, 1, 1, 100, 200, false);
        let hits = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0)];
        seed(&mut v, &hits, SEED);
        set_voxel(&mut v, 5, 0, 0, 3);
        let result = v.add_points(pf(0.5, 0.5, 0.5, 1.0), &[pf(5.5, 0.5, 0.5, 2.0)]);
        let mut expected = with_i(&hits, SEED - 2);
        expected.push(pi(5, 0, 0, 5));
        assert_eq!(sorted(result), sorted(expected));
        assert!(v.compressed_voxels.is_none());
    }

    // ------------------------------------------- accumulation / change detection

    #[test]
    fn accumulates_across_calls() {
        let mut v = plain();
        let hits = [(0, 0, 0), (1, 0, 0), (2, 0, 0)];
        seed(&mut v, &hits, SEED);
        let sensor = pf(0.5, 0.5, 0.5, 1.0);
        let points = [pf(3.5, 0.5, 0.5, 1.0)];
        let mut expected = with_i(&hits, SEED - 1);
        expected.push(pi(3, 0, 0, 1));
        assert_eq!(
            sorted(v.add_points(sensor.clone(), &points)),
            sorted(expected)
        );
        let mut expected = with_i(&hits, SEED - 2);
        expected.push(pi(3, 0, 0, 2));
        assert_eq!(sorted(v.add_points(sensor, &points)), sorted(expected));
        assert_eq!(get_voxel(&v, 3, 0, 0), Some(2));
    }

    #[test]
    fn unchanged_voxel_not_reported() {
        let mut v = plain();
        set_voxel(&mut v, 0, 0, 0, 5);
        set_voxel(&mut v, 1, 0, 0, 1);
        // First ray hits (1,0,0): 1 -> 0, second ray ends there: 0 -> 1.
        let result = v.add_points(
            pf(0.5, 0.5, 0.5, 1.0),
            &[pf(2.5, 0.5, 0.5, 1.0), pf(1.5, 0.5, 0.5, 1.0)],
        );
        assert_eq!(sorted(result), vec![pi(0, 0, 0, 3), pi(2, 0, 0, 1)]);
        assert_eq!(get_voxel(&v, 1, 0, 0), Some(1));
    }

    #[test]
    fn second_call_reports_only_its_changes() {
        let mut v = plain();
        seed(&mut v, &[(0, 0, 0), (1, 0, 0), (2, 0, 0)], SEED);
        let sensor = pf(0.5, 0.5, 0.5, 1.0);
        // Straight up along z: hits (0,0,0), (0,0,1), (0,0,2).
        assert_eq!(
            sorted(v.add_points(sensor.clone(), &[pf(0.5, 0.5, 3.5, 1.0)])),
            vec![pi(0, 0, 0, SEED - 1), pi(0, 0, 3, 1)]
        );
        let second = v.add_points(sensor, &[pf(3.5, 0.5, 0.5, 1.0)]);
        assert_eq!(
            sorted(second),
            vec![
                pi(0, 0, 0, SEED - 2),
                pi(1, 0, 0, SEED - 1),
                pi(2, 0, 0, SEED - 1),
                pi(3, 0, 0, 1)
            ]
        );
        assert_eq!(
            nonzero_voxels(&v),
            sorted(vec![
                pi(0, 0, 0, SEED - 2),
                pi(1, 0, 0, SEED - 1),
                pi(2, 0, 0, SEED - 1),
                pi(0, 0, 3, 1),
                pi(3, 0, 0, 1),
            ])
        );
    }

    #[test]
    fn multiple_points_one_call() {
        let mut v = plain();
        seed(
            &mut v,
            &[(0, 0, 0), (1, 0, 0), (2, 0, 0), (0, 1, 0), (0, 2, 0)],
            SEED,
        );
        let result = v.add_points(
            pf(0.5, 0.5, 0.5, 1.0),
            &[pf(3.5, 0.5, 0.5, 1.0), pf(0.5, 3.5, 0.5, 2.0)],
        );
        let expected = sorted(vec![
            pi(0, 0, 0, SEED - 1 - 2),
            pi(1, 0, 0, SEED - 1),
            pi(2, 0, 0, SEED - 1),
            pi(3, 0, 0, 1),
            pi(0, 1, 0, SEED - 2),
            pi(0, 2, 0, SEED - 2),
            pi(0, 3, 0, 2),
        ]);
        assert_eq!(sorted(result), expected);
        assert_eq!(nonzero_voxels(&v), expected);
    }

    // ------------------------------------------------------------------ paging

    #[test]
    fn hits_skip_missing_pages() {
        let mut v = plain();
        set_voxel(&mut v, 0, 0, 0, SEED);
        let sensor = pf(0.5, 0.5, 0.5, 1.0);
        let points = [pf(20.5, 0.5, 0.5, 1.0)];

        // The ray crosses fine pages 0..=5 along x but only page 0 exists.
        let first = v.add_points(sensor.clone(), &points);
        assert_eq!(sorted(first), vec![pi(0, 0, 0, SEED - 1), pi(20, 0, 0, 1)]);
        // Only the seeded page and the target's page exist afterwards.
        assert_eq!(v.voxels.len(), 2);
        for x in 1..4 {
            assert_eq!(get_voxel(&v, x, 0, 0), Some(0), "voxel x = {x}");
        }
        for x in 4..20 {
            assert_eq!(get_voxel(&v, x, 0, 0), None, "voxel x = {x}");
        }

        let second = v.add_points(sensor.clone(), &points);
        assert_eq!(sorted(second), vec![pi(0, 0, 0, SEED - 2), pi(20, 0, 0, 2)]);
        assert_eq!(v.voxels.len(), 2);

        // Once a page exists on the way its voxels do get hit.
        set_voxel(&mut v, 8, 0, 0, SEED);
        let third = v.add_points(sensor, &points);
        assert_eq!(
            sorted(third),
            vec![
                pi(0, 0, 0, SEED - 3),
                pi(8, 0, 0, SEED - 1),
                pi(20, 0, 0, 3)
            ]
        );
        assert_eq!(v.voxels.len(), 3);
    }

    #[test]
    fn negative_page() {
        let mut v = plain();
        let result = v.add_points(pf(-0.5, -0.5, -0.5, 1.0), &[pf(-3.5, -0.5, -0.5, 3.0)]);
        assert_eq!(result, vec![pi(-4, -1, -1, 3)]);
        assert_eq!(get_voxel(&v, -4, -1, -1), Some(3));
        assert_eq!(v.voxels.len(), 1);
        assert!(v.voxels.contains_key(&(-1, -1, -1)));
    }

    #[test]
    fn target_page_created_lazily() {
        let mut v = plain();
        assert!(v.voxels.is_empty());
        v.add_points(pf(0.5, 0.5, 0.5, 1.0), &[pf(9.5, 0.5, 0.5, 1.0)]);
        assert_eq!(v.voxels.len(), 1);
        assert!(v.voxels.contains_key(&(2, 0, 0)));
    }

    // -------------------------------------------------------- ray march arms

    #[test]
    fn axis_x_negative() {
        check_ray(
            pf(5.5, 0.5, 0.5, 1.0),
            pf(0.5, 0.5, 0.5, 1.0),
            &[(5, 0, 0), (4, 0, 0), (3, 0, 0), (2, 0, 0), (1, 0, 0)],
            (0, 0, 0),
        );
    }

    #[test]
    fn axis_y_positive() {
        check_ray(
            pf(0.5, 0.5, 0.5, 1.0),
            pf(0.5, 4.5, 0.5, 1.0),
            &[(0, 0, 0), (0, 1, 0), (0, 2, 0), (0, 3, 0)],
            (0, 4, 0),
        );
    }

    #[test]
    fn axis_y_negative() {
        check_ray(
            pf(0.5, 4.5, 0.5, 1.0),
            pf(0.5, 0.5, 0.5, 1.0),
            &[(0, 4, 0), (0, 3, 0), (0, 2, 0), (0, 1, 0)],
            (0, 0, 0),
        );
    }

    #[test]
    fn axis_z_positive() {
        check_ray(
            pf(0.5, 0.5, 0.5, 1.0),
            pf(0.5, 0.5, 4.5, 1.0),
            &[(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3)],
            (0, 0, 4),
        );
    }

    #[test]
    fn axis_z_negative() {
        check_ray(
            pf(0.5, 0.5, 4.5, 1.0),
            pf(0.5, 0.5, 0.5, 1.0),
            &[(0, 0, 4), (0, 0, 3), (0, 0, 2), (0, 0, 1)],
            (0, 0, 0),
        );
    }

    #[test]
    fn plane_xy_forward() {
        check_ray(
            pf(0.5, 0.5, 0.5, 1.0),
            pf(3.5, 1.7, 0.5, 1.0),
            &[(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0)],
            (3, 1, 0),
        );
    }

    #[test]
    fn plane_xy_reverse() {
        check_ray(
            pf(3.5, 1.7, 0.5, 1.0),
            pf(0.5, 0.5, 0.5, 1.0),
            &[(3, 1, 0), (2, 1, 0), (1, 1, 0), (1, 0, 0)],
            (0, 0, 0),
        );
    }

    #[test]
    fn plane_xz() {
        check_ray(
            pf(0.5, 0.5, 0.5, 1.0),
            pf(3.5, 0.5, 1.7, 1.0),
            &[(0, 0, 0), (1, 0, 0), (1, 0, 1), (2, 0, 1)],
            (3, 0, 1),
        );
    }

    #[test]
    fn plane_yz() {
        check_ray(
            pf(0.5, 0.5, 0.5, 1.0),
            pf(0.5, 3.5, 1.7, 1.0),
            &[(0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 2, 1)],
            (0, 3, 1),
        );
    }

    #[test]
    fn full_3d_forward() {
        check_ray(
            pf(0.5, 0.5, 0.5, 1.0),
            pf(2.5, 1.6, 3.5, 1.0),
            &[
                (0, 0, 0),
                (0, 0, 1),
                (1, 0, 1),
                (1, 1, 1),
                (1, 1, 2),
                (2, 1, 2),
            ],
            (2, 1, 3),
        );
    }

    #[test]
    fn full_3d_reverse() {
        check_ray(
            pf(2.5, 1.6, 3.5, 1.0),
            pf(0.5, 0.5, 0.5, 1.0),
            &[
                (2, 1, 3),
                (2, 1, 2),
                (1, 1, 2),
                (1, 1, 1),
                (1, 0, 1),
                (0, 0, 1),
            ],
            (0, 0, 0),
        );
    }

    #[test]
    fn sensor_on_grid_line_2d() {
        // Line y = 0.6x from the origin: crosses x=1 at t=0.4 (y=0.6), y=1 at
        // t=0.667 (x=1.67), x=2 at t=0.8.
        check_ray(
            pf(0.0, 0.0, 0.0, 1.0),
            pf(2.5, 1.5, 0.5, 1.0),
            &[(0, 0, 0), (1, 0, 0), (1, 1, 0)],
            (2, 1, 0),
        );
    }

    #[test]
    fn ray_across_page_boundary() {
        // Hits span fine pages (0,0,0) and (1,0,0) along x.
        check_ray(
            pf(2.5, 0.5, 0.5, 1.0),
            pf(6.5, 1.7, 0.5, 1.0),
            &[(2, 0, 0), (3, 0, 0), (4, 0, 0), (4, 1, 0), (5, 1, 0)],
            (6, 1, 0),
        );
    }

    // ------------------------------------------- point inside the sensor voxel

    #[test]
    fn point_in_sensor_voxel_fresh() {
        let mut v = plain();
        let result = v.add_points(pf(0.5, 0.5, 0.5, 1.0), &[pf(0.7, 0.5, 0.5, 3.0)]);
        assert_eq!(result, vec![pi(0, 0, 0, 3)]);
        assert_eq!(nonzero_voxels(&v), vec![pi(0, 0, 0, 3)]);
    }

    #[test]
    fn point_in_sensor_voxel_accumulates() {
        // The target voxel only ever receives `+i` in every other arm, so the
        // degenerate arm must add as well instead of cancelling itself out.
        let mut v = plain();
        set_voxel(&mut v, 0, 0, 0, 5);
        let result = v.add_points(pf(0.5, 0.5, 0.5, 1.0), &[pf(0.7, 0.5, 0.5, 3.0)]);
        assert_eq!(result, vec![pi(0, 0, 0, 8)]);
        assert_eq!(get_voxel(&v, 0, 0, 0), Some(8));
    }

    // ----------------------------------------------------------- compress mode

    #[test]
    fn compress_helpers_self_check() {
        assert_eq!(page_bit(0, 0, 0), 1);
        assert_eq!(page_bit(3, 0, 0), 1 << 3);
        assert_eq!(page_bit(0, 3, 0), 1 << 12);
        assert_eq!(page_bit(0, 0, 3), 1 << 48);
        assert_eq!(page_bit(3, 3, 3), 1 << 63);
        assert_eq!(block_bit(3, 3, 1), 1 << 31);
        assert_eq!(block_bit(3, 3, 3), 1 << 31);
        assert_eq!(block_of(5, 9, 7), (4, 8, 6));
        assert_eq!(block_of(-1, -1, -1), (-4, -4, -2));
        assert_eq!(block_of(-5, -9, -6), (-8, -12, -6));
    }

    #[test]
    fn compress_construct_sets_bit() {
        let mut v = compressed();
        let result = along_x(&mut v, 0, 3, 0, 0, 1.0);
        assert_eq!(get_voxel(&v, 3, 0, 0), Some(1));
        assert_eq!(compressed_word(&v, 3, 0, 0), Some(page_bit(3, 0, 0)));
        assert_eq!(compressed_bits(&v), vec![pi(3, 0, 0, 1)]);
        let map = v.compressed_voxels.as_ref().unwrap();
        assert_eq!(map.len(), 1);
        assert!(map.contains_key(&(0, 0, 0)));
        assert_eq!(map[&(0, 0, 0)].len(), COMPRESSED_PAGE_LEN);
        assert_eq!(result, vec![block(3, 0, 0, block_bit(3, 0, 0))]);
        assert_eq!(result, vec![pi(0, 0, 0, 1 << 3)]);
    }

    #[test]
    fn compress_no_recross_no_report() {
        let mut v = compressed();
        along_x(&mut v, 0, 3, 0, 0, 1.0);
        // 1 -> 2 stays above the construct threshold: nothing to report, even
        // though the voxel itself changed.
        let second = along_x(&mut v, 0, 3, 0, 0, 1.0);
        assert!(second.is_empty());
        assert_eq!(get_voxel(&v, 3, 0, 0), Some(2));
        assert_eq!(compressed_word(&v, 3, 0, 0), Some(page_bit(3, 0, 0)));
        assert_eq!(compressed_bits(&v), vec![pi(3, 0, 0, 1)]);
    }

    #[test]
    fn compress_plain_changes_not_returned() {
        // Below the construct threshold a voxel change is invisible in
        // compress mode, while plain mode reports it.
        let mut c = SingleThreaded::new(1.0, 1, 1, 0, 5, true);
        assert!(along_x(&mut c, 0, 3, 0, 0, 1.0).is_empty());
        assert_eq!(get_voxel(&c, 3, 0, 0), Some(1));
        assert!(c.compressed_voxels.as_ref().unwrap().is_empty());
        let mut p = SingleThreaded::new(1.0, 1, 1, 0, 5, false);
        assert_eq!(along_x(&mut p, 0, 3, 0, 0, 1.0), vec![pi(3, 0, 0, 1)]);
    }

    #[test]
    fn compress_destroy_clears_bit() {
        let mut v = compressed();
        along_x(&mut v, 0, 3, 0, 0, 1.0);
        // Passes through (3,0,0): 1 -> 0, destroyed. Ends in (7,0,0): constructed.
        let result = along_x(&mut v, 0, 7, 0, 0, 1.0);
        assert_eq!(get_voxel(&v, 3, 0, 0), Some(0));
        assert_eq!(get_voxel(&v, 7, 0, 0), Some(1));
        assert_eq!(compressed_word(&v, 3, 0, 0), Some(0));
        assert_eq!(compressed_word(&v, 7, 0, 0), Some(page_bit(7, 0, 0)));
        assert_eq!(compressed_bits(&v), vec![pi(7, 0, 0, 1)]);
        assert_eq!(
            sorted(result),
            sorted(vec![block(3, 0, 0, 0), block(7, 0, 0, block_bit(7, 0, 0))])
        );
    }

    #[test]
    fn compress_construct_threshold() {
        let mut v = SingleThreaded::new(1.0, 1, 1, 0, 3, true);
        assert!(along_x(&mut v, 0, 3, 0, 0, 1.0).is_empty());
        assert_eq!(get_voxel(&v, 3, 0, 0), Some(1));
        assert_eq!(compressed_word(&v, 3, 0, 0), None);
        assert!(along_x(&mut v, 0, 3, 0, 0, 1.0).is_empty());
        assert_eq!(get_voxel(&v, 3, 0, 0), Some(2));
        assert_eq!(compressed_word(&v, 3, 0, 0), None);
        let third = along_x(&mut v, 0, 3, 0, 0, 1.0);
        assert_eq!(get_voxel(&v, 3, 0, 0), Some(3));
        assert_eq!(compressed_word(&v, 3, 0, 0), Some(page_bit(3, 0, 0)));
        assert_eq!(third, vec![block(3, 0, 0, block_bit(3, 0, 0))]);

        // Jumping over the threshold in one call constructs as well.
        let mut v = SingleThreaded::new(1.0, 1, 1, 0, 3, true);
        let result = along_x(&mut v, 0, 3, 0, 0, 7.0);
        assert_eq!(result, vec![block(3, 0, 0, block_bit(3, 0, 0))]);
        assert_eq!(compressed_bits(&v), vec![pi(3, 0, 0, 1)]);
    }

    #[test]
    fn compress_destroy_threshold() {
        let mut v = SingleThreaded::new(1.0, 1, 1, 2, 5, true);
        assert_eq!(
            along_x(&mut v, 0, 3, 0, 0, 5.0),
            vec![block(3, 0, 0, block_bit(3, 0, 0))]
        );
        // 5 -> 4 stays above the destroy threshold; (7,0,0) -> 1 stays below
        // the construct threshold.
        assert!(along_x(&mut v, 0, 7, 0, 0, 1.0).is_empty());
        assert_eq!(get_voxel(&v, 3, 0, 0), Some(4));
        assert_eq!(compressed_word(&v, 3, 0, 0), Some(page_bit(3, 0, 0)));
        // 4 -> 2 reaches the destroy threshold.
        let result = along_x(&mut v, 0, 7, 0, 0, 2.0);
        assert_eq!(get_voxel(&v, 3, 0, 0), Some(2));
        assert_eq!(get_voxel(&v, 7, 0, 0), Some(3));
        assert_eq!(compressed_word(&v, 3, 0, 0), Some(0));
        assert!(compressed_bits(&v).is_empty());
        assert_eq!(result, vec![block(3, 0, 0, 0)]);
        // 2 -> 1 is already below: no second destroy.
        assert!(along_x(&mut v, 0, 7, 0, 0, 1.0).is_empty());
        assert_eq!(compressed_word(&v, 3, 0, 0), Some(0));
    }

    #[test]
    fn compress_net_zero_within_call() {
        let mut v = compressed();
        // First ray constructs (1,0,0); second ray hits it back to 0 and
        // constructs (2,0,0). Only the net change counts.
        let result = v.add_points(
            pf(0.5, 0.5, 0.5, 1.0),
            &[pf(1.5, 0.5, 0.5, 1.0), pf(2.5, 0.5, 0.5, 1.0)],
        );
        assert_eq!(get_voxel(&v, 1, 0, 0), Some(0));
        assert_eq!(get_voxel(&v, 2, 0, 0), Some(1));
        assert_eq!(compressed_word(&v, 1, 0, 0), Some(page_bit(2, 0, 0)));
        assert_eq!(compressed_bits(&v), vec![pi(2, 0, 0, 1)]);
        assert_eq!(result, vec![block(2, 0, 0, block_bit(2, 0, 0))]);
    }

    #[test]
    fn compress_bit_positions() {
        for x in 0..4 {
            for y in 0..4 {
                for z in 0..4 {
                    let mut v = compressed();
                    let result = from_above(&mut v, (x, y, z), 1.0);
                    assert_eq!(
                        compressed_word(&v, x, y, z),
                        Some(1 << (x + 4 * y + 16 * z)),
                        "voxel ({x}, {y}, {z})"
                    );
                    assert_eq!(
                        compressed_bits(&v),
                        vec![pi(x, y, z, 1)],
                        "voxel ({x}, {y}, {z})"
                    );
                    assert_eq!(
                        result,
                        vec![pi(0, 0, z & !1, 1 << (x + 4 * y + 16 * (z & 1)))],
                        "voxel ({x}, {y}, {z})"
                    );
                    assert_eq!(result, vec![block(x, y, z, block_bit(x, y, z))]);
                }
            }
        }
    }

    #[test]
    fn compress_both_halves_same_page() {
        let mut v = compressed();
        // Sensor in (0,0,1); one ray down into (0,0,0), one up into (0,0,2).
        let result = v.add_points(
            pf(0.5, 0.5, 1.5, 1.0),
            &[pf(0.5, 0.5, 0.5, 1.0), pf(0.5, 0.5, 2.5, 1.0)],
        );
        assert_eq!(
            compressed_word(&v, 0, 0, 0),
            Some(page_bit(0, 0, 0) | page_bit(0, 0, 2))
        );
        assert_eq!(compressed_bits(&v), vec![pi(0, 0, 0, 1), pi(0, 0, 2, 1)]);
        assert_eq!(
            sorted(result),
            vec![
                block(0, 0, 0, block_bit(0, 0, 0)),
                block(0, 0, 2, block_bit(0, 0, 2))
            ]
        );
    }

    #[test]
    fn compress_two_voxels_same_block() {
        let mut v = compressed();
        let result = v.add_points(
            pf(0.5, 0.5, 1.5, 1.0),
            &[pf(0.5, 0.5, 0.5, 1.0), pf(1.5, 1.5, 1.5, 1.0)],
        );
        assert_eq!(
            compressed_word(&v, 0, 0, 0),
            Some(page_bit(0, 0, 0) | page_bit(1, 1, 1))
        );
        assert_eq!(compressed_bits(&v), vec![pi(0, 0, 0, 1), pi(1, 1, 1, 1)]);
        // One block, both bits, reported once.
        assert_eq!(
            result,
            vec![block(0, 0, 0, block_bit(0, 0, 0) | block_bit(1, 1, 1))]
        );
        assert_eq!(result, vec![pi(0, 0, 0, 1 | 1 << 21)]);
    }

    #[test]
    fn compress_low_half_change_with_high_bits_present() {
        let mut v = compressed();
        let first = from_above(&mut v, (0, 0, 3), 1.0);
        assert_eq!(first, vec![block(0, 0, 3, block_bit(0, 0, 3))]);
        assert_eq!(first, vec![pi(0, 0, 2, 1 << 16)]);
        // A low half change must report the low half, not the (non-zero) high one.
        let second = from_above(&mut v, (1, 0, 0), 1.0);
        assert_eq!(
            compressed_word(&v, 0, 0, 0),
            Some(page_bit(0, 0, 3) | page_bit(1, 0, 0))
        );
        assert_eq!(second, vec![block(1, 0, 0, block_bit(1, 0, 0))]);
        assert_eq!(second, vec![pi(0, 0, 0, 1 << 1)]);
    }

    #[test]
    fn compress_page_keys() {
        let mut v = compressed();
        assert_eq!(
            along_x(&mut v, 3, 4, 0, 0, 1.0),
            vec![block(4, 0, 0, block_bit(4, 0, 0))]
        );
        assert_eq!(
            along_x(&mut v, 63, 64, 0, 0, 1.0),
            vec![block(64, 0, 0, block_bit(64, 0, 0))]
        );
        assert_eq!(
            from_above(&mut v, (-1, -1, -1), 1.0),
            vec![block(-1, -1, -1, block_bit(-1, -1, -1))]
        );
        assert_eq!(
            block(-1, -1, -1, block_bit(-1, -1, -1)),
            pi(-4, -4, -2, 1 << 31)
        );

        let map = v.compressed_voxels.as_ref().unwrap();
        assert_eq!(map.len(), 3);
        // Fine page (1,0,0) lives in compressed page (0,0,0) at z-order [1,0,0].
        let inner = v.zorder.index_of([1u16, 0, 0]) as usize;
        assert_eq!(map[&(0, 0, 0)][inner], page_bit(4, 0, 0));
        // Fine page (16,0,0) is the first slot of compressed page (1,0,0).
        assert_eq!(map[&(1, 0, 0)][0], page_bit(64, 0, 0));
        // Fine page (-1,-1,-1) is the last slot of compressed page (-1,-1,-1).
        assert_eq!(v.zorder.index_of([15u16, 15, 15]), 4095);
        assert_eq!(map[&(-1, -1, -1)][4095], page_bit(-1, -1, -1));
        assert_eq!(
            compressed_bits(&v),
            sorted(vec![pi(4, 0, 0, 1), pi(64, 0, 0, 1), pi(-1, -1, -1, 1)])
        );
    }

    #[test]
    fn compress_block_coords_far() {
        let mut v = compressed();
        let result = from_above(&mut v, (5, 9, 7), 1.0);
        assert_eq!(result, vec![block(5, 9, 7, block_bit(5, 9, 7))]);
        assert_eq!(result, vec![pi(4, 8, 6, 1 << 21)]);
        assert_eq!(compressed_word(&v, 5, 9, 7), Some(1 << 53));

        let result = from_above(&mut v, (-5, -9, -6), 1.0);
        assert_eq!(result, vec![block(-5, -9, -6, block_bit(-5, -9, -6))]);
        assert_eq!(result, vec![pi(-8, -12, -6, 1 << 15)]);
        assert_eq!(compressed_word(&v, -5, -9, -6), Some(1 << 47));

        assert_eq!(
            compressed_bits(&v),
            sorted(vec![pi(5, 9, 7, 1), pi(-5, -9, -6, 1)])
        );
    }

    #[test]
    fn compress_mask_accumulates_across_calls() {
        let mut v = compressed();
        assert_eq!(from_above(&mut v, (0, 0, 0), 1.0), vec![pi(0, 0, 0, 1)]);
        // The pack is the block's full current state, not a delta.
        assert_eq!(from_above(&mut v, (1, 0, 0), 1.0), vec![pi(0, 0, 0, 0b11)]);
        assert_eq!(
            from_above(&mut v, (2, 0, 1), 1.0),
            vec![pi(0, 0, 0, 0b11 | 1 << 18)]
        );
        assert_eq!(
            compressed_word(&v, 0, 0, 0),
            Some(page_bit(0, 0, 0) | page_bit(1, 0, 0) | page_bit(2, 0, 1))
        );
        assert_eq!(
            compressed_bits(&v),
            vec![pi(0, 0, 0, 1), pi(1, 0, 0, 1), pi(2, 0, 1, 1)]
        );
    }

    #[test]
    fn compress_result_map_cleared_after_call() {
        let mut v = compressed();
        along_x(&mut v, 0, 3, 0, 0, 1.0);
        assert!(v.result.is_empty());
        along_x(&mut v, 0, 7, 0, 0, 1.0);
        assert!(v.result.is_empty());
    }

    #[test]
    fn compress_false_leaves_none() {
        let mut v = plain();
        along_x(&mut v, 0, 3, 0, 0, 1.0);
        along_x(&mut v, 0, 7, 0, 0, 1.0);
        assert!(v.compressed_voxels.is_none());
    }

    #[test]
    #[should_panic(expected = "It must have existed")]
    fn compress_destroy_without_construct_panics() {
        // Construct threshold high enough that the target does not create the
        // compressed page, so the destroy of the seeded voxel finds nothing.
        let mut v = SingleThreaded::new(1.0, 1, 1, 0, 10, true);
        set_voxel(&mut v, 0, 0, 0, 5);
        along_x(&mut v, 0, 3, 0, 0, 5.0);
    }
}
