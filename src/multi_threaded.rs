use rayon::{
    ThreadPool, ThreadPoolBuildError, ThreadPoolBuilder,
    iter::{IntoParallelRefIterator, ParallelBridge, ParallelIterator},
};
use rustc_hash::FxBuildHasher;

use crate::{
    Multipliers, Voxelizer, VoxelizerParams,
    geometry::{PointF, PointI, ZOrder},
};

const BITS_FOR_ZORDER: u32 = 2;
const BITS_FOR_ZORDER_MASK: i32 = 2_i32.pow(BITS_FOR_ZORDER) - 1;
const PAGE_SIZE: u32 = 4096;
const BITS_FOR_COMPRESSED_ZORDER: u32 = PAGE_SIZE.highest_one().unwrap() / 3;
const BITS_FOR_COMPRESSED_ZORDER_MASK: i32 = 2_i32.pow(BITS_FOR_COMPRESSED_ZORDER) - 1;

type DashMap<K, V> = dashmap::DashMap<K, V, FxBuildHasher>;
type DashSet<K> = dashmap::DashSet<K, FxBuildHasher>;
type ThreeDim = (i32, i32, i32);
type PagedIndex = (ThreeDim, usize);

#[derive(Debug)]
pub struct MultiThreaded {
    voxels: DashMap<ThreeDim, [u32; 2_usize.pow(3 * BITS_FOR_ZORDER)]>,
    compressed_voxels:
        Option<DashMap<ThreeDim, [u64; 2_usize.pow(3 * BITS_FOR_COMPRESSED_ZORDER)]>>,
    result: DashMap<PagedIndex, u32>,
    // Pages a voxel of which got zeroed during the current call
    maybe_empty: DashSet<ThreeDim>,
    resolution: f32,
    multipliers: Multipliers,
    destroy_threshold: u32,
    construct_threshold: u32,
    max_value: u32,
    zorder: ZOrder,
    thread_pool: ThreadPool,
}

impl MultiThreaded {
    #[inline]
    pub fn new(params: VoxelizerParams, thread_count: usize) -> Result<Self, ThreadPoolBuildError> {
        let VoxelizerParams {
            resolution,
            multipliers,
            destroy_threshold,
            construct_threshold,
            max_value,
            compress,
        } = params;
        Ok(Self {
            voxels: DashMap::default(),
            compressed_voxels: if compress {
                Some(DashMap::default())
            } else {
                None
            },
            result: DashMap::default(),
            maybe_empty: DashSet::default(),
            resolution,
            multipliers,
            destroy_threshold,
            construct_threshold,
            max_value,
            zorder: ZOrder::default(),
            thread_pool: ThreadPoolBuilder::new().num_threads(thread_count).build()?,
        })
    }
}

impl Voxelizer for MultiThreaded {
    fn add_points(&mut self, mut sensor_pos: PointF, points: &[PointF]) -> Vec<PointI> {
        sensor_pos /= self.resolution;
        let sensor_pos_i = PointI {
            x: sensor_pos.x.floor() as i32,
            y: sensor_pos.y.floor() as i32,
            z: sensor_pos.z.floor() as i32,
            i: sensor_pos.i as u32,
        };
        self.thread_pool.install(|| {
            points
                .par_iter()
                .for_each(|v| self.ray_march(sensor_pos_i.clone(), &sensor_pos, v.clone()));
            let resolution = self.resolution;
            let intensity_multiplier = self.multipliers.intensity;
            points
                .par_iter()
                .map(move |v| v / resolution)
                .map(|v| PointI {
                    x: v.x.floor() as i32,
                    y: v.y.floor() as i32,
                    z: v.z.floor() as i32,
                    i: (v.i * intensity_multiplier) as u32,
                })
                .for_each(|point| self.add_voxel(&point));
        });
        let map = std::mem::take(&mut self.result);
        // Compressed pages a word of which got zeroed
        let emptied_compressed = DashSet::default();
        let result = self.thread_pool.install(|| {
            let iter = map
                .into_iter()
                .par_bridge()
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
            if let Some(compressed_voxels) = self.compressed_voxels.as_ref() {
                let compressed_dedup = DashMap::default();
                iter.for_each(|(idx, prev, voxel)| {
                    let [x, y, z] = self.zorder.coord_of(idx.1 as u64);
                    let idx2 = || {
                        self.zorder.index_of([
                            (idx.0.0 & BITS_FOR_COMPRESSED_ZORDER_MASK) as u16,
                            (idx.0.1 & BITS_FOR_COMPRESSED_ZORDER_MASK) as u16,
                            (idx.0.2 & BITS_FOR_COMPRESSED_ZORDER_MASK) as u16,
                        ]) as usize
                    };
                    let changed = |v| {
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
                        let v =
                            &mut compressed_voxels.entry(idx()).or_insert_with(|| [0; _])[idx2()];
                        if *v & bit == 0 {
                            *v |= bit;
                            changed(*v);
                        }
                    } else if prev > self.destroy_threshold
                        && voxel <= self.destroy_threshold
                        && let Some(mut words) = compressed_voxels.get_mut(&idx())
                        && words[idx2()] & bit != 0
                    {
                        let v = &mut words[idx2()];
                        *v &= !bit;
                        changed(*v);
                        if *v == 0 {
                            emptied_compressed.insert(idx());
                        }
                    }
                });
                compressed_dedup
                    .into_iter()
                    .map(|((x, y, z), i)| PointI { x, y, z, i })
                    .collect()
            } else {
                iter.map(|(idx, _, voxel)| self.point_from_index(idx, voxel))
                    .collect()
            }
        });
        self.remove_emptied(emptied_compressed);
        result
    }
    #[inline]
    fn set_multipliers(&mut self, multipliers: Multipliers) {
        self.multipliers = multipliers;
    }
    #[inline]
    fn multipliers(&self) -> Multipliers {
        self.multipliers
    }
}

impl MultiThreaded {
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
    fn hit_voxel(&self, point: &PointI) {
        let idx = self.get_index(point);
        // If voxel is not hit it does not exists anyway
        if let Some(mut v) = self.voxels.get_mut(&idx.0) {
            use dashmap::Entry;
            match self.result.entry(idx) {
                Entry::Occupied(mut occupied_entry) => {
                    if *occupied_entry.get() < v[idx.1] {
                        *occupied_entry.get_mut() = v[idx.1];
                    }
                }
                Entry::Vacant(vacant_entry) => {
                    vacant_entry.insert(v[idx.1]);
                }
            }
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
    fn remove_emptied(&mut self, emptied_compressed: DashSet<ThreeDim>) {
        for page in std::mem::take(&mut self.maybe_empty) {
            let (key, inner) = self.compressed_index(page);
            let constructed = self
                .compressed_voxels
                .as_ref()
                .and_then(|compressed_voxels| compressed_voxels.get(&key))
                .is_some_and(|words| words[inner] != 0);
            if !constructed {
                self.voxels
                    .remove_if(&page, |_, voxels| voxels.iter().all(|&v| v == 0));
            }
        }
        if let Some(compressed_voxels) = self.compressed_voxels.as_ref() {
            for key in emptied_compressed {
                compressed_voxels.remove_if(&key, |_, words| words.iter().all(|&w| w == 0));
            }
        }
    }
    #[inline]
    fn add_voxel(&self, point: &PointI) {
        let idx = self.get_index(point);
        let mut entry = self.voxels.entry(idx.0).or_insert_with(|| [0; _]);
        self.result.entry(idx).or_insert(entry[idx.1]);
        entry[idx.1] = entry[idx.1]
            .saturating_add(point.i * self.multipliers.target)
            .min(self.max_value);
    }
    fn ray_march(&self, mut sensor_pos: PointI, sensor_pos_divided: &PointF, mut point: PointF) {
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
    use super::{
        BITS_FOR_COMPRESSED_ZORDER, BITS_FOR_COMPRESSED_ZORDER_MASK, BITS_FOR_ZORDER,
        MultiThreaded, PAGE_SIZE,
    };
    use crate::{
        Multipliers, Voxelizer, VoxelizerParams,
        geometry::PointI,
        test::{
            Coord, PAGE_LEN, PLAIN, PagedIndex, SCENARIO_PARAMS, SWITCHED, TestVoxelizer,
            assert_snapshot_eq, compressed_bits, compressed_word, cube_faces, get_voxel,
            nonzero_voxels, page_bit, pf, pi, scenario_with, seed, sorted, with_i,
        },
    };

    /// Thread count used by the shared suite.
    const THREADS: usize = 4;

    /// Without a cap on voxel values or scaling of intensities.
    fn new(
        resolution: f32,
        hit_multiplier: u32,
        target_multiplier: u32,
        destroy_threshold: u32,
        construct_threshold: u32,
        compress: bool,
        thread_count: usize,
    ) -> MultiThreaded {
        MultiThreaded::new(
            VoxelizerParams {
                resolution,
                multipliers: Multipliers {
                    intensity: 1.0,
                    hit: hit_multiplier,
                    target: target_multiplier,
                },
                destroy_threshold,
                construct_threshold,
                max_value: u32::MAX,
                compress,
            },
            thread_count,
        )
        .expect("thread pool must build")
    }

    impl TestVoxelizer for MultiThreaded {
        const BITS_FOR_ZORDER: u32 = BITS_FOR_ZORDER;
        const BITS_FOR_COMPRESSED_ZORDER: u32 = BITS_FOR_COMPRESSED_ZORDER;
        const BITS_FOR_COMPRESSED_ZORDER_MASK: i32 = BITS_FOR_COMPRESSED_ZORDER_MASK;
        const PAGE_SIZE: u32 = PAGE_SIZE;
        fn with_params(params: VoxelizerParams) -> Self {
            MultiThreaded::new(params, THREADS).expect("thread pool must build")
        }
        fn get_index(&self, point: &PointI) -> PagedIndex {
            MultiThreaded::get_index(self, point)
        }
        fn point_from_index(&self, idx: PagedIndex, i: u32) -> PointI {
            MultiThreaded::point_from_index(self, idx, i)
        }
        fn closest_boundary(upper: bool, position: f32, len: f32) -> f32 {
            MultiThreaded::closest_boundary(upper, position, len)
        }
        fn set_voxel(&mut self, x: i32, y: i32, z: i32, value: u32) {
            let idx = MultiThreaded::get_index(self, &pi(x, y, z, 0));
            self.voxels.entry(idx.0).or_insert_with(|| [0; PAGE_LEN])[idx.1] = value;
        }
        fn pages(&self) -> Vec<(Coord, [u32; PAGE_LEN])> {
            let mut out: Vec<_> = self.voxels.iter().map(|r| (*r.key(), *r.value())).collect();
            out.sort_by_key(|(k, _)| *k);
            out
        }
        fn compressed_pages(&self) -> Option<Vec<(Coord, Vec<u64>)>> {
            let mut out: Vec<_> = self
                .compressed_voxels
                .as_ref()?
                .iter()
                .map(|r| (*r.key(), r.value().to_vec()))
                .collect();
            out.sort_by_key(|(k, _)| *k);
            Some(out)
        }
        fn pending_results(&self) -> usize {
            self.result.len() + self.maybe_empty.len()
        }
    }

    crate::test::shared_tests!(MultiThreaded);

    // ------------------------------------------------------------ constructor

    #[test]
    fn new_stores_parameters() {
        let params = VoxelizerParams {
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
        };
        let v = MultiThreaded::new(params, 3).expect("thread pool must build");
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
        assert_eq!(v.thread_pool.current_num_threads(), 3);
        assert!(v.voxels.is_empty());
        assert!(v.result.is_empty());
        let v = new(1.0, 1, 1, 0, 1, false, 1);
        assert_eq!(v.thread_pool.current_num_threads(), 1);
    }

    #[test]
    fn new_thread_count_zero_uses_default() {
        let v = MultiThreaded::new(PLAIN, 0).expect("0 means default");
        assert!(v.thread_pool.current_num_threads() >= 1);
    }

    #[test]
    fn new_compress_flag() {
        assert!(new(1.0, 1, 1, 0, 1, false, 1).compressed_voxels.is_none());
        let v = new(1.0, 1, 1, 0, 1, true, 1);
        assert!(v.compressed_voxels.as_ref().is_some_and(|m| m.is_empty()));
    }

    // ------------------------------------------------------------ concurrency

    #[test]
    fn same_result_for_any_thread_count() {
        let make =
            |params, threads| MultiThreaded::new(params, threads).expect("thread pool must build");
        for switch in [None, Some(SWITCHED)] {
            for params in SCENARIO_PARAMS {
                let expected = scenario_with(make(params.clone(), 1), switch);
                assert!(expected.results.iter().all(|r| !r.is_empty()));
                for threads in [2, 4, 8, 16] {
                    let got = scenario_with(make(params.clone(), threads), switch);
                    assert_snapshot_eq(
                        &got,
                        &expected,
                        &format!("{params:?}, switch = {switch:?}, threads = {threads}"),
                    );
                }
            }
        }
    }

    #[test]
    fn concurrent_same_target() {
        // Every thread adds to the same target and subtracts from the same
        // three voxels on the way: no update may get lost and the result
        // must hold the final values once.
        const N: u32 = 10_000;
        let mut v = new(1.0, 1, 1, 0, 1, false, 8);
        let hits = [(0, 0, 0), (1, 0, 0), (2, 0, 0)];
        seed(&mut v, &hits, 2 * N);
        let points = vec![pf(3.5, 0.5, 0.5, 1.0); N as usize];
        let result = v.add_points(pf(0.5, 0.5, 0.5, 1.0), &points);
        let mut expected = with_i(&hits, N);
        expected.push(pi(3, 0, 0, N));
        let expected = sorted(expected);
        assert_eq!(sorted(result), expected);
        assert_eq!(nonzero_voxels(&v), expected);
        assert_eq!(v.pending_results(), 0);
    }

    #[test]
    fn concurrent_distinct_pages() {
        // Targets 8 apart on the faces of a cube: each in its own fine page,
        // none on the path of another ray from the origin.
        let mut v = new(1.0, 1, 1, 0, 1, false, 8);
        let targets = cube_faces(40, 32, 8);
        let intensity = |n: usize| (1 + n % 7) as u32;
        let points: Vec<_> = targets
            .iter()
            .enumerate()
            .map(|(n, &(x, y, z))| {
                pf(
                    x as f32 + 0.5,
                    y as f32 + 0.5,
                    z as f32 + 0.5,
                    intensity(n) as f32,
                )
            })
            .collect();
        let result = v.add_points(pf(0.5, 0.5, 0.5, 1.0), &points);
        let expected = sorted(
            targets
                .iter()
                .enumerate()
                .map(|(n, &(x, y, z))| pi(x, y, z, intensity(n)))
                .collect(),
        );
        assert_eq!(sorted(result), expected);
        assert_eq!(nonzero_voxels(&v), expected);
        // Only the targets' pages exist: nothing else got written.
        assert_eq!(v.pages().len(), targets.len());
        assert_eq!(v.pending_results(), 0);
    }

    #[test]
    fn concurrent_compress_fills_block() {
        // From high above every ray enters the z = 1 layer already inside its
        // target column, so no target lies on another ray's path. 16 voxels
        // per block, each sent twice, must merge into one entry per block.
        let mut v = new(1.0, 1, 1, 0, 1, true, 8);
        let blocks = [(0, 0), (4, 4)];
        let mut points = Vec::new();
        let mut voxels = Vec::new();
        for (bx, by) in blocks {
            for x in 0..4 {
                for y in 0..4 {
                    let p = pf((bx + x) as f32 + 0.5, (by + y) as f32 + 0.5, 1.5, 1.0);
                    points.push(p.clone());
                    points.push(p);
                    voxels.push((bx + x, by + y, 1));
                }
            }
        }
        let result = v.add_points(pf(1.5, 1.5, 20.5, 1.0), &points);
        // Whole z = 1 layer of a block: bits 16..32 of the low word half.
        const LAYER: u32 = 0xFFFF_0000;
        assert_eq!(sorted(result), vec![pi(0, 0, 0, LAYER), pi(4, 4, 0, LAYER)]);
        assert_eq!(compressed_word(&v, 0, 0, 1), Some(LAYER as u64));
        assert_eq!(compressed_word(&v, 4, 4, 1), Some(LAYER as u64));
        assert_eq!(compressed_bits(&v), sorted(with_i(&voxels, 1)));
        assert_eq!(nonzero_voxels(&v), sorted(with_i(&voxels, 2)));
        assert_eq!(v.pages().len(), 2);
    }

    #[test]
    fn compress_net_zero_within_call_consistent() {
        // Same input as the sequential `compress_net_zero_within_call`: ray 1
        // constructs (1,0,0), ray 2 passes through it into (2,0,0). Whether
        // the hit lands before the add (or before the page even exists) or
        // after it depends on scheduling, so both outcomes are valid, but the
        // reported block must match the stored state either way.
        let mut v = new(1.0, 1, 1, 0, 1, true, 4);
        let result = v.add_points(
            pf(0.5, 0.5, 0.5, 1.0),
            &[pf(1.5, 0.5, 0.5, 1.0), pf(2.5, 0.5, 0.5, 1.0)],
        );
        assert_eq!(get_voxel(&v, 2, 0, 0), Some(1));
        let one = get_voxel(&v, 1, 0, 0).expect("page exists");
        assert!(one <= 1, "(1,0,0) = {one}");
        let mut word = page_bit(2, 0, 0);
        let mut bits = vec![pi(2, 0, 0, 1)];
        if one == 1 {
            word |= page_bit(1, 0, 0);
            bits.push(pi(1, 0, 0, 1));
        }
        assert_eq!(compressed_word(&v, 0, 0, 0), Some(word));
        assert_eq!(compressed_bits(&v), sorted(bits));
        assert_eq!(result, vec![pi(0, 0, 0, word as u32)]);
    }
}
