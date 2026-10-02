use dashmap::DashMap;
use rayon::{
    ThreadPool, ThreadPoolBuildError, ThreadPoolBuilder,
    iter::{IntoParallelRefIterator, ParallelBridge, ParallelIterator},
};

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

#[derive(Debug)]
pub struct MultithreadedThreaded {
    voxels: DashMap<ThreeDim, [u32; 2_usize.pow(3 * BITS_FOR_ZORDER)]>,
    compressed_voxels:
        Option<DashMap<ThreeDim, [u64; 2_usize.pow(3 * BITS_FOR_COMPRESSED_ZORDER)]>>,
    result: DashMap<PagedIndex, u32>,
    resolution: f32,
    hit_multiplayer: u32,
    target_multiplayer: u32,
    destroy_threshold: u32,
    construct_threshold: u32,
    zorder: ZOrder,
    thread_pool: ThreadPool,
}

impl MultithreadedThreaded {
    #[inline]
    pub fn new(
        resolution: f32,
        hit_multiplayer: u32,
        target_multiplayer: u32,
        destroy_threshold: u32,
        construct_threshold: u32,
        compress: bool,
        thread_count: usize,
    ) -> Result<Self, ThreadPoolBuildError> {
        Ok(Self {
            voxels: DashMap::new(),
            compressed_voxels: if compress { Some(DashMap::new()) } else { None },
            result: DashMap::new(),
            resolution,
            hit_multiplayer,
            target_multiplayer,
            destroy_threshold,
            construct_threshold,
            zorder: ZOrder::default(),
            thread_pool: ThreadPoolBuilder::new().num_threads(thread_count).build()?,
        })
    }
}

impl Voxelizer for MultithreadedThreaded {
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
            points
                .par_iter()
                .map(move |v| v / resolution)
                .map(|v| PointI {
                    x: v.x.floor() as i32,
                    y: v.y.floor() as i32,
                    z: v.z.floor() as i32,
                    i: v.i as u32,
                })
                .for_each(|point| self.add_voxel(&point));
        });
        let map = std::mem::replace(&mut self.result, DashMap::new());
        self.thread_pool.install(|| {
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
                let compressed_dedup = DashMap::new();
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
                });
                compressed_dedup
                    .into_iter()
                    .map(|((x, y, z), i)| PointI { x, y, z, i })
                    .collect()
            } else {
                iter.map(|(idx, _, voxel)| self.point_from_index(idx, voxel))
                    .collect()
            }
        })
    }
}

impl MultithreadedThreaded {
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
            v[idx.1] = v[idx.1].saturating_sub(point.i * self.hit_multiplayer);
        }
    }
    #[inline]
    fn add_voxel(&self, point: &PointI) {
        let idx = self.get_index(point);
        let mut entry = self.voxels.entry(idx.0).or_insert_with(|| [0; _]);
        self.result.entry(idx).or_insert(entry[idx.1]);
        entry[idx.1] += point.i * self.target_multiplayer;
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
    }
}

#[cfg(test)]
mod test {
    use super::{
        BITS_FOR_COMPRESSED_ZORDER, BITS_FOR_COMPRESSED_ZORDER_MASK, BITS_FOR_ZORDER,
        MultithreadedThreaded, PAGE_SIZE,
    };
    use crate::{
        Voxelizer,
        geometry::PointI,
        test::{
            Coord, PAGE_LEN, PagedIndex, SCENARIO_PARAMS, TestVoxelizer, assert_snapshot_eq,
            compressed_bits, compressed_word, cube_faces, get_voxel, nonzero_voxels, page_bit, pf,
            pi, scenario, seed, sorted, with_i,
        },
    };

    /// Thread count used by the shared suite.
    const THREADS: usize = 4;

    fn new(
        resolution: f32,
        hit_multiplayer: u32,
        target_multiplayer: u32,
        destroy_threshold: u32,
        construct_threshold: u32,
        compress: bool,
        thread_count: usize,
    ) -> MultithreadedThreaded {
        MultithreadedThreaded::new(
            resolution,
            hit_multiplayer,
            target_multiplayer,
            destroy_threshold,
            construct_threshold,
            compress,
            thread_count,
        )
        .expect("thread pool must build")
    }

    impl TestVoxelizer for MultithreadedThreaded {
        const BITS_FOR_ZORDER: u32 = BITS_FOR_ZORDER;
        const BITS_FOR_COMPRESSED_ZORDER: u32 = BITS_FOR_COMPRESSED_ZORDER;
        const BITS_FOR_COMPRESSED_ZORDER_MASK: i32 = BITS_FOR_COMPRESSED_ZORDER_MASK;
        const PAGE_SIZE: u32 = PAGE_SIZE;
        fn make(
            resolution: f32,
            hit_multiplayer: u32,
            target_multiplayer: u32,
            destroy_threshold: u32,
            construct_threshold: u32,
            compress: bool,
        ) -> Self {
            new(
                resolution,
                hit_multiplayer,
                target_multiplayer,
                destroy_threshold,
                construct_threshold,
                compress,
                THREADS,
            )
        }
        fn get_index(&self, point: &PointI) -> PagedIndex {
            MultithreadedThreaded::get_index(self, point)
        }
        fn point_from_index(&self, idx: PagedIndex, i: u32) -> PointI {
            MultithreadedThreaded::point_from_index(self, idx, i)
        }
        fn closest_boundary(upper: bool, position: f32, len: f32) -> f32 {
            MultithreadedThreaded::closest_boundary(upper, position, len)
        }
        fn set_voxel(&mut self, x: i32, y: i32, z: i32, value: u32) {
            let idx = MultithreadedThreaded::get_index(self, &pi(x, y, z, 0));
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
            self.result.len()
        }
    }

    crate::test::shared_tests!(MultithreadedThreaded);

    // ------------------------------------------------------------ constructor

    #[test]
    fn new_stores_parameters() {
        let v = new(0.5, 2, 3, 4, 5, false, 3);
        assert_eq!(v.resolution, 0.5);
        assert_eq!(v.hit_multiplayer, 2);
        assert_eq!(v.target_multiplayer, 3);
        assert_eq!(v.destroy_threshold, 4);
        assert_eq!(v.construct_threshold, 5);
        assert_eq!(v.thread_pool.current_num_threads(), 3);
        assert!(v.voxels.is_empty());
        assert!(v.result.is_empty());
        let v = new(1.0, 1, 1, 0, 1, false, 1);
        assert_eq!(v.thread_pool.current_num_threads(), 1);
    }

    #[test]
    fn new_thread_count_zero_uses_default() {
        let v = MultithreadedThreaded::new(1.0, 1, 1, 0, 1, false, 0).expect("0 means default");
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
        for (destroy, construct, compress) in SCENARIO_PARAMS {
            let expected = scenario(new(1.0, 1, 1, destroy, construct, compress, 1));
            assert!(expected.results.iter().all(|r| !r.is_empty()));
            for threads in [2, 4, 8, 16] {
                let got = scenario(new(1.0, 1, 1, destroy, construct, compress, threads));
                assert_snapshot_eq(
                    &got,
                    &expected,
                    &format!("compress = {compress}, threads = {threads}"),
                );
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
