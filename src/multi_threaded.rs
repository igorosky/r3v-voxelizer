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
        });
        let mut result = Vec::new();
        let compressed_dedup = DashMap::new();
        let iter = std::mem::replace(&mut self.result, DashMap::new())
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
            iter.for_each(|(idx, prev, voxel)| {
                let [x, y, z] = self.zorder.coord_of(idx.1 as u64);
                let idx2 = || {
                    self.zorder.index_of([
                        (idx.0.0 & BITS_FOR_COMPRESSED_ZORDER_MASK) as u16,
                        (idx.0.1 & BITS_FOR_COMPRESSED_ZORDER_MASK) as u16,
                        (idx.0.2 & BITS_FOR_COMPRESSED_ZORDER_MASK) as u16,
                    ]) as usize
                };
                let changed = |v: u64| {
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
                }
            });
            result.extend(
                compressed_dedup
                    .into_iter()
                    .map(|((x, y, z), i)| PointI { x, y, z, i }),
            );
        } else {
            result = iter
                .map(|(idx, _, voxel)| self.point_from_index(idx, voxel))
                .collect();
        }
        result
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
            self.result.entry(idx).or_insert(v[idx.1]);
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
        self.add_voxel(&sensor_pos);
    }
}
