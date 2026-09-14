use crate::{
    geometry::{PointF, PointI, ZOrder},
    voxelizer::Voxelizer,
};

const BITS_FOR_ZORDER: u32 = 4;
const BITS_FOR_ZORDER_MASK: i32 = 2_i32.pow(BITS_FOR_ZORDER) - 1;

type ThreeDim = (i32, i32, i32);
type PagedIndex = (ThreeDim, usize);

#[derive(Debug, Clone)]
struct SingleThreaded {
    voxels: hashbrown::HashMap<ThreeDim, [PointI; 2_usize.pow(3 * BITS_FOR_ZORDER)]>,
    compressed_voxels: hashbrown::HashMap<ThreeDim, [u64; 2_usize.pow(3 * BITS_FOR_ZORDER)]>,
    result: hashbrown::HashMap<PagedIndex, i32>,
    resolution: f32,
    zorder: ZOrder,
}

impl SingleThreaded {
    #[inline]
    pub(crate) fn new(resolution: f32) -> Self {
        Self {
            voxels: hashbrown::HashMap::new(),
            compressed_voxels: hashbrown::HashMap::new(),
            result: hashbrown::HashMap::new(),
            resolution,
            zorder: ZOrder::new(),
        }
    }
}

impl Voxelizer for SingleThreaded {
    fn add_points(&mut self, mut sensor_pos: PointF, points: &[PointF]) -> Vec<PointI> {
        sensor_pos /= self.resolution;
        let sensor_pos_i = PointI {
            x: sensor_pos.x as i32,
            y: sensor_pos.y as i32,
            z: sensor_pos.z as i32,
            i: sensor_pos.i as i32,
        };
        points
            .iter()
            .for_each(|v| self.ray_march(sensor_pos_i.clone(), &sensor_pos, v.clone()));
        let mut result = Vec::new();
        for (idx, prev) in self.result.drain() {
            let voxel = &self
                .voxels
                .get(&idx.0)
                .expect("This voxel was touched - it must exist")[idx.1];
            if voxel.i != prev {
                result.push(voxel.clone());
            }
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
    fn closest_boundary(upper: bool, position: f32, len: f32) -> f32 {
        let boundary = if upper {
            position.ceil()
        } else {
            position.floor()
        };
        let result = (boundary - position) / len;
        debug_assert!(result >= 0.0);
        result
    }
    #[inline]
    fn hit_voxel(&mut self, point: &PointI) {
        let idx = self.get_index(point);
        // If voxel is not hit it does not exists anyway
        if let Some(v) = self.voxels.get_mut(&idx.0) {
            self.result.entry(idx).or_insert(v[idx.1].i);
            v[idx.1].i -= point.i;
        }
    }
    #[inline]
    fn add_voxel(&mut self, point: &PointI) {
        let idx = self.get_index(point);
        let entry = self.voxels.entry(idx.0).or_insert_with(|| {
            core::array::from_fn(|_| PointI {
                x: point.x,
                y: point.y,
                z: point.z,
                i: 0,
            })
        });
        self.result.entry(idx).or_insert(entry[idx.1].i);
        entry[idx.1].i += point.i << 1; // Times two because it was first subtracted
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
            x: point.x as i32,
            y: point.y as i32,
            z: point.z as i32,
            i: point.i as i32,
        };
        let dir_x = (sensor_pos.x - point.x as i32).signum();
        let dir_y = (sensor_pos.y - point.y as i32).signum();
        let dir_z = (sensor_pos.z - point.z as i32).signum();
        point -= &sensor_pos_divided;
        use std::cmp::Ordering;
        match (dir_x.cmp(&0), dir_y.cmp(&0), dir_z.cmp(&0)) {
            (Ordering::Equal, Ordering::Equal, Ordering::Equal) => todo!(),
            (_, Ordering::Equal, Ordering::Equal) => {
                while target_point.x != sensor_pos.x {
                    sensor_pos.x += dir_x;
                    self.hit_voxel(&sensor_pos);
                }
            }
            (Ordering::Equal, _, Ordering::Equal) => {
                while target_point.y != sensor_pos.y {
                    sensor_pos.y += dir_y;
                    self.hit_voxel(&sensor_pos);
                }
            }
            (Ordering::Equal, Ordering::Equal, _) => {
                while target_point.z != sensor_pos.z {
                    sensor_pos.z += dir_z;
                    self.hit_voxel(&sensor_pos);
                }
            }
            (x, y, Ordering::Equal) => {
                let hypot = point.len();
                let dx = hypot / point.x;
                let dy = hypot / point.y;
                let mut max_x = Self::closest_boundary(x.is_gt(), sensor_pos_divided.x, point.x);
                let mut max_y = Self::closest_boundary(y.is_gt(), sensor_pos_divided.y, point.y);
                while target_point != sensor_pos {
                    debug_assert!(max_x < 1.0);
                    debug_assert!(max_y < 1.0);
                    if max_x < max_y {
                        sensor_pos.x += dir_x;
                        max_x += dx;
                    } else {
                        sensor_pos.y += dir_y;
                        max_y += dy;
                    }
                    self.hit_voxel(&sensor_pos);
                }
            }
            (x, Ordering::Equal, z) => {
                let hypot = point.len();
                let dx = hypot / point.x;
                let dz = hypot / point.z;
                let mut max_x = Self::closest_boundary(x.is_gt(), sensor_pos_divided.x, point.x);
                let mut max_z = Self::closest_boundary(z.is_gt(), sensor_pos_divided.z, point.z);
                while target_point != sensor_pos {
                    debug_assert!(max_x < 1.0);
                    debug_assert!(max_z < 1.0);
                    if max_x < max_z {
                        sensor_pos.x += dir_x;
                        max_x += dx;
                    } else {
                        sensor_pos.z += dir_z;
                        max_z += dz;
                    }
                    self.hit_voxel(&sensor_pos);
                }
            }
            (Ordering::Equal, y, z) => {
                let hypot = point.len();
                let dy = hypot / point.y;
                let dz = hypot / point.z;
                let mut max_y = Self::closest_boundary(y.is_gt(), sensor_pos_divided.y, point.y);
                let mut max_z = Self::closest_boundary(z.is_gt(), sensor_pos_divided.z, point.z);
                while target_point != sensor_pos {
                    debug_assert!(max_y < 1.0);
                    debug_assert!(max_z < 1.0);
                    if max_y < max_z {
                        sensor_pos.y += dir_y;
                        max_y += dy;
                    } else {
                        sensor_pos.z += dir_z;
                        max_z += dz;
                    }
                    self.hit_voxel(&sensor_pos);
                }
            }
            (x, y, z) => {
                let hypot = point.len();
                let dx = hypot / point.x;
                let dy = hypot / point.y;
                let dz = hypot / point.z;
                let mut max_x = Self::closest_boundary(x.is_gt(), sensor_pos_divided.x, point.x);
                let mut max_y = Self::closest_boundary(y.is_gt(), sensor_pos_divided.y, point.y);
                let mut max_z = Self::closest_boundary(z.is_gt(), sensor_pos_divided.z, point.z);
                while target_point != sensor_pos {
                    debug_assert!(max_x < 1.0);
                    debug_assert!(max_y < 1.0);
                    debug_assert!(max_z < 1.0);
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
                    self.hit_voxel(&sensor_pos);
                }
            }
        };
        self.add_voxel(&sensor_pos);
    }
}

#[cfg(test)]
mod test {}
