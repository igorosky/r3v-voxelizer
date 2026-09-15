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
    voxels: hashbrown::HashMap<ThreeDim, [i32; 2_usize.pow(3 * BITS_FOR_ZORDER)]>,
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
            x: sensor_pos.x.floor() as i32,
            y: sensor_pos.y.floor() as i32,
            z: sensor_pos.z.floor() as i32,
            i: sensor_pos.i as i32,
        };
        points
            .iter()
            .for_each(|v| self.ray_march(sensor_pos_i.clone(), &sensor_pos, v.clone()));
        let mut result = Vec::new();
        let mut map = hashbrown::HashMap::new();
        std::mem::swap(&mut map, &mut self.result);
        for (idx, prev) in map.drain() {
            let voxel = self
                .voxels
                .get(&idx.0)
                .expect("This voxel was touched - it must exist")[idx.1];
            if voxel != prev {
                result.push(self.point_from_index(idx, voxel));
            }
        }
        std::mem::swap(&mut map, &mut self.result);
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
    fn point_from_index(&self, idx: PagedIndex, i: i32) -> PointI {
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
        let boundary = if upper {
            position.floor() + 1.0
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
            self.result.entry(idx).or_insert(v[idx.1]);
            v[idx.1] -= point.i;
        }
    }
    #[inline]
    fn add_voxel(&mut self, point: &PointI) {
        let idx = self.get_index(point);
        let entry = self.voxels.entry(idx.0).or_insert_with(|| [0; _]);
        self.result.entry(idx).or_insert(entry[idx.1]);
        entry[idx.1] += point.i;
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
            i: point.i as i32,
        };
        let dir_x = (target_point.x as i32 - sensor_pos.x).signum();
        let dir_y = (target_point.y as i32 - sensor_pos.y).signum();
        let dir_z = (target_point.z as i32 - sensor_pos.z).signum();
        sensor_pos.i = target_point.i;
        point -= &sensor_pos_divided;
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
    use super::BITS_FOR_ZORDER;
    use crate::{
        geometry::{PointF, PointI},
        voxelizer::{Voxelizer, single_threaded::SingleThreaded},
    };

    const PAGE_LEN: usize = 2_usize.pow(3 * BITS_FOR_ZORDER);

    #[inline]
    fn pf(x: f32, y: f32, z: f32, i: f32) -> PointF {
        PointF { x, y, z, i }
    }
    #[inline]
    fn pi(x: i32, y: i32, z: i32, i: i32) -> PointI {
        PointI { x, y, z, i }
    }
    #[inline]
    fn sorted<T: Ord>(mut v: Vec<T>) -> Vec<T> {
        v.sort();
        v
    }
    /// Builds points from coordinate triples sharing one intensity.
    fn with_i(coords: &[(i32, i32, i32)], i: i32) -> Vec<PointI> {
        coords.iter().map(|&(x, y, z)| pi(x, y, z, i)).collect()
    }
    /// Writes a voxel directly, creating its (zero filled) page when missing.
    /// Used to make pages exist so that hits become observable.
    fn set_voxel(v: &mut SingleThreaded, x: i32, y: i32, z: i32, value: i32) {
        let idx = v.get_index(&pi(x, y, z, 0));
        v.voxels.entry(idx.0).or_insert_with(|| [0; PAGE_LEN])[idx.1] = value;
    }
    fn get_voxel(v: &SingleThreaded, x: i32, y: i32, z: i32) -> Option<i32> {
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
    /// Seeds page (0,0,0) with zeros, fires one ray of intensity 1 and checks
    /// that exactly `hit` voxels got -1 and `target` got +1.
    fn check_ray(sensor: PointF, point: PointF, hit: &[(i32, i32, i32)], target: (i32, i32, i32)) {
        let mut v = SingleThreaded::new(1.0);
        set_voxel(&mut v, 0, 0, 0, 0);
        let result = v.add_points(sensor, &[point]);
        let mut expected = with_i(hit, -1);
        expected.push(pi(target.0, target.1, target.2, 1));
        let expected = sorted(expected);
        assert_eq!(sorted(result), expected);
        assert_eq!(nonzero_voxels(&v), expected);
    }

    // ---------------------------------------------------------------- indexing

    #[test]
    fn get_index_page_and_inner() {
        let v = SingleThreaded::new(1.0);
        assert_eq!(v.get_index(&pi(0, 0, 0, 0)), ((0, 0, 0), 0));
        assert_eq!(v.get_index(&pi(17, 33, -1, 0)).0, (1, 2, -1));
        assert_eq!(v.get_index(&pi(15, 15, 15, 0)).0, (0, 0, 0));
        assert_eq!(v.get_index(&pi(16, 16, 16, 0)).0, (1, 1, 1));
        assert_eq!(v.get_index(&pi(-16, -16, -16, 0)).0, (-1, -1, -1));
        assert_eq!(v.get_index(&pi(-17, -17, -17, 0)).0, (-2, -2, -2));
        for c in [-1000, -17, -1, 0, 15, 16, 1000] {
            assert!(v.get_index(&pi(c, c, c, 0)).1 < PAGE_LEN);
        }
        // Same offset inside a page gives the same inner index, whatever the page.
        let inner = v.get_index(&pi(3, 5, 7, 0)).1;
        assert_eq!(v.get_index(&pi(19, 21, 23, 0)).1, inner);
        assert_eq!(v.get_index(&pi(-13, -11, -9, 0)).1, inner);
    }

    #[test]
    fn index_roundtrip() {
        let v = SingleThreaded::new(1.0);
        let coords = [-1000, -17, -16, -15, -1, 0, 1, 15, 16, 17, 1000];
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
        let v = SingleThreaded::new(1.0);
        let mut seen = vec![false; PAGE_LEN];
        for x in 0..16 {
            for y in 0..16 {
                for z in 0..16 {
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

    // -------------------------------------------------------- add_points basics

    #[test]
    fn basic_test_1d() {
        let mut x = SingleThreaded::new(1.0);
        let result = x.add_points(pf(0.0, 0.0, 0.0, 1.0), &[pf(5.0, 0.0, 0.0, 1.0)]);
        assert_eq!(sorted(result), sorted(vec![pi(5, 0, 0, 1)]));
    }

    #[test]
    fn empty_points_no_change() {
        let mut v = SingleThreaded::new(1.0);
        assert!(v.add_points(pf(0.5, 0.5, 0.5, 1.0), &[]).is_empty());
        assert!(v.voxels.is_empty());
        assert!(v.result.is_empty());
    }

    #[test]
    fn result_map_cleared_after_call() {
        let mut v = SingleThreaded::new(1.0);
        let sensor = pf(0.5, 0.5, 0.5, 1.0);
        let points = [pf(3.5, 0.5, 0.5, 1.0)];
        v.add_points(sensor.clone(), &points);
        assert!(v.result.is_empty());
        v.add_points(sensor, &points);
        assert!(v.result.is_empty());
    }

    #[test]
    fn returns_absolute_value_not_delta() {
        let mut v = SingleThreaded::new(1.0);
        set_voxel(&mut v, 5, 0, 0, 3);
        let result = v.add_points(pf(0.5, 0.5, 0.5, 1.0), &[pf(5.5, 0.5, 0.5, 2.0)]);
        let mut expected = with_i(&[(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0)], -2);
        expected.push(pi(5, 0, 0, 5));
        let expected = sorted(expected);
        assert_eq!(sorted(result), expected);
        assert_eq!(nonzero_voxels(&v), expected);
    }

    #[test]
    fn sensor_intensity_ignored() {
        let mut v = SingleThreaded::new(1.0);
        let result = v.add_points(pf(0.5, 0.5, 0.5, 99.0), &[pf(3.5, 0.5, 0.5, 1.0)]);
        assert_eq!(result, vec![pi(3, 0, 0, 1)]);
        assert_eq!(get_voxel(&v, 3, 0, 0), Some(1));
        // Page did not exist while the ray was marching, so nothing was hit.
        assert_eq!(get_voxel(&v, 0, 0, 0), Some(0));
    }

    #[test]
    fn intensity_truncated_toward_zero() {
        let mut v = SingleThreaded::new(1.0);
        let result = v.add_points(pf(0.5, 0.5, 0.5, 1.0), &[pf(3.5, 0.5, 0.5, 2.9)]);
        assert_eq!(result, vec![pi(3, 0, 0, 2)]);
    }

    #[test]
    fn zero_intensity_reports_nothing() {
        let mut v = SingleThreaded::new(1.0);
        set_voxel(&mut v, 0, 0, 0, 0);
        let result = v.add_points(pf(0.5, 0.5, 0.5, 1.0), &[pf(3.5, 0.5, 0.5, 0.0)]);
        assert!(result.is_empty());
        assert!(nonzero_voxels(&v).is_empty());
    }

    #[test]
    fn resolution_scales_coords_not_intensity() {
        let mut v = SingleThreaded::new(0.25);
        set_voxel(&mut v, 0, 0, 0, 0);
        let result = v.add_points(pf(0.1, 0.1, 0.1, 1.0), &[pf(1.0, 0.1, 0.1, 7.0)]);
        let mut expected = with_i(&[(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)], -7);
        expected.push(pi(4, 0, 0, 7));
        assert_eq!(sorted(result), sorted(expected));

        let mut v = SingleThreaded::new(2.0);
        let result = v.add_points(pf(0.0, 0.0, 0.0, 1.0), &[pf(5.0, 0.0, 0.0, 7.0)]);
        assert_eq!(result, vec![pi(2, 0, 0, 7)]);
    }

    #[test]
    fn resolution_scales_sensor_too() {
        let mut v = SingleThreaded::new(2.0);
        set_voxel(&mut v, 0, 0, 0, 0);
        // sensor 2.0 -> 1.0 -> voxel 1, point 7.0 -> 3.5 -> voxel 3
        let result = v.add_points(pf(2.0, 0.0, 0.0, 1.0), &[pf(7.0, 0.0, 0.0, 1.0)]);
        assert_eq!(
            sorted(result),
            vec![pi(1, 0, 0, -1), pi(2, 0, 0, -1), pi(3, 0, 0, 1)]
        );
    }

    #[test]
    fn large_coords() {
        let mut v = SingleThreaded::new(1.0);
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
            let mut v = SingleThreaded::new(1.0);
            let result = v.add_points(pf(-3.5, 0.5, 0.5, 1.0), &[pf(x, 0.5, 0.5, 1.0)]);
            assert_eq!(result, vec![pi(voxel, 0, 0, 1)], "point x = {x}");
        }
    }

    // ------------------------------------------- accumulation / change detection

    #[test]
    fn accumulates_across_calls() {
        let mut v = SingleThreaded::new(1.0);
        let sensor = pf(0.5, 0.5, 0.5, 1.0);
        let points = [pf(3.5, 0.5, 0.5, 1.0)];
        assert_eq!(v.add_points(sensor.clone(), &points), vec![pi(3, 0, 0, 1)]);
        let second = v.add_points(sensor, &points);
        assert_eq!(
            sorted(second),
            vec![
                pi(0, 0, 0, -1),
                pi(1, 0, 0, -1),
                pi(2, 0, 0, -1),
                pi(3, 0, 0, 2)
            ]
        );
        assert_eq!(get_voxel(&v, 3, 0, 0), Some(2));
    }

    #[test]
    fn unchanged_voxel_not_reported() {
        let mut v = SingleThreaded::new(1.0);
        set_voxel(&mut v, 0, 0, 0, 0);
        // First ray hits (1,0,0), second ray ends there: -1 + 1 = 0.
        let result = v.add_points(
            pf(0.5, 0.5, 0.5, 1.0),
            &[pf(2.5, 0.5, 0.5, 1.0), pf(1.5, 0.5, 0.5, 1.0)],
        );
        assert_eq!(sorted(result), vec![pi(0, 0, 0, -2), pi(2, 0, 0, 1)]);
        assert_eq!(get_voxel(&v, 1, 0, 0), Some(0));
    }

    #[test]
    fn second_call_reports_only_its_changes() {
        let mut v = SingleThreaded::new(1.0);
        let sensor = pf(0.5, 0.5, 0.5, 1.0);
        assert_eq!(
            v.add_points(sensor.clone(), &[pf(0.5, 0.5, 3.5, 1.0)]),
            vec![pi(0, 0, 3, 1)]
        );
        let second = v.add_points(sensor, &[pf(3.5, 0.5, 0.5, 1.0)]);
        assert_eq!(
            sorted(second),
            vec![
                pi(0, 0, 0, -1),
                pi(1, 0, 0, -1),
                pi(2, 0, 0, -1),
                pi(3, 0, 0, 1)
            ]
        );
        assert_eq!(
            nonzero_voxels(&v),
            sorted(vec![
                pi(0, 0, 3, 1),
                pi(3, 0, 0, 1),
                pi(0, 0, 0, -1),
                pi(1, 0, 0, -1),
                pi(2, 0, 0, -1),
            ])
        );
    }

    #[test]
    fn multiple_points_one_call() {
        let mut v = SingleThreaded::new(1.0);
        set_voxel(&mut v, 0, 0, 0, 0);
        let result = v.add_points(
            pf(0.5, 0.5, 0.5, 1.0),
            &[pf(3.5, 0.5, 0.5, 1.0), pf(0.5, 3.5, 0.5, 2.0)],
        );
        let expected = sorted(vec![
            pi(0, 0, 0, -3),
            pi(1, 0, 0, -1),
            pi(2, 0, 0, -1),
            pi(3, 0, 0, 1),
            pi(0, 1, 0, -2),
            pi(0, 2, 0, -2),
            pi(0, 3, 0, 2),
        ]);
        assert_eq!(sorted(result), expected);
        assert_eq!(nonzero_voxels(&v), expected);
    }

    // ------------------------------------------------------------------ paging

    #[test]
    fn hits_skip_missing_pages() {
        let mut v = SingleThreaded::new(1.0);
        set_voxel(&mut v, 0, 0, 0, 0);
        let sensor = pf(0.5, 0.5, 0.5, 1.0);
        let points = [pf(20.5, 0.5, 0.5, 1.0)];

        let first = v.add_points(sensor.clone(), &points);
        let mut expected: Vec<PointI> = (0..16).map(|x| pi(x, 0, 0, -1)).collect();
        expected.push(pi(20, 0, 0, 1));
        assert_eq!(sorted(first), expected);
        // Page (1,0,0) was created by the add, after the ray passed through it.
        for x in 16..20 {
            assert_eq!(get_voxel(&v, x, 0, 0), Some(0), "voxel x = {x}");
        }

        let second = v.add_points(sensor, &points);
        let mut expected: Vec<PointI> = (0..16).map(|x| pi(x, 0, 0, -2)).collect();
        expected.extend((16..20).map(|x| pi(x, 0, 0, -1)));
        expected.push(pi(20, 0, 0, 2));
        assert_eq!(sorted(second), expected);
    }

    #[test]
    fn negative_page() {
        let mut v = SingleThreaded::new(1.0);
        let result = v.add_points(pf(-0.5, -0.5, -0.5, 1.0), &[pf(-3.5, -0.5, -0.5, 3.0)]);
        assert_eq!(result, vec![pi(-4, -1, -1, 3)]);
        assert_eq!(get_voxel(&v, -4, -1, -1), Some(3));
        assert_eq!(v.voxels.len(), 1);
        assert!(v.voxels.contains_key(&(-1, -1, -1)));
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

    // ------------------------------------------- point inside the sensor voxel

    #[test]
    fn point_in_sensor_voxel_fresh() {
        let mut v = SingleThreaded::new(1.0);
        let result = v.add_points(pf(0.5, 0.5, 0.5, 1.0), &[pf(0.7, 0.5, 0.5, 3.0)]);
        assert_eq!(result, vec![pi(0, 0, 0, 3)]);
        assert_eq!(nonzero_voxels(&v), vec![pi(0, 0, 0, 3)]);
    }

    #[test]
    fn point_in_sensor_voxel_accumulates() {
        // The target voxel only ever receives `+i` in every other arm, so the
        // degenerate arm must add as well instead of cancelling itself out.
        let mut v = SingleThreaded::new(1.0);
        set_voxel(&mut v, 0, 0, 0, 5);
        let result = v.add_points(pf(0.5, 0.5, 0.5, 1.0), &[pf(0.7, 0.5, 0.5, 3.0)]);
        assert_eq!(result, vec![pi(0, 0, 0, 8)]);
        assert_eq!(get_voxel(&v, 0, 0, 0), Some(8));
    }
}
