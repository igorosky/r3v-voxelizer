//! Tests shared by every [`Voxelizer`] implementation.
//!
//! An implementation exposes its private internals through [`TestVoxelizer`]
//! inside its own `mod test` and instantiates the suite with
//! [`shared_tests!`]. Tests whose outcome depends on the order in which the
//! rays of one call are processed do not belong here.

use std::panic::{AssertUnwindSafe, catch_unwind};

use crate::{
    Voxelizer,
    geometry::{PointF, PointI, ZOrder},
};

pub(crate) type Coord = (i32, i32, i32);
pub(crate) type PagedIndex = (Coord, usize);

/// A fine page is 4x4x4 voxels.
pub(crate) const PAGE_LEN: usize = 64;
/// A compressed page is 16x16x16 fine pages, one u64 word each.
pub(crate) const COMPRESSED_PAGE_LEN: usize = 4096;
/// Hits saturate at zero, so a voxel must hold something before a hit on
/// it becomes observable. Hit voxels are seeded with this value.
pub(crate) const SEED: u32 = 10;

/// Test only view of an implementation's internals.
pub(crate) trait TestVoxelizer: Voxelizer + Sized {
    const BITS_FOR_ZORDER: u32;
    const BITS_FOR_COMPRESSED_ZORDER: u32;
    const BITS_FOR_COMPRESSED_ZORDER_MASK: i32;
    const PAGE_SIZE: u32;
    /// `new` without anything implementation specific.
    fn make(
        resolution: f32,
        hit_multiplayer: u32,
        target_multiplayer: u32,
        destroy_threshold: u32,
        construct_threshold: u32,
        compress: bool,
    ) -> Self;
    fn get_index(&self, point: &PointI) -> PagedIndex;
    fn point_from_index(&self, idx: PagedIndex, i: u32) -> PointI;
    fn closest_boundary(upper: bool, position: f32, len: f32) -> f32;
    /// Writes a voxel directly, creating its (zero filled) page when missing.
    /// Used to make pages exist so that hits become observable.
    fn set_voxel(&mut self, x: i32, y: i32, z: i32, value: u32);
    /// Snapshot of every fine page, sorted by key.
    fn pages(&self) -> Vec<(Coord, [u32; PAGE_LEN])>;
    /// Snapshot of every compressed page, sorted by key. `None` when
    /// compression is off.
    fn compressed_pages(&self) -> Option<Vec<(Coord, Vec<u64>)>>;
    /// Entries still held in the `result` map. Must be 0 between calls.
    fn pending_results(&self) -> usize;
}

// ------------------------------------------------------------------ helpers

#[inline]
pub(crate) fn pf(x: f32, y: f32, z: f32, i: f32) -> PointF {
    PointF { x, y, z, i }
}
#[inline]
pub(crate) fn pi(x: i32, y: i32, z: i32, i: u32) -> PointI {
    PointI { x, y, z, i }
}
#[inline]
pub(crate) fn sorted<T: Ord>(mut v: Vec<T>) -> Vec<T> {
    v.sort();
    v
}
/// Resolution 1, multipliers 1, destroy at 0, construct at 1, no compression.
pub(crate) fn plain<T: TestVoxelizer>() -> T {
    T::make(1.0, 1, 1, 0, 1, false)
}
/// Same as [`plain`] but with compression on.
pub(crate) fn compressed<T: TestVoxelizer>() -> T {
    T::make(1.0, 1, 1, 0, 1, true)
}
/// Builds points from coordinate triples sharing one intensity.
pub(crate) fn with_i(coords: &[Coord], i: u32) -> Vec<PointI> {
    coords.iter().map(|&(x, y, z)| pi(x, y, z, i)).collect()
}
pub(crate) fn seed<T: TestVoxelizer>(v: &mut T, coords: &[Coord], value: u32) {
    for &(x, y, z) in coords {
        v.set_voxel(x, y, z, value);
    }
}
/// The fine page with key `key`, if it exists.
pub(crate) fn page<T: TestVoxelizer>(v: &T, key: Coord) -> Option<[u32; PAGE_LEN]> {
    v.pages()
        .into_iter()
        .find(|(k, _)| *k == key)
        .map(|(_, p)| p)
}
pub(crate) fn get_voxel<T: TestVoxelizer>(v: &T, x: i32, y: i32, z: i32) -> Option<u32> {
    let idx = v.get_index(&pi(x, y, z, 0));
    page(v, idx.0).map(|p| p[idx.1])
}
/// Every voxel with a non-zero value, sorted.
pub(crate) fn nonzero_voxels<T: TestVoxelizer>(v: &T) -> Vec<PointI> {
    let mut out = Vec::new();
    for (page, arr) in v.pages() {
        for (inner, val) in arr.into_iter().enumerate() {
            if val != 0 {
                out.push(v.point_from_index((page, inner), val));
            }
        }
    }
    sorted(out)
}
/// Seeds every `hit` voxel with `SEED`, fires one ray of intensity 1 and
/// checks that exactly those voxels dropped to `SEED - 1` and that
/// `target` went to 1.
pub(crate) fn check_ray<T: TestVoxelizer>(
    sensor: PointF,
    point: PointF,
    hit: &[Coord],
    target: Coord,
) {
    let mut v = plain::<T>();
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
pub(crate) fn page_bit(x: i32, y: i32, z: i32) -> u64 {
    1 << ((x & 3) + 4 * (y & 3) + 16 * (z & 3))
}
/// Voxel origin of the 4x4x2 block that `(x, y, z)` belongs to.
pub(crate) fn block_of(x: i32, y: i32, z: i32) -> Coord {
    (x & !3, y & !3, z & !1)
}
/// Bit of voxel `(x, y, z)` inside the 32 bit pack of its block.
pub(crate) fn block_bit(x: i32, y: i32, z: i32) -> u32 {
    1 << ((x & 3) + 4 * (y & 3) + 16 * (z & 1))
}
/// Expected compress mode output for the block containing `(x, y, z)`.
pub(crate) fn block(x: i32, y: i32, z: i32, mask: u32) -> PointI {
    let (bx, by, bz) = block_of(x, y, z);
    pi(bx, by, bz, mask)
}
/// The compressed page with key `key`, if compression is on and it exists.
pub(crate) fn compressed_page<T: TestVoxelizer>(v: &T, key: Coord) -> Option<Vec<u64>> {
    v.compressed_pages()?
        .into_iter()
        .find(|(k, _)| *k == key)
        .map(|(_, p)| p)
}
/// The u64 word of the fine page containing voxel `(x, y, z)`, if its
/// compressed page exists.
pub(crate) fn compressed_word<T: TestVoxelizer>(v: &T, x: i32, y: i32, z: i32) -> Option<u64> {
    let page = v.get_index(&pi(x, y, z, 0)).0;
    let key = (
        page.0 >> T::BITS_FOR_COMPRESSED_ZORDER,
        page.1 >> T::BITS_FOR_COMPRESSED_ZORDER,
        page.2 >> T::BITS_FOR_COMPRESSED_ZORDER,
    );
    let inner = ZOrder::default().index_of([
        (page.0 & T::BITS_FOR_COMPRESSED_ZORDER_MASK) as u16,
        (page.1 & T::BITS_FOR_COMPRESSED_ZORDER_MASK) as u16,
        (page.2 & T::BITS_FOR_COMPRESSED_ZORDER_MASK) as u16,
    ]) as usize;
    compressed_page(v, key).map(|words| words[inner])
}
/// Every voxel whose bit is set in the compressed pages, with `i = 1`, sorted.
pub(crate) fn compressed_bits<T: TestVoxelizer>(v: &T) -> Vec<PointI> {
    let Some(map) = v.compressed_pages() else {
        return Vec::new();
    };
    let zorder = ZOrder::default();
    let mut out = Vec::new();
    for (key, words) in map {
        for (inner, word) in words.iter().enumerate() {
            if *word == 0 {
                continue;
            }
            let [px, py, pz] = zorder.coord_of(inner as u64);
            let page = (
                (key.0 << T::BITS_FOR_COMPRESSED_ZORDER) | px as i32,
                (key.1 << T::BITS_FOR_COMPRESSED_ZORDER) | py as i32,
                (key.2 << T::BITS_FOR_COMPRESSED_ZORDER) | pz as i32,
            );
            for bit in 0..64i32 {
                if word & (1u64 << bit) != 0 {
                    out.push(pi(
                        (page.0 << T::BITS_FOR_ZORDER) | (bit & 3),
                        (page.1 << T::BITS_FOR_ZORDER) | ((bit >> 2) & 3),
                        (page.2 << T::BITS_FOR_ZORDER) | (bit >> 4),
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
pub(crate) fn from_above<T: TestVoxelizer>(v: &mut T, (x, y, z): Coord, i: f32) -> Vec<PointI> {
    let (fx, fy, fz) = (x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5);
    v.add_points(pf(fx, fy, fz + 1.0, 1.0), &[pf(fx, fy, fz, i)])
}
/// Fires one ray of intensity `i` along x from voxel `from_x` to voxel
/// `to_x`, hitting every voxel in between (including `from_x`).
pub(crate) fn along_x<T: TestVoxelizer>(
    v: &mut T,
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

// --------------------------------------------------------------- scenario
//
// A deterministic, schedule independent workload used to compare
// implementations and thread counts.

/// `(destroy_threshold, construct_threshold, compress)` pairs [`scenario`]
/// is meant to run with. With compression `destroy = construct - 1`, so a
/// voxel above the destroy threshold has always been constructed and the
/// destroy path never meets a missing compressed page.
pub(crate) const SCENARIO_PARAMS: [(u32, u32, bool); 2] = [(0, 1, false), (2, 3, true)];

/// Tiny LCG so that the scenario needs no extra dependency.
fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 33) as u32
}

/// Voxel coordinates on the six faces of the cube `|coord| = dist`, spaced
/// `step` apart within `-half..=half` on the other two axes. With
/// `half < dist` the faces do not overlap and, seen from the origin, no
/// target lies on the ray to another one.
pub(crate) fn cube_faces(dist: i32, half: i32, step: usize) -> Vec<Coord> {
    assert!(half < dist);
    let mut out = Vec::new();
    for a in (-half..=half).step_by(step) {
        for b in (-half..=half).step_by(step) {
            out.extend([
                (dist, a, b),
                (-dist, a, b),
                (a, dist, b),
                (a, -dist, b),
                (a, b, dist),
                (a, b, -dist),
            ]);
        }
    }
    out
}

#[derive(Debug, PartialEq)]
pub(crate) struct Snapshot {
    /// Sorted output of every call.
    pub(crate) results: Vec<Vec<PointI>>,
    pub(crate) pages: Vec<(Coord, [u32; PAGE_LEN])>,
    pub(crate) compressed: Option<Vec<(Coord, Vec<u64>)>>,
}

/// Two calls from the origin. Call 1 targets the faces of a cube of size 50
/// (intensities 1..=5). Call 2 targets a cube of size 100 (intensities
/// 1..=3) and every one of its rays crosses exactly one target of call 1 on
/// the way, so hits on non zero voxels do happen. Within a call no target
/// lies on another ray's path and hits only subtract, so the outcome does
/// not depend on the order in which rays are processed.
pub(crate) fn scenario<T: TestVoxelizer>(mut v: T) -> Snapshot {
    let sensor = pf(0.5, 0.5, 0.5, 1.0);
    let mut rng = 0x5EED;
    let mut results = Vec::new();
    for (dist, half, step, max_i) in [(50, 20, 4, 5), (100, 40, 8, 3)] {
        let points: Vec<PointF> = cube_faces(dist, half, step)
            .into_iter()
            .map(|(x, y, z)| {
                let i = 1 + lcg(&mut rng) % max_i;
                pf(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5, i as f32)
            })
            .collect();
        results.push(sorted(v.add_points(sensor.clone(), &points)));
    }
    Snapshot {
        results,
        pages: v.pages(),
        compressed: v.compressed_pages(),
    }
}

pub(crate) fn assert_snapshot_eq(got: &Snapshot, expected: &Snapshot, ctx: &str) {
    assert_eq!(
        got.results.len(),
        expected.results.len(),
        "{ctx}: call count"
    );
    for (call, (g, e)) in got.results.iter().zip(&expected.results).enumerate() {
        assert_eq!(g, e, "{ctx}: result of call {call}");
    }
    assert_eq!(got.pages, expected.pages, "{ctx}: voxel pages");
    assert_eq!(
        got.compressed, expected.compressed,
        "{ctx}: compressed pages"
    );
}

// ------------------------------------------------------------- shared tests

/// Instantiates every shared test for one implementation.
macro_rules! shared_tests {
    ($ty:ty) => {
        $crate::test::shared_tests!(@each $ty;
            constants,
            get_index_page_and_inner,
            index_roundtrip,
            inner_index_bijective_within_page,
            closest_boundary_fractional,
            closest_boundary_on_integer_lower,
            closest_boundary_on_integer_upper,
            basic_test_1d,
            empty_points_no_change,
            result_map_cleared_after_call,
            returns_absolute_value_not_delta,
            sensor_intensity_ignored,
            intensity_truncated_toward_zero,
            zero_intensity_reports_nothing,
            resolution_scales_coords_not_intensity,
            resolution_scales_sensor_too,
            large_coords,
            negative_coords_use_floor,
            hit_multiplayer_scales_hits,
            target_multiplayer_scales_target,
            both_multiplayers,
            hit_saturates_at_zero,
            hit_on_zero_voxel_not_reported,
            thresholds_ignored_without_compress,
            accumulates_across_calls,
            unchanged_voxel_not_reported,
            second_call_reports_only_its_changes,
            multiple_points_one_call,
            hits_skip_missing_pages,
            negative_page,
            target_page_created_lazily,
            axis_x_negative,
            axis_y_positive,
            axis_y_negative,
            axis_z_positive,
            axis_z_negative,
            plane_xy_forward,
            plane_xy_reverse,
            plane_xz,
            plane_yz,
            full_3d_forward,
            full_3d_reverse,
            sensor_on_grid_line_2d,
            ray_across_page_boundary,
            point_in_sensor_voxel_fresh,
            point_in_sensor_voxel_accumulates,
            compress_construct_sets_bit,
            compress_no_recross_no_report,
            compress_plain_changes_not_returned,
            compress_destroy_clears_bit,
            compress_construct_threshold,
            compress_destroy_threshold,
            compress_bit_positions,
            compress_both_halves_same_page,
            compress_two_voxels_same_block,
            compress_low_half_change_with_high_bits_present,
            compress_page_keys,
            compress_block_coords_far,
            compress_mask_accumulates_across_calls,
            compress_result_map_cleared_after_call,
            compress_false_leaves_none,
            compress_destroy_without_construct_panics,
            scenario_repeatable,
        );
    };
    (@each $ty:ty; $($name:ident),* $(,)?) => {
        $(
            #[test]
            fn $name() {
                $crate::test::$name::<$ty>();
            }
        )*
    };
}
pub(crate) use shared_tests;

// ---------------------------------------------------------------- indexing

pub(crate) fn constants<T: TestVoxelizer>() {
    assert_eq!(T::BITS_FOR_ZORDER, 2);
    assert_eq!(2_usize.pow(3 * T::BITS_FOR_ZORDER), PAGE_LEN);
    assert_eq!(PAGE_LEN, 64);
    // One u64 word holds exactly one fine page.
    assert_eq!(PAGE_LEN, u64::BITS as usize);
    assert_eq!(T::BITS_FOR_COMPRESSED_ZORDER, 4);
    assert_eq!(T::BITS_FOR_COMPRESSED_ZORDER_MASK, 15);
    assert_eq!(
        2_usize.pow(3 * T::BITS_FOR_COMPRESSED_ZORDER),
        COMPRESSED_PAGE_LEN
    );
    assert_eq!(COMPRESSED_PAGE_LEN, T::PAGE_SIZE as usize);
}

pub(crate) fn get_index_page_and_inner<T: TestVoxelizer>() {
    let v = plain::<T>();
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

pub(crate) fn index_roundtrip<T: TestVoxelizer>() {
    let v = plain::<T>();
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

pub(crate) fn inner_index_bijective_within_page<T: TestVoxelizer>() {
    let v = plain::<T>();
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

pub(crate) fn closest_boundary_fractional<T: TestVoxelizer>() {
    assert_eq!(T::closest_boundary(true, 0.5, 2.0), 0.25);
    assert_eq!(T::closest_boundary(false, 0.5, -2.0), 0.25);
    assert_eq!(T::closest_boundary(true, 2.25, 4.0), 0.1875);
    assert_eq!(T::closest_boundary(false, 2.25, -1.0), 0.25);
    assert_eq!(T::closest_boundary(true, -0.5, 2.0), 0.25);
    assert_eq!(T::closest_boundary(false, -0.5, -2.0), 0.25);
}

pub(crate) fn closest_boundary_on_integer_lower<T: TestVoxelizer>() {
    // Voxel k spans [k, k+1); standing on k and going down crosses at once.
    assert_eq!(T::closest_boundary(false, 3.0, -2.0), 0.0);
}

pub(crate) fn closest_boundary_on_integer_upper<T: TestVoxelizer>() {
    // Standing on 3.0 means voxel [3, 4); going up the next face is 4.0.
    assert_eq!(T::closest_boundary(true, 3.0, 2.0), 0.5);
}

// -------------------------------------------------------- add_points basics

pub(crate) fn basic_test_1d<T: TestVoxelizer>() {
    let mut x = plain::<T>();
    let result = x.add_points(pf(0.0, 0.0, 0.0, 1.0), &[pf(5.0, 0.0, 0.0, 1.0)]);
    assert_eq!(sorted(result), sorted(vec![pi(5, 0, 0, 1)]));
}

pub(crate) fn empty_points_no_change<T: TestVoxelizer>() {
    let mut v = plain::<T>();
    assert!(v.add_points(pf(0.5, 0.5, 0.5, 1.0), &[]).is_empty());
    assert!(v.pages().is_empty());
    assert_eq!(v.pending_results(), 0);
}

pub(crate) fn result_map_cleared_after_call<T: TestVoxelizer>() {
    let mut v = plain::<T>();
    let sensor = pf(0.5, 0.5, 0.5, 1.0);
    let points = [pf(3.5, 0.5, 0.5, 1.0)];
    v.add_points(sensor.clone(), &points);
    assert_eq!(v.pending_results(), 0);
    v.add_points(sensor, &points);
    assert_eq!(v.pending_results(), 0);
}

pub(crate) fn returns_absolute_value_not_delta<T: TestVoxelizer>() {
    let mut v = plain::<T>();
    let hits = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0)];
    seed(&mut v, &hits, SEED);
    v.set_voxel(5, 0, 0, 3);
    let result = v.add_points(pf(0.5, 0.5, 0.5, 1.0), &[pf(5.5, 0.5, 0.5, 2.0)]);
    let mut expected = with_i(&hits, SEED - 2);
    expected.push(pi(5, 0, 0, 5));
    let expected = sorted(expected);
    assert_eq!(sorted(result), expected);
    assert_eq!(nonzero_voxels(&v), expected);
}

pub(crate) fn sensor_intensity_ignored<T: TestVoxelizer>() {
    let mut v = plain::<T>();
    v.set_voxel(0, 0, 0, SEED);
    let result = v.add_points(pf(0.5, 0.5, 0.5, 99.0), &[pf(3.5, 0.5, 0.5, 1.0)]);
    // Hits use the point's intensity, not the sensor's.
    assert_eq!(sorted(result), vec![pi(0, 0, 0, SEED - 1), pi(3, 0, 0, 1)]);
    assert_eq!(get_voxel(&v, 0, 0, 0), Some(SEED - 1));
    assert_eq!(get_voxel(&v, 3, 0, 0), Some(1));
}

pub(crate) fn intensity_truncated_toward_zero<T: TestVoxelizer>() {
    let mut v = plain::<T>();
    let result = v.add_points(pf(0.5, 0.5, 0.5, 1.0), &[pf(3.5, 0.5, 0.5, 2.9)]);
    assert_eq!(result, vec![pi(3, 0, 0, 2)]);
}

pub(crate) fn zero_intensity_reports_nothing<T: TestVoxelizer>() {
    let mut v = plain::<T>();
    v.set_voxel(0, 0, 0, SEED);
    let result = v.add_points(pf(0.5, 0.5, 0.5, 1.0), &[pf(3.5, 0.5, 0.5, 0.0)]);
    assert!(result.is_empty());
    assert_eq!(get_voxel(&v, 0, 0, 0), Some(SEED));
    assert_eq!(get_voxel(&v, 3, 0, 0), Some(0));
}

pub(crate) fn resolution_scales_coords_not_intensity<T: TestVoxelizer>() {
    let mut v = T::make(0.25, 1, 1, 0, 1, false);
    let hits = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)];
    seed(&mut v, &hits, SEED);
    let result = v.add_points(pf(0.1, 0.1, 0.1, 1.0), &[pf(1.0, 0.1, 0.1, 7.0)]);
    let mut expected = with_i(&hits, SEED - 7);
    expected.push(pi(4, 0, 0, 7));
    assert_eq!(sorted(result), sorted(expected));

    let mut v = T::make(2.0, 1, 1, 0, 1, false);
    let result = v.add_points(pf(0.0, 0.0, 0.0, 1.0), &[pf(5.0, 0.0, 0.0, 7.0)]);
    assert_eq!(result, vec![pi(2, 0, 0, 7)]);
}

pub(crate) fn resolution_scales_sensor_too<T: TestVoxelizer>() {
    let mut v = T::make(2.0, 1, 1, 0, 1, false);
    seed(&mut v, &[(1, 0, 0), (2, 0, 0)], SEED);
    // sensor 2.0 -> 1.0 -> voxel 1, point 7.0 -> 3.5 -> voxel 3
    let result = v.add_points(pf(2.0, 0.0, 0.0, 1.0), &[pf(7.0, 0.0, 0.0, 1.0)]);
    assert_eq!(
        sorted(result),
        vec![pi(1, 0, 0, SEED - 1), pi(2, 0, 0, SEED - 1), pi(3, 0, 0, 1)]
    );
}

pub(crate) fn large_coords<T: TestVoxelizer>() {
    let mut v = plain::<T>();
    let result = v.add_points(
        pf(1000.5, -2000.5, 29999.5, 1.0),
        &[pf(1000.5, -2000.5, 30000.5, 4.0)],
    );
    assert_eq!(result, vec![pi(1000, -2001, 30000, 4)]);
    assert_eq!(get_voxel(&v, 1000, -2001, 30000), Some(4));
}

pub(crate) fn negative_coords_use_floor<T: TestVoxelizer>() {
    for (x, voxel) in [(-0.5, -1), (-1.0, -1), (-1.5, -2)] {
        let mut v = plain::<T>();
        let result = v.add_points(pf(-3.5, 0.5, 0.5, 1.0), &[pf(x, 0.5, 0.5, 1.0)]);
        assert_eq!(result, vec![pi(voxel, 0, 0, 1)], "point x = {x}");
    }
}

// ------------------------------------------------ multipliers / saturation

pub(crate) fn hit_multiplayer_scales_hits<T: TestVoxelizer>() {
    let mut v = T::make(1.0, 3, 1, 0, 1, false);
    v.set_voxel(0, 0, 0, SEED);
    let result = v.add_points(pf(0.5, 0.5, 0.5, 1.0), &[pf(3.5, 0.5, 0.5, 2.0)]);
    assert_eq!(
        sorted(result),
        vec![pi(0, 0, 0, SEED - 2 * 3), pi(3, 0, 0, 2)]
    );
}

pub(crate) fn target_multiplayer_scales_target<T: TestVoxelizer>() {
    let mut v = T::make(1.0, 1, 5, 0, 1, false);
    v.set_voxel(0, 0, 0, SEED);
    let result = v.add_points(pf(0.5, 0.5, 0.5, 1.0), &[pf(3.5, 0.5, 0.5, 2.0)]);
    assert_eq!(
        sorted(result),
        vec![pi(0, 0, 0, SEED - 2), pi(3, 0, 0, 2 * 5)]
    );
}

pub(crate) fn both_multiplayers<T: TestVoxelizer>() {
    let mut v = T::make(1.0, 2, 3, 0, 1, false);
    v.set_voxel(0, 0, 0, SEED);
    let result = v.add_points(pf(0.5, 0.5, 0.5, 1.0), &[pf(3.5, 0.5, 0.5, 2.0)]);
    assert_eq!(
        sorted(result),
        vec![pi(0, 0, 0, SEED - 2 * 2), pi(3, 0, 0, 2 * 3)]
    );
    assert_eq!(get_voxel(&v, 0, 0, 0), Some(SEED - 4));
    assert_eq!(get_voxel(&v, 3, 0, 0), Some(6));
}

pub(crate) fn hit_saturates_at_zero<T: TestVoxelizer>() {
    let mut v = plain::<T>();
    v.set_voxel(0, 0, 0, 1);
    // (1,0,0) and (2,0,0) exist (same page) and hold 0.
    let result = v.add_points(pf(0.5, 0.5, 0.5, 1.0), &[pf(3.5, 0.5, 0.5, 5.0)]);
    // 1 -> 0 is a change and gets reported as 0; 0 -> 0 is not.
    assert_eq!(sorted(result), vec![pi(0, 0, 0, 0), pi(3, 0, 0, 5)]);
    assert_eq!(get_voxel(&v, 0, 0, 0), Some(0));
    assert_eq!(get_voxel(&v, 1, 0, 0), Some(0));
    assert_eq!(get_voxel(&v, 2, 0, 0), Some(0));

    // The multiplier saturates too.
    let mut v = T::make(1.0, 100, 1, 0, 1, false);
    v.set_voxel(0, 0, 0, SEED);
    let result = v.add_points(pf(0.5, 0.5, 0.5, 1.0), &[pf(3.5, 0.5, 0.5, 1.0)]);
    assert_eq!(sorted(result), vec![pi(0, 0, 0, 0), pi(3, 0, 0, 1)]);
}

pub(crate) fn hit_on_zero_voxel_not_reported<T: TestVoxelizer>() {
    let mut v = plain::<T>();
    v.set_voxel(0, 0, 0, 0);
    let result = v.add_points(pf(0.5, 0.5, 0.5, 1.0), &[pf(3.5, 0.5, 0.5, 1.0)]);
    assert_eq!(result, vec![pi(3, 0, 0, 1)]);
    assert_eq!(nonzero_voxels(&v), vec![pi(3, 0, 0, 1)]);
}

pub(crate) fn thresholds_ignored_without_compress<T: TestVoxelizer>() {
    let mut v = T::make(1.0, 1, 1, 100, 200, false);
    let hits = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0)];
    seed(&mut v, &hits, SEED);
    v.set_voxel(5, 0, 0, 3);
    let result = v.add_points(pf(0.5, 0.5, 0.5, 1.0), &[pf(5.5, 0.5, 0.5, 2.0)]);
    let mut expected = with_i(&hits, SEED - 2);
    expected.push(pi(5, 0, 0, 5));
    assert_eq!(sorted(result), sorted(expected));
    assert!(v.compressed_pages().is_none());
}

// ------------------------------------------- accumulation / change detection

pub(crate) fn accumulates_across_calls<T: TestVoxelizer>() {
    let mut v = plain::<T>();
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

pub(crate) fn unchanged_voxel_not_reported<T: TestVoxelizer>() {
    let mut v = plain::<T>();
    v.set_voxel(0, 0, 0, 5);
    v.set_voxel(1, 0, 0, 1);
    // First ray hits (1,0,0): 1 -> 0, second ray ends there: 0 -> 1.
    // (The other order gives 1 -> 2 -> 1: same outcome.)
    let result = v.add_points(
        pf(0.5, 0.5, 0.5, 1.0),
        &[pf(2.5, 0.5, 0.5, 1.0), pf(1.5, 0.5, 0.5, 1.0)],
    );
    assert_eq!(sorted(result), vec![pi(0, 0, 0, 3), pi(2, 0, 0, 1)]);
    assert_eq!(get_voxel(&v, 1, 0, 0), Some(1));
}

pub(crate) fn second_call_reports_only_its_changes<T: TestVoxelizer>() {
    let mut v = plain::<T>();
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

pub(crate) fn multiple_points_one_call<T: TestVoxelizer>() {
    let mut v = plain::<T>();
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

pub(crate) fn hits_skip_missing_pages<T: TestVoxelizer>() {
    let mut v = plain::<T>();
    v.set_voxel(0, 0, 0, SEED);
    let sensor = pf(0.5, 0.5, 0.5, 1.0);
    let points = [pf(20.5, 0.5, 0.5, 1.0)];

    // The ray crosses fine pages 0..=5 along x but only page 0 exists.
    let first = v.add_points(sensor.clone(), &points);
    assert_eq!(sorted(first), vec![pi(0, 0, 0, SEED - 1), pi(20, 0, 0, 1)]);
    // Only the seeded page and the target's page exist afterwards.
    assert_eq!(v.pages().len(), 2);
    for x in 1..4 {
        assert_eq!(get_voxel(&v, x, 0, 0), Some(0), "voxel x = {x}");
    }
    for x in 4..20 {
        assert_eq!(get_voxel(&v, x, 0, 0), None, "voxel x = {x}");
    }

    let second = v.add_points(sensor.clone(), &points);
    assert_eq!(sorted(second), vec![pi(0, 0, 0, SEED - 2), pi(20, 0, 0, 2)]);
    assert_eq!(v.pages().len(), 2);

    // Once a page exists on the way its voxels do get hit.
    v.set_voxel(8, 0, 0, SEED);
    let third = v.add_points(sensor, &points);
    assert_eq!(
        sorted(third),
        vec![
            pi(0, 0, 0, SEED - 3),
            pi(8, 0, 0, SEED - 1),
            pi(20, 0, 0, 3)
        ]
    );
    assert_eq!(v.pages().len(), 3);
}

pub(crate) fn negative_page<T: TestVoxelizer>() {
    let mut v = plain::<T>();
    let result = v.add_points(pf(-0.5, -0.5, -0.5, 1.0), &[pf(-3.5, -0.5, -0.5, 3.0)]);
    assert_eq!(result, vec![pi(-4, -1, -1, 3)]);
    assert_eq!(get_voxel(&v, -4, -1, -1), Some(3));
    assert_eq!(v.pages().len(), 1);
    assert!(page(&v, (-1, -1, -1)).is_some());
}

pub(crate) fn target_page_created_lazily<T: TestVoxelizer>() {
    let mut v = plain::<T>();
    assert!(v.pages().is_empty());
    v.add_points(pf(0.5, 0.5, 0.5, 1.0), &[pf(9.5, 0.5, 0.5, 1.0)]);
    assert_eq!(v.pages().len(), 1);
    assert!(page(&v, (2, 0, 0)).is_some());
}

// -------------------------------------------------------- ray march arms

pub(crate) fn axis_x_negative<T: TestVoxelizer>() {
    check_ray::<T>(
        pf(5.5, 0.5, 0.5, 1.0),
        pf(0.5, 0.5, 0.5, 1.0),
        &[(5, 0, 0), (4, 0, 0), (3, 0, 0), (2, 0, 0), (1, 0, 0)],
        (0, 0, 0),
    );
}

pub(crate) fn axis_y_positive<T: TestVoxelizer>() {
    check_ray::<T>(
        pf(0.5, 0.5, 0.5, 1.0),
        pf(0.5, 4.5, 0.5, 1.0),
        &[(0, 0, 0), (0, 1, 0), (0, 2, 0), (0, 3, 0)],
        (0, 4, 0),
    );
}

pub(crate) fn axis_y_negative<T: TestVoxelizer>() {
    check_ray::<T>(
        pf(0.5, 4.5, 0.5, 1.0),
        pf(0.5, 0.5, 0.5, 1.0),
        &[(0, 4, 0), (0, 3, 0), (0, 2, 0), (0, 1, 0)],
        (0, 0, 0),
    );
}

pub(crate) fn axis_z_positive<T: TestVoxelizer>() {
    check_ray::<T>(
        pf(0.5, 0.5, 0.5, 1.0),
        pf(0.5, 0.5, 4.5, 1.0),
        &[(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3)],
        (0, 0, 4),
    );
}

pub(crate) fn axis_z_negative<T: TestVoxelizer>() {
    check_ray::<T>(
        pf(0.5, 0.5, 4.5, 1.0),
        pf(0.5, 0.5, 0.5, 1.0),
        &[(0, 0, 4), (0, 0, 3), (0, 0, 2), (0, 0, 1)],
        (0, 0, 0),
    );
}

pub(crate) fn plane_xy_forward<T: TestVoxelizer>() {
    check_ray::<T>(
        pf(0.5, 0.5, 0.5, 1.0),
        pf(3.5, 1.7, 0.5, 1.0),
        &[(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0)],
        (3, 1, 0),
    );
}

pub(crate) fn plane_xy_reverse<T: TestVoxelizer>() {
    check_ray::<T>(
        pf(3.5, 1.7, 0.5, 1.0),
        pf(0.5, 0.5, 0.5, 1.0),
        &[(3, 1, 0), (2, 1, 0), (1, 1, 0), (1, 0, 0)],
        (0, 0, 0),
    );
}

pub(crate) fn plane_xz<T: TestVoxelizer>() {
    check_ray::<T>(
        pf(0.5, 0.5, 0.5, 1.0),
        pf(3.5, 0.5, 1.7, 1.0),
        &[(0, 0, 0), (1, 0, 0), (1, 0, 1), (2, 0, 1)],
        (3, 0, 1),
    );
}

pub(crate) fn plane_yz<T: TestVoxelizer>() {
    check_ray::<T>(
        pf(0.5, 0.5, 0.5, 1.0),
        pf(0.5, 3.5, 1.7, 1.0),
        &[(0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 2, 1)],
        (0, 3, 1),
    );
}

pub(crate) fn full_3d_forward<T: TestVoxelizer>() {
    check_ray::<T>(
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

pub(crate) fn full_3d_reverse<T: TestVoxelizer>() {
    check_ray::<T>(
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

pub(crate) fn sensor_on_grid_line_2d<T: TestVoxelizer>() {
    // Line y = 0.6x from the origin: crosses x=1 at t=0.4 (y=0.6), y=1 at
    // t=0.667 (x=1.67), x=2 at t=0.8.
    check_ray::<T>(
        pf(0.0, 0.0, 0.0, 1.0),
        pf(2.5, 1.5, 0.5, 1.0),
        &[(0, 0, 0), (1, 0, 0), (1, 1, 0)],
        (2, 1, 0),
    );
}

pub(crate) fn ray_across_page_boundary<T: TestVoxelizer>() {
    // Hits span fine pages (0,0,0) and (1,0,0) along x.
    check_ray::<T>(
        pf(2.5, 0.5, 0.5, 1.0),
        pf(6.5, 1.7, 0.5, 1.0),
        &[(2, 0, 0), (3, 0, 0), (4, 0, 0), (4, 1, 0), (5, 1, 0)],
        (6, 1, 0),
    );
}

// ------------------------------------------- point inside the sensor voxel

pub(crate) fn point_in_sensor_voxel_fresh<T: TestVoxelizer>() {
    let mut v = plain::<T>();
    let result = v.add_points(pf(0.5, 0.5, 0.5, 1.0), &[pf(0.7, 0.5, 0.5, 3.0)]);
    assert_eq!(result, vec![pi(0, 0, 0, 3)]);
    assert_eq!(nonzero_voxels(&v), vec![pi(0, 0, 0, 3)]);
}

pub(crate) fn point_in_sensor_voxel_accumulates<T: TestVoxelizer>() {
    // The target voxel only ever receives `+i` in every other arm, so the
    // degenerate arm must add as well instead of cancelling itself out.
    let mut v = plain::<T>();
    v.set_voxel(0, 0, 0, 5);
    let result = v.add_points(pf(0.5, 0.5, 0.5, 1.0), &[pf(0.7, 0.5, 0.5, 3.0)]);
    assert_eq!(result, vec![pi(0, 0, 0, 8)]);
    assert_eq!(get_voxel(&v, 0, 0, 0), Some(8));
}

// ----------------------------------------------------------- compress mode

pub(crate) fn compress_construct_sets_bit<T: TestVoxelizer>() {
    let mut v = compressed::<T>();
    let result = along_x(&mut v, 0, 3, 0, 0, 1.0);
    assert_eq!(get_voxel(&v, 3, 0, 0), Some(1));
    assert_eq!(compressed_word(&v, 3, 0, 0), Some(page_bit(3, 0, 0)));
    assert_eq!(compressed_bits(&v), vec![pi(3, 0, 0, 1)]);
    let map = v.compressed_pages().unwrap();
    assert_eq!(map.len(), 1);
    assert_eq!(map[0].0, (0, 0, 0));
    assert_eq!(map[0].1.len(), COMPRESSED_PAGE_LEN);
    assert_eq!(result, vec![block(3, 0, 0, block_bit(3, 0, 0))]);
    assert_eq!(result, vec![pi(0, 0, 0, 1 << 3)]);
}

pub(crate) fn compress_no_recross_no_report<T: TestVoxelizer>() {
    let mut v = compressed::<T>();
    along_x(&mut v, 0, 3, 0, 0, 1.0);
    // 1 -> 2 stays above the construct threshold: nothing to report, even
    // though the voxel itself changed.
    let second = along_x(&mut v, 0, 3, 0, 0, 1.0);
    assert!(second.is_empty());
    assert_eq!(get_voxel(&v, 3, 0, 0), Some(2));
    assert_eq!(compressed_word(&v, 3, 0, 0), Some(page_bit(3, 0, 0)));
    assert_eq!(compressed_bits(&v), vec![pi(3, 0, 0, 1)]);
}

pub(crate) fn compress_plain_changes_not_returned<T: TestVoxelizer>() {
    // Below the construct threshold a voxel change is invisible in
    // compress mode, while plain mode reports it.
    let mut c = T::make(1.0, 1, 1, 0, 5, true);
    assert!(along_x(&mut c, 0, 3, 0, 0, 1.0).is_empty());
    assert_eq!(get_voxel(&c, 3, 0, 0), Some(1));
    assert!(c.compressed_pages().unwrap().is_empty());
    let mut p = T::make(1.0, 1, 1, 0, 5, false);
    assert_eq!(along_x(&mut p, 0, 3, 0, 0, 1.0), vec![pi(3, 0, 0, 1)]);
}

pub(crate) fn compress_destroy_clears_bit<T: TestVoxelizer>() {
    let mut v = compressed::<T>();
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

pub(crate) fn compress_construct_threshold<T: TestVoxelizer>() {
    let mut v = T::make(1.0, 1, 1, 0, 3, true);
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
    let mut v = T::make(1.0, 1, 1, 0, 3, true);
    let result = along_x(&mut v, 0, 3, 0, 0, 7.0);
    assert_eq!(result, vec![block(3, 0, 0, block_bit(3, 0, 0))]);
    assert_eq!(compressed_bits(&v), vec![pi(3, 0, 0, 1)]);
}

pub(crate) fn compress_destroy_threshold<T: TestVoxelizer>() {
    let mut v = T::make(1.0, 1, 1, 2, 5, true);
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

pub(crate) fn compress_bit_positions<T: TestVoxelizer>() {
    for x in 0..4 {
        for y in 0..4 {
            for z in 0..4 {
                let mut v = compressed::<T>();
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

pub(crate) fn compress_both_halves_same_page<T: TestVoxelizer>() {
    let mut v = compressed::<T>();
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

pub(crate) fn compress_two_voxels_same_block<T: TestVoxelizer>() {
    let mut v = compressed::<T>();
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

pub(crate) fn compress_low_half_change_with_high_bits_present<T: TestVoxelizer>() {
    let mut v = compressed::<T>();
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

pub(crate) fn compress_page_keys<T: TestVoxelizer>() {
    let mut v = compressed::<T>();
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

    let zorder = ZOrder::default();
    assert_eq!(v.compressed_pages().unwrap().len(), 3);
    // Fine page (1,0,0) lives in compressed page (0,0,0) at z-order [1,0,0].
    let inner = zorder.index_of([1u16, 0, 0]) as usize;
    assert_eq!(
        compressed_page(&v, (0, 0, 0)).unwrap()[inner],
        page_bit(4, 0, 0)
    );
    // Fine page (16,0,0) is the first slot of compressed page (1,0,0).
    assert_eq!(
        compressed_page(&v, (1, 0, 0)).unwrap()[0],
        page_bit(64, 0, 0)
    );
    // Fine page (-1,-1,-1) is the last slot of compressed page (-1,-1,-1).
    assert_eq!(zorder.index_of([15u16, 15, 15]), 4095);
    assert_eq!(
        compressed_page(&v, (-1, -1, -1)).unwrap()[4095],
        page_bit(-1, -1, -1)
    );
    assert_eq!(
        compressed_bits(&v),
        sorted(vec![pi(4, 0, 0, 1), pi(64, 0, 0, 1), pi(-1, -1, -1, 1)])
    );
}

pub(crate) fn compress_block_coords_far<T: TestVoxelizer>() {
    let mut v = compressed::<T>();
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

pub(crate) fn compress_mask_accumulates_across_calls<T: TestVoxelizer>() {
    let mut v = compressed::<T>();
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

pub(crate) fn compress_result_map_cleared_after_call<T: TestVoxelizer>() {
    let mut v = compressed::<T>();
    along_x(&mut v, 0, 3, 0, 0, 1.0);
    assert_eq!(v.pending_results(), 0);
    along_x(&mut v, 0, 7, 0, 0, 1.0);
    assert_eq!(v.pending_results(), 0);
}

pub(crate) fn compress_false_leaves_none<T: TestVoxelizer>() {
    let mut v = plain::<T>();
    along_x(&mut v, 0, 3, 0, 0, 1.0);
    along_x(&mut v, 0, 7, 0, 0, 1.0);
    assert!(v.compressed_pages().is_none());
}

pub(crate) fn compress_destroy_without_construct_panics<T: TestVoxelizer>() {
    // Construct threshold high enough that the target does not create the
    // compressed page, so the destroy of the seeded voxel finds nothing.
    let mut v = T::make(1.0, 1, 1, 0, 10, true);
    v.set_voxel(0, 0, 0, 5);
    let err = catch_unwind(AssertUnwindSafe(|| along_x(&mut v, 0, 3, 0, 0, 5.0)))
        .expect_err("destroying a voxel that was never constructed must panic");
    let msg = err
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| err.downcast_ref::<&str>().copied())
        .unwrap_or("");
    assert!(
        msg.contains("It must have existed"),
        "unexpected panic message: {msg:?}"
    );
}

// ---------------------------------------------------------------- scenario

pub(crate) fn scenario_repeatable<T: TestVoxelizer>() {
    for (destroy, construct, compress) in SCENARIO_PARAMS {
        let a = scenario(T::make(1.0, 1, 1, destroy, construct, compress));
        assert!(a.results.iter().all(|r| !r.is_empty()));
        let b = scenario(T::make(1.0, 1, 1, destroy, construct, compress));
        assert_snapshot_eq(&a, &b, &format!("compress = {compress}"));
    }
}

// ------------------------------------------------------- helper self checks

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
fn cube_faces_self_check() {
    let faces = cube_faces(50, 20, 4);
    assert_eq!(faces.len(), 11 * 11 * 6);
    let unique = sorted(faces.clone()).windows(2).all(|w| w[0] != w[1]);
    assert!(unique);
    assert!(
        faces
            .iter()
            .all(|c| [c.0, c.1, c.2].contains(&50) || [c.0, c.1, c.2].contains(&-50))
    );
    // Every call 2 target of `scenario` sits right behind a call 1 target.
    let far = cube_faces(100, 40, 8);
    assert_eq!(far.len(), 11 * 11 * 6);
    assert!(
        far.iter()
            .all(|&(x, y, z)| faces.contains(&(x / 2, y / 2, z / 2)))
    );
}

#[cfg(all(feature = "single_threaded", feature = "multi_threaded"))]
#[test]
fn single_and_multi_agree() {
    use crate::{MultithreadedThreaded, SingleThreaded};
    for (destroy, construct, compress) in SCENARIO_PARAMS {
        let expected = scenario(SingleThreaded::new(1.0, 1, 1, destroy, construct, compress));
        assert!(expected.results.iter().all(|r| !r.is_empty()));
        for threads in [1, 2, 4, 8] {
            let got = scenario(
                MultithreadedThreaded::new(1.0, 1, 1, destroy, construct, compress, threads)
                    .expect("thread pool must build"),
            );
            assert_snapshot_eq(
                &got,
                &expected,
                &format!("compress = {compress}, threads = {threads}"),
            );
        }
    }
}
