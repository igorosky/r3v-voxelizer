//! Tests shared by every [`Voxelizer`] implementation.
//!
//! An implementation exposes its private internals through [`TestVoxelizer`]
//! inside its own `mod test` and instantiates the suite with
//! [`shared_tests!`]. Tests whose outcome depends on the order in which the
//! rays of one call are processed do not belong here.
//!
//! Without any implementation feature there is nothing to run it against.
#![cfg(any(
    feature = "single_threaded",
    feature = "multi_threaded",
    feature = "wgpu"
))]

use crate::{
    Multipliers, Voxelizer, VoxelizerParams,
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
/// Multipliers leaving intensities as they are.
pub(crate) const UNSCALED: Multipliers = Multipliers {
    intensity: 1.0,
    hit: 1,
    target: 1,
};
/// Resolution 1, multipliers 1, destroy at 0, construct at 1, no cap, no
/// compression.
pub(crate) const PLAIN: VoxelizerParams = VoxelizerParams {
    resolution: 1.0,
    multipliers: UNSCALED,
    destroy_threshold: 0,
    construct_threshold: 1,
    max_value: u32::MAX,
    compress: false,
};

/// Test only view of an implementation's internals.
pub(crate) trait TestVoxelizer: Voxelizer + Sized {
    const BITS_FOR_ZORDER: u32;
    const BITS_FOR_COMPRESSED_ZORDER: u32;
    const BITS_FOR_COMPRESSED_ZORDER_MASK: i32;
    const PAGE_SIZE: u32;
    /// `new` without anything implementation specific.
    fn with_params(params: VoxelizerParams) -> Self;
    /// [`TestVoxelizer::with_params`] without a cap on voxel values or
    /// scaling of intensities.
    fn make(
        resolution: f32,
        hit_multiplier: u32,
        target_multiplier: u32,
        destroy_threshold: u32,
        construct_threshold: u32,
        compress: bool,
    ) -> Self {
        Self::with_params(VoxelizerParams {
            resolution,
            multipliers: Multipliers {
                hit: hit_multiplier,
                target: target_multiplier,
                ..UNSCALED
            },
            destroy_threshold,
            construct_threshold,
            max_value: u32::MAX,
            compress,
        })
    }
    fn get_index(&self, point: &PointI) -> PagedIndex;
    fn point_from_index(&self, idx: PagedIndex, i: u32) -> PointI;
    fn closest_boundary(upper: bool, position: f32, len: f32) -> f32;
    /// Writes a voxel directly, creating its (zero filled) page when missing.
    /// Used to make pages exist so that hits become observable.
    fn set_voxel(&mut self, x: i32, y: i32, z: i32, value: u32);
    /// [`TestVoxelizer::set_voxel`] for every `(x, y, z, value)` of `voxels`.
    fn set_voxels(&mut self, voxels: &[(i32, i32, i32, u32)]) {
        for &(x, y, z, value) in voxels {
            self.set_voxel(x, y, z, value);
        }
    }
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
/// Built with [`PLAIN`].
pub(crate) fn plain<T: TestVoxelizer>() -> T {
    T::with_params(PLAIN)
}
/// Same as [`plain`] but with compression on.
pub(crate) fn compressed<T: TestVoxelizer>() -> T {
    T::with_params(VoxelizerParams {
        compress: true,
        ..PLAIN
    })
}
/// Builds points from coordinate triples sharing one intensity.
pub(crate) fn with_i(coords: &[Coord], i: u32) -> Vec<PointI> {
    coords.iter().map(|&(x, y, z)| pi(x, y, z, i)).collect()
}
pub(crate) fn seed<T: TestVoxelizer>(v: &mut T, coords: &[Coord], value: u32) {
    let voxels: Vec<_> = coords.iter().map(|&(x, y, z)| (x, y, z, value)).collect();
    v.set_voxels(&voxels);
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

/// Parameters [`scenario`] is meant to run with. The second compressed set
/// leaves a gap between the thresholds: call 1 targets of intensity 3 stay
/// in it without being constructed and call 2 hits drop both them and
/// constructed ones below the destroy threshold. The caps are below the
/// intensities of call 1, whose targets call 2 hits. The intensities are
/// integers, so scaling them by 1.5 truncates the odd ones.
pub(crate) const SCENARIO_PARAMS: [VoxelizerParams; 6] = [
    PLAIN,
    VoxelizerParams {
        destroy_threshold: 2,
        construct_threshold: 3,
        compress: true,
        ..PLAIN
    },
    VoxelizerParams {
        destroy_threshold: 2,
        construct_threshold: 4,
        compress: true,
        ..PLAIN
    },
    VoxelizerParams {
        max_value: 3,
        ..PLAIN
    },
    VoxelizerParams {
        destroy_threshold: 2,
        construct_threshold: 3,
        max_value: 4,
        compress: true,
        ..PLAIN
    },
    VoxelizerParams {
        multipliers: Multipliers {
            intensity: 1.5,
            ..UNSCALED
        },
        ..PLAIN
    },
];

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

/// Multipliers [`scenario_with`] switches to before call 2, different from
/// [`UNSCALED`] in every field.
pub(crate) const SWITCHED: Multipliers = Multipliers {
    intensity: 1.5,
    hit: 2,
    target: 3,
};

/// Two calls from the origin. Call 1 targets the faces of a cube of size 50
/// (intensities 1..=5). Call 2 targets a cube of size 100 (intensities
/// 1..=3) and every one of its rays crosses exactly one target of call 1 on
/// the way, so hits on non zero voxels do happen. Within a call no target
/// lies on another ray's path and hits only subtract, so the outcome does
/// not depend on the order in which rays are processed.
pub(crate) fn scenario<T: TestVoxelizer>(v: T) -> Snapshot {
    scenario_with(v, None)
}

/// [`scenario`] with the multipliers set to `switch` (when given) between
/// the two calls.
pub(crate) fn scenario_with<T: TestVoxelizer>(mut v: T, switch: Option<Multipliers>) -> Snapshot {
    let sensor = pf(0.5, 0.5, 0.5, 1.0);
    let mut rng = 0x5EED;
    let mut results = Vec::new();
    for (call, (dist, half, step, max_i)) in
        [(50, 20, 4, 5), (100, 40, 8, 3)].into_iter().enumerate()
    {
        if call == 1
            && let Some(multipliers) = switch
        {
            v.set_multipliers(multipliers);
        }
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
            hit_multiplier_scales_hits,
            target_multiplier_scales_target,
            both_multipliers,
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
            emptied_page_removed,
            partially_emptied_page_kept,
            refilled_page_kept,
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
            plane_xy_point_on_corner,
            plane_xz_point_on_corner,
            plane_yz_point_on_corner,
            full_3d_point_on_corner_z,
            full_3d_point_on_corner_y,
            ray_ends_on_voxel_corners,
            ray_ends_near_voxel_corners,
            point_in_sensor_voxel_fresh,
            point_in_sensor_voxel_accumulates,
            compress_construct_sets_bit,
            compress_no_recross_no_report,
            compress_plain_changes_not_returned,
            compress_destroy_clears_bit,
            compress_construct_threshold,
            compress_destroy_threshold,
            compress_emptied_pages_removed,
            compress_constructed_voxel_keeps_page,
            compress_bit_positions,
            compress_both_halves_same_page,
            compress_two_voxels_same_block,
            compress_low_half_change_with_high_bits_present,
            compress_page_keys,
            compress_block_coords_far,
            compress_mask_accumulates_across_calls,
            compress_result_map_cleared_after_call,
            compress_false_leaves_none,
            compress_destroy_without_construct_ignored,
            compress_destroy_without_construct_keeps_other_bits,
            compress_reconstruct_without_destroy_not_reported,
            max_value_caps_target,
            max_value_caps_within_one_call,
            max_value_voxel_not_permanent,
            max_value_saturates_instead_of_overflowing,
            max_value_with_compress,
            intensity_multiplier_scales_target,
            intensity_multiplier_scales_hit,
            intensity_multiplier_truncates_after_scaling,
            intensity_multiplier_zero_or_negative,
            multipliers_returns_constructor_values,
            set_multipliers_roundtrip,
            multipliers_unchanged_by_add_points,
            set_multipliers_applies_to_next_call,
            set_multipliers_keeps_voxels,
            set_intensity_multiplier_only,
            set_hit_multiplier_only,
            set_target_multiplier_only,
            set_hit_multiplier_zero_spares_hit_voxels,
            set_multipliers_with_compress,
            set_multipliers_same_as_constructed,
            scenario_repeatable,
            switched_scenario_repeatable,
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
    let mut seen = [false; PAGE_LEN];
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

pub(crate) fn hit_multiplier_scales_hits<T: TestVoxelizer>() {
    let mut v = T::make(1.0, 3, 1, 0, 1, false);
    v.set_voxel(0, 0, 0, SEED);
    let result = v.add_points(pf(0.5, 0.5, 0.5, 1.0), &[pf(3.5, 0.5, 0.5, 2.0)]);
    assert_eq!(
        sorted(result),
        vec![pi(0, 0, 0, SEED - 2 * 3), pi(3, 0, 0, 2)]
    );
}

pub(crate) fn target_multiplier_scales_target<T: TestVoxelizer>() {
    let mut v = T::make(1.0, 1, 5, 0, 1, false);
    v.set_voxel(0, 0, 0, SEED);
    let result = v.add_points(pf(0.5, 0.5, 0.5, 1.0), &[pf(3.5, 0.5, 0.5, 2.0)]);
    assert_eq!(
        sorted(result),
        vec![pi(0, 0, 0, SEED - 2), pi(3, 0, 0, 2 * 5)]
    );
}

pub(crate) fn both_multipliers<T: TestVoxelizer>() {
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

pub(crate) fn emptied_page_removed<T: TestVoxelizer>() {
    let mut v = plain::<T>();
    v.set_voxel(1, 0, 0, 1);
    // Zeroes (1,0,0), the only non zero voxel of page (0,0,0), and ends in
    // page (1,0,0).
    let result = along_x(&mut v, 1, 5, 0, 0, 1.0);
    // The drop to 0 is still reported.
    assert_eq!(sorted(result), vec![pi(1, 0, 0, 0), pi(5, 0, 0, 1)]);
    assert!(page(&v, (0, 0, 0)).is_none());
    assert_eq!(v.pages().len(), 1);
    assert_eq!(v.pending_results(), 0);
    // Hits on the removed page change nothing, a target creates it again.
    assert_eq!(along_x(&mut v, 1, 2, 0, 0, 1.0), vec![pi(2, 0, 0, 1)]);
    assert_eq!(get_voxel(&v, 1, 0, 0), Some(0));
    assert_eq!(v.pages().len(), 2);
}

pub(crate) fn partially_emptied_page_kept<T: TestVoxelizer>() {
    let mut v = plain::<T>();
    v.set_voxel(1, 0, 0, 1);
    v.set_voxel(1, 1, 0, 1);
    let result = along_x(&mut v, 1, 5, 0, 0, 1.0);
    assert_eq!(sorted(result), vec![pi(1, 0, 0, 0), pi(5, 0, 0, 1)]);
    assert_eq!(get_voxel(&v, 1, 0, 0), Some(0));
    assert_eq!(get_voxel(&v, 1, 1, 0), Some(1));
    assert_eq!(v.pages().len(), 2);
}

pub(crate) fn refilled_page_kept<T: TestVoxelizer>() {
    let mut v = plain::<T>();
    v.set_voxel(1, 0, 0, 1);
    // The ray to (5,0,0) zeroes (1,0,0), the second point refills it before
    // the end of the call, so it is not even reported.
    let result = v.add_points(
        pf(0.5, 0.5, 0.5, 1.0),
        &[pf(5.5, 0.5, 0.5, 1.0), pf(1.5, 0.5, 0.5, 1.0)],
    );
    assert_eq!(result, vec![pi(5, 0, 0, 1)]);
    assert_eq!(get_voxel(&v, 1, 0, 0), Some(1));
    assert_eq!(v.pages().len(), 2);
    assert_eq!(v.pending_results(), 0);
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

// ---------------------------------------------- point on voxel boundaries
//
// A point lying on boundaries of several axes makes the last crossings tie at
// the end of the ray. On a tie an axis that already reached the target must
// not be stepped, as it would walk past the target and never come back.

pub(crate) fn plane_xy_point_on_corner<T: TestVoxelizer>() {
    check_ray::<T>(
        pf(0.0, 2.0, 0.5, 1.0),
        pf(4.0, 0.0, 0.5, 1.0),
        &[
            (0, 2, 0),
            (0, 1, 0),
            (1, 1, 0),
            (1, 0, 0),
            (2, 0, 0),
            (3, 0, 0),
        ],
        (4, 0, 0),
    );
}

pub(crate) fn plane_xz_point_on_corner<T: TestVoxelizer>() {
    check_ray::<T>(
        pf(0.0, 0.5, 2.0, 1.0),
        pf(4.0, 0.5, 0.0, 1.0),
        &[
            (0, 0, 2),
            (0, 0, 1),
            (1, 0, 1),
            (1, 0, 0),
            (2, 0, 0),
            (3, 0, 0),
        ],
        (4, 0, 0),
    );
}

pub(crate) fn plane_yz_point_on_corner<T: TestVoxelizer>() {
    check_ray::<T>(
        pf(0.5, 0.0, 2.0, 1.0),
        pf(0.5, 4.0, 0.0, 1.0),
        &[
            (0, 0, 2),
            (0, 0, 1),
            (0, 1, 1),
            (0, 1, 0),
            (0, 2, 0),
            (0, 3, 0),
        ],
        (0, 4, 0),
    );
}

pub(crate) fn full_3d_point_on_corner_z<T: TestVoxelizer>() {
    check_ray::<T>(
        pf(0.0, 0.0, 2.0, 1.0),
        pf(4.0, 2.0, 0.0, 1.0),
        &[
            (0, 0, 2),
            (0, 0, 1),
            (1, 0, 1),
            (1, 0, 0),
            (1, 1, 0),
            (2, 1, 0),
            (3, 1, 0),
            (3, 2, 0),
        ],
        (4, 2, 0),
    );
}

pub(crate) fn full_3d_point_on_corner_y<T: TestVoxelizer>() {
    check_ray::<T>(
        pf(0.0, 2.0, 0.0, 1.0),
        pf(4.0, 0.0, 2.0, 1.0),
        &[
            (0, 2, 0),
            (0, 1, 0),
            (1, 1, 0),
            (1, 1, 1),
            (1, 0, 1),
            (2, 0, 1),
            (3, 0, 1),
            (3, 0, 2),
        ],
        (4, 0, 2),
    );
}

/// Every voxel of the box with corners `a` and `b`.
fn box_coords(a: Coord, b: Coord) -> Vec<Coord> {
    let mut out = Vec::new();
    for x in a.0.min(b.0)..=a.0.max(b.0) {
        for y in a.1.min(b.1)..=a.1.max(b.1) {
            for z in a.2.min(b.2)..=a.2.max(b.2) {
                out.push((x, y, z));
            }
        }
    }
    out
}

/// Voxel holding `p`, the same way `add_points` finds it.
fn voxel_of(p: &PointF, resolution: f32) -> Coord {
    let p = p / resolution;
    (p.x.floor() as i32, p.y.floor() as i32, p.z.floor() as i32)
}

/// Built so that targets add nothing and `add_points` reports only hits.
fn hits_only<T: TestVoxelizer>(resolution: f32) -> T {
    T::make(resolution, 1, 0, 0, 1, false)
}

/// Fires one ray of intensity 1 into a [`hits_only`] voxelizer and checks
/// that it hit every voxel from the sensor one up to the target one, one
/// voxel at a time and every step towards the target. Every voxel the ray
/// should hit must hold more than 0.
fn check_ray_path<T: TestVoxelizer>(v: &mut T, resolution: f32, sensor: PointF, point: PointF) {
    let (from, to) = (voxel_of(&sensor, resolution), voxel_of(&point, resolution));
    let distance = |a: Coord, b: Coord| (a.0 - b.0).abs() + (a.1 - b.1).abs() + (a.2 - b.2).abs();
    let mut path: Vec<Coord> = v
        .add_points(sensor.clone(), std::slice::from_ref(&point))
        .into_iter()
        .map(|p| (p.x, p.y, p.z))
        .collect();
    path.sort_by_key(|&c| std::cmp::Reverse(distance(c, to)));
    path.push(to);
    let ctx = format!("sensor {sensor:?}, point {point:?}, path {path:?}");
    assert_eq!(path[0], from, "{ctx}");
    for step in path.windows(2) {
        assert_eq!(distance(step[0], step[1]), 1, "{ctx}");
        assert_eq!(distance(step[0], to), distance(step[1], to) + 1, "{ctx}");
    }
}

pub(crate) fn ray_ends_on_voxel_corners<T: TestVoxelizer>() {
    let mut v = hits_only::<T>(1.0);
    // Enough for every ray to hit the sensor voxels
    seed(&mut v, &box_coords((-3, -3, -3), (3, 3, 3)), 1 << 16);
    for sensor in [
        pf(0.0, 0.0, 0.0, 1.0),
        pf(0.5, 0.5, 0.5, 1.0),
        pf(0.37, 0.61, 0.13, 1.0),
    ] {
        for c in box_coords((-2, -2, -2), (2, 2, 2)) {
            let point = pf(c.0 as f32, c.1 as f32, c.2 as f32, 1.0);
            check_ray_path(&mut v, 1.0, sensor.clone(), point);
        }
    }
}

pub(crate) fn ray_ends_near_voxel_corners<T: TestVoxelizer>() {
    // Points on (or close to) boundaries of several axes, where float errors
    // decide which last crossing comes first
    const RESOLUTION: f32 = 0.1;
    let mut v = hits_only::<T>(RESOLUTION);
    for (sensor, point) in [
        (pf(0.013, 1.007, 0.0047, 1.0), pf(4.0, 0.5, 0.3, 1.0)),
        (pf(0.05, 1.05, 0.05, 1.0), pf(-3.0, 0.0, 2.0, 1.0)),
        (pf(-0.117, 1.008, 0.0047, 1.0), pf(2.0, 0.0, 0.0, 1.0)),
        (
            pf(0.013, 1.007, 0.0047, 1.0),
            pf(4.0137, 0.5213, 0.3341, 1.0),
        ),
        (pf(-0.117, 1.008, 0.0047, 1.0), pf(2.0173, 0.0, 0.0313, 1.0)),
    ] {
        let corners = (voxel_of(&sensor, RESOLUTION), voxel_of(&point, RESOLUTION));
        seed(&mut v, &box_coords(corners.0, corners.1), SEED);
        check_ray_path(&mut v, RESOLUTION, sensor, point);
    }
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
    // The page of (3,0,0) got empty and is removed, the compressed page
    // stays for (7,0,0).
    assert_eq!(get_voxel(&v, 3, 0, 0), None);
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
    // No bit is left, so the compressed page is removed.
    assert_eq!(compressed_word(&v, 3, 0, 0), None);
    assert!(v.compressed_pages().unwrap().is_empty());
    assert_eq!(result, vec![block(3, 0, 0, 0)]);
    // 2 -> 1 is already below: no second destroy.
    assert!(along_x(&mut v, 0, 7, 0, 0, 1.0).is_empty());
    assert_eq!(compressed_word(&v, 3, 0, 0), None);
}

pub(crate) fn compress_emptied_pages_removed<T: TestVoxelizer>() {
    let mut v = compressed::<T>();
    along_x(&mut v, 0, 3, 0, 0, 1.0);
    // Destroys (3,0,0), emptying its page and its compressed page. The
    // target (70,0,0) lies in the next compressed page.
    let result = along_x(&mut v, 3, 70, 0, 0, 1.0);
    assert_eq!(
        sorted(result),
        sorted(vec![
            block(3, 0, 0, 0),
            block(70, 0, 0, block_bit(70, 0, 0))
        ])
    );
    assert_eq!(get_voxel(&v, 3, 0, 0), None);
    assert_eq!(compressed_word(&v, 3, 0, 0), None);
    assert_eq!(compressed_bits(&v), vec![pi(70, 0, 0, 1)]);
    assert_eq!(v.pages().len(), 1);
    assert_eq!(v.compressed_pages().unwrap().len(), 1);
    assert_eq!(v.pending_results(), 0);
}

pub(crate) fn compress_constructed_voxel_keeps_page<T: TestVoxelizer>() {
    // With the destroy threshold above the construct one a voxel can drop to
    // 0 without being destroyed: its page is all 0 but still has a bit set.
    let mut v = T::make(1.0, 1, 1, 5, 3, true);
    along_x(&mut v, 0, 3, 0, 0, 4.0);
    let result = along_x(&mut v, 3, 9, 0, 0, 4.0);
    assert_eq!(result, vec![block(9, 0, 0, block_bit(9, 0, 0))]);
    assert_eq!(get_voxel(&v, 3, 0, 0), Some(0));
    assert_eq!(compressed_word(&v, 3, 0, 0), Some(page_bit(3, 0, 0)));
    assert_eq!(v.pages().len(), 2);
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

pub(crate) fn compress_destroy_without_construct_ignored<T: TestVoxelizer>() {
    // Between the thresholds a voxel is not constructed, so dropping to the
    // destroy threshold from there destroys nothing.
    let mut v = T::make(1.0, 1, 1, 0, 10, true);
    assert!(along_x(&mut v, 0, 3, 0, 0, 5.0).is_empty());
    assert_eq!(get_voxel(&v, 3, 0, 0), Some(5));
    // 5 -> 0 crosses the destroy threshold, (7,0,0) -> 5 stays below the
    // construct one.
    assert!(along_x(&mut v, 0, 7, 0, 0, 5.0).is_empty());
    assert_eq!(get_voxel(&v, 3, 0, 0), None);
    assert_eq!(get_voxel(&v, 7, 0, 0), Some(5));
    assert!(v.compressed_pages().unwrap().is_empty());
    assert_eq!(v.pending_results(), 0);
}

pub(crate) fn compress_destroy_without_construct_keeps_other_bits<T: TestVoxelizer>() {
    // Same, with a constructed voxel in the same page: its bit stays the
    // only one.
    let mut v = T::make(1.0, 1, 1, 0, 10, true);
    assert!(along_x(&mut v, 0, 3, 0, 0, 5.0).is_empty());
    assert_eq!(
        along_x(&mut v, 0, 3, 1, 0, 10.0),
        vec![block(3, 1, 0, block_bit(3, 1, 0))]
    );
    assert!(along_x(&mut v, 0, 7, 0, 0, 5.0).is_empty());
    assert_eq!(get_voxel(&v, 3, 0, 0), Some(0));
    assert_eq!(compressed_word(&v, 3, 0, 0), Some(page_bit(3, 1, 0)));
    assert_eq!(compressed_bits(&v), vec![pi(3, 1, 0, 1)]);
}

pub(crate) fn compress_reconstruct_without_destroy_not_reported<T: TestVoxelizer>() {
    let mut v = T::make(1.0, 1, 1, 2, 5, true);
    assert_eq!(
        along_x(&mut v, 0, 3, 0, 0, 5.0),
        vec![block(3, 0, 0, block_bit(3, 0, 0))]
    );
    // 5 -> 3 stays above the destroy threshold, so it stays constructed.
    assert!(along_x(&mut v, 0, 7, 0, 0, 2.0).is_empty());
    assert_eq!(get_voxel(&v, 3, 0, 0), Some(3));
    // 3 -> 5 crosses the construct threshold, but it is constructed already.
    assert!(along_x(&mut v, 0, 3, 0, 0, 2.0).is_empty());
    assert_eq!(get_voxel(&v, 3, 0, 0), Some(5));
    assert_eq!(compressed_bits(&v), vec![pi(3, 0, 0, 1)]);
    // 5 -> 2 destroys it, (7,0,0) 2 -> 5 gets constructed.
    assert_eq!(
        sorted(along_x(&mut v, 0, 7, 0, 0, 3.0)),
        sorted(vec![block(3, 0, 0, 0), block(7, 0, 0, block_bit(7, 0, 0))])
    );
    assert_eq!(compressed_bits(&v), vec![pi(7, 0, 0, 1)]);
}

// --------------------------------------------------------------- max value

/// [`PLAIN`] with voxel values capped at `max_value`.
fn capped<T: TestVoxelizer>(max_value: u32) -> T {
    T::with_params(VoxelizerParams { max_value, ..PLAIN })
}

pub(crate) fn max_value_caps_target<T: TestVoxelizer>() {
    let mut v = capped::<T>(5);
    assert_eq!(from_above(&mut v, (3, 0, 0), 3.0), vec![pi(3, 0, 0, 3)]);
    assert_eq!(from_above(&mut v, (3, 0, 0), 3.0), vec![pi(3, 0, 0, 5)]);
    // Already at the cap: nothing changes, nothing is reported.
    assert!(from_above(&mut v, (3, 0, 0), 3.0).is_empty());
    assert_eq!(get_voxel(&v, 3, 0, 0), Some(5));
    assert_eq!(v.pending_results(), 0);
}

pub(crate) fn max_value_caps_within_one_call<T: TestVoxelizer>() {
    // The cap applies after the target multiplier: 4 * 1 * 3 = 12 > 10.
    let mut v = T::with_params(VoxelizerParams {
        multipliers: Multipliers {
            target: 3,
            ..UNSCALED
        },
        max_value: 10,
        ..PLAIN
    });
    let points = vec![pf(3.5, 0.5, 0.5, 1.0); 4];
    let result = v.add_points(pf(3.5, 0.5, 1.5, 1.0), &points);
    assert_eq!(result, vec![pi(3, 0, 0, 10)]);
    assert_eq!(nonzero_voxels(&v), vec![pi(3, 0, 0, 10)]);
}

pub(crate) fn max_value_voxel_not_permanent<T: TestVoxelizer>() {
    // However often a voxel is seen, as many hits as the cap clear it.
    let mut v = capped::<T>(3);
    for _ in 0..10 {
        from_above(&mut v, (3, 0, 0), 1.0);
    }
    assert_eq!(get_voxel(&v, 3, 0, 0), Some(3));
    for value in [2, 1] {
        assert_eq!(
            sorted(along_x(&mut v, 3, 5, 0, 0, 1.0)),
            vec![pi(3, 0, 0, value), pi(5, 0, 0, 3 - value)]
        );
    }
    // The last hit empties the page of (3,0,0), which gets removed.
    let result = along_x(&mut v, 3, 4, 0, 0, 1.0);
    assert_eq!(sorted(result), vec![pi(3, 0, 0, 0), pi(4, 0, 0, 1)]);
    assert!(page(&v, (0, 0, 0)).is_none());
    assert_eq!(v.pages().len(), 1);
    assert_eq!(v.pending_results(), 0);
}

pub(crate) fn max_value_saturates_instead_of_overflowing<T: TestVoxelizer>() {
    let mut v = plain::<T>();
    v.set_voxel(3, 0, 0, u32::MAX - 1);
    assert_eq!(
        from_above(&mut v, (3, 0, 0), 5.0),
        vec![pi(3, 0, 0, u32::MAX)]
    );
    assert!(from_above(&mut v, (3, 0, 0), 5.0).is_empty());
    assert_eq!(get_voxel(&v, 3, 0, 0), Some(u32::MAX));
}

pub(crate) fn max_value_with_compress<T: TestVoxelizer>() {
    let mut v = T::with_params(VoxelizerParams {
        destroy_threshold: 2,
        construct_threshold: 3,
        max_value: 3,
        compress: true,
        ..PLAIN
    });
    assert!(from_above(&mut v, (3, 0, 0), 2.0).is_empty());
    // Reaching the cap constructs.
    assert_eq!(
        from_above(&mut v, (3, 0, 0), 2.0),
        vec![block(3, 0, 0, block_bit(3, 0, 0))]
    );
    for _ in 0..10 {
        assert!(from_above(&mut v, (3, 0, 0), 2.0).is_empty());
    }
    assert_eq!(get_voxel(&v, 3, 0, 0), Some(3));
    // A single hit destroys it again.
    let result = along_x(&mut v, 3, 5, 0, 0, 1.0);
    assert_eq!(result, vec![block(3, 0, 0, 0)]);
    assert_eq!(get_voxel(&v, 3, 0, 0), Some(2));
    assert!(compressed_bits(&v).is_empty());
}

// ---------------------------------------------------- intensity multiplier

/// [`PLAIN`] with intensities scaled by `intensity_multiplier` and the
/// given hit and target multipliers.
fn scaled<T: TestVoxelizer>(
    intensity_multiplier: f32,
    hit_multiplier: u32,
    target_multiplier: u32,
) -> T {
    T::with_params(VoxelizerParams {
        multipliers: Multipliers {
            intensity: intensity_multiplier,
            hit: hit_multiplier,
            target: target_multiplier,
        },
        ..PLAIN
    })
}

pub(crate) fn intensity_multiplier_scales_target<T: TestVoxelizer>() {
    // (3 * 2.5) as u32 * 2 = 7 * 2
    let mut v = scaled::<T>(2.5, 1, 2);
    assert_eq!(from_above(&mut v, (3, 0, 0), 3.0), vec![pi(3, 0, 0, 14)]);
    assert_eq!(get_voxel(&v, 3, 0, 0), Some(14));
}

pub(crate) fn intensity_multiplier_scales_hit<T: TestVoxelizer>() {
    // (3 * 2.5) as u32 = 7, times 2 subtracted and times 1 added.
    let mut v = scaled::<T>(2.5, 2, 1);
    v.set_voxel(0, 0, 0, 20);
    let result = v.add_points(pf(0.5, 0.5, 0.5, 1.0), &[pf(3.5, 0.5, 0.5, 3.0)]);
    assert_eq!(sorted(result), vec![pi(0, 0, 0, 20 - 14), pi(3, 0, 0, 7)]);
    assert_eq!(get_voxel(&v, 0, 0, 0), Some(6));
    assert_eq!(get_voxel(&v, 3, 0, 0), Some(7));
}

pub(crate) fn intensity_multiplier_truncates_after_scaling<T: TestVoxelizer>() {
    // (0.5 * 3) as u32 = 1, truncating first would give 0 * 3.
    let mut v = scaled::<T>(3.0, 1, 2);
    v.set_voxel(0, 0, 0, SEED);
    let result = v.add_points(pf(0.5, 0.5, 0.5, 1.0), &[pf(3.5, 0.5, 0.5, 0.5)]);
    assert_eq!(sorted(result), vec![pi(0, 0, 0, SEED - 1), pi(3, 0, 0, 2)]);
}

pub(crate) fn intensity_multiplier_zero_or_negative<T: TestVoxelizer>() {
    // Scaled intensities below zero saturate at it.
    for intensity_multiplier in [0.0, -1.0] {
        let mut v = scaled::<T>(intensity_multiplier, 1, 1);
        v.set_voxel(0, 0, 0, SEED);
        let result = v.add_points(pf(0.5, 0.5, 0.5, 1.0), &[pf(3.5, 0.5, 0.5, 5.0)]);
        assert!(
            result.is_empty(),
            "intensity_multiplier = {intensity_multiplier}"
        );
        assert_eq!(get_voxel(&v, 0, 0, 0), Some(SEED));
        assert_eq!(get_voxel(&v, 3, 0, 0), Some(0));
        assert_eq!(v.pending_results(), 0);
    }
}

// --------------------------------------------------------- set multipliers

/// Multipliers different from [`UNSCALED`] and [`SWITCHED`] in every field.
const OTHER: Multipliers = Multipliers {
    intensity: 2.5,
    hit: 2,
    target: 3,
};

pub(crate) fn multipliers_returns_constructor_values<T: TestVoxelizer>() {
    assert_eq!(plain::<T>().multipliers(), UNSCALED);
    let v = T::with_params(VoxelizerParams {
        multipliers: OTHER,
        ..PLAIN
    });
    assert_eq!(v.multipliers(), OTHER);
}

pub(crate) fn set_multipliers_roundtrip<T: TestVoxelizer>() {
    let mut v = plain::<T>();
    for multipliers in [
        OTHER,
        Multipliers {
            intensity: 0.0,
            hit: 0,
            target: 0,
        },
        Multipliers {
            intensity: -1.5,
            hit: u32::MAX,
            target: u32::MAX,
        },
        UNSCALED,
    ] {
        v.set_multipliers(multipliers);
        assert_eq!(v.multipliers(), multipliers);
    }
    // The last of several sets wins.
    v.set_multipliers(OTHER);
    v.set_multipliers(SWITCHED);
    assert_eq!(v.multipliers(), SWITCHED);
}

pub(crate) fn multipliers_unchanged_by_add_points<T: TestVoxelizer>() {
    for compress in [false, true] {
        let mut v = T::with_params(VoxelizerParams {
            multipliers: OTHER,
            compress,
            ..PLAIN
        });
        along_x(&mut v, 0, 3, 0, 0, 1.0);
        along_x(&mut v, 0, 7, 0, 0, 2.0);
        assert_eq!(v.multipliers(), OTHER, "compress = {compress}");
        v.set_multipliers(SWITCHED);
        along_x(&mut v, 0, 7, 0, 0, 2.0);
        assert_eq!(v.multipliers(), SWITCHED, "compress = {compress}");
    }
}

pub(crate) fn set_multipliers_applies_to_next_call<T: TestVoxelizer>() {
    let mut v = plain::<T>();
    v.set_voxel(0, 0, 0, 20);
    let sensor = pf(0.5, 0.5, 0.5, 1.0);
    let points = [pf(3.5, 0.5, 0.5, 3.0)];
    assert_eq!(
        sorted(v.add_points(sensor.clone(), &points)),
        vec![pi(0, 0, 0, 17), pi(3, 0, 0, 3)]
    );
    v.set_multipliers(OTHER);
    // (3 * 2.5) as u32 = 7, times 2 subtracted from and times 3 added to what
    // the voxels hold already.
    assert_eq!(
        sorted(v.add_points(sensor.clone(), &points)),
        vec![pi(0, 0, 0, 17 - 7 * 2), pi(3, 0, 0, 3 + 7 * 3)]
    );
    // Setting the old ones back restores the old behaviour.
    v.set_multipliers(UNSCALED);
    assert_eq!(
        sorted(v.add_points(sensor, &points)),
        vec![pi(0, 0, 0, 0), pi(3, 0, 0, 24 + 3)]
    );
    assert_eq!(nonzero_voxels(&v), vec![pi(3, 0, 0, 27)]);
    assert_eq!(v.pending_results(), 0);
}

pub(crate) fn set_multipliers_keeps_voxels<T: TestVoxelizer>() {
    for compress in [false, true] {
        let mut v = T::with_params(VoxelizerParams { compress, ..PLAIN });
        v.set_voxel(0, 0, 0, SEED);
        along_x(&mut v, 0, 3, 0, 0, 2.0);
        let pages = v.pages();
        let compressed_pages = v.compressed_pages();
        v.set_multipliers(OTHER);
        assert_eq!(v.pages(), pages, "compress = {compress}");
        assert_eq!(
            v.compressed_pages(),
            compressed_pages,
            "compress = {compress}"
        );
        // Nothing to report until points are added.
        assert!(v.add_points(pf(0.5, 0.5, 0.5, 1.0), &[]).is_empty());
        assert_eq!(v.pages(), pages, "compress = {compress}");
        assert_eq!(v.pending_results(), 0);
    }
}

pub(crate) fn set_intensity_multiplier_only<T: TestVoxelizer>() {
    let mut v = plain::<T>();
    v.set_voxel(0, 0, 0, SEED);
    v.set_multipliers(Multipliers {
        intensity: 0.5,
        ..UNSCALED
    });
    // (5 * 0.5) as u32 = 2 on both sides.
    let result = v.add_points(pf(0.5, 0.5, 0.5, 1.0), &[pf(3.5, 0.5, 0.5, 5.0)]);
    assert_eq!(sorted(result), vec![pi(0, 0, 0, SEED - 2), pi(3, 0, 0, 2)]);
}

pub(crate) fn set_hit_multiplier_only<T: TestVoxelizer>() {
    let mut v = plain::<T>();
    v.set_voxel(0, 0, 0, SEED);
    v.set_multipliers(Multipliers { hit: 4, ..UNSCALED });
    let result = v.add_points(pf(0.5, 0.5, 0.5, 1.0), &[pf(3.5, 0.5, 0.5, 2.0)]);
    assert_eq!(
        sorted(result),
        vec![pi(0, 0, 0, SEED - 2 * 4), pi(3, 0, 0, 2)]
    );
}

pub(crate) fn set_target_multiplier_only<T: TestVoxelizer>() {
    let mut v = plain::<T>();
    v.set_voxel(0, 0, 0, SEED);
    v.set_multipliers(Multipliers {
        target: 4,
        ..UNSCALED
    });
    let result = v.add_points(pf(0.5, 0.5, 0.5, 1.0), &[pf(3.5, 0.5, 0.5, 2.0)]);
    assert_eq!(
        sorted(result),
        vec![pi(0, 0, 0, SEED - 2), pi(3, 0, 0, 2 * 4)]
    );
}

pub(crate) fn set_hit_multiplier_zero_spares_hit_voxels<T: TestVoxelizer>() {
    let mut v = plain::<T>();
    let hits = [(0, 0, 0), (1, 0, 0), (2, 0, 0)];
    seed(&mut v, &hits, SEED);
    v.set_multipliers(Multipliers { hit: 0, ..UNSCALED });
    // Rays pass through without changing (or reporting) anything.
    assert_eq!(along_x(&mut v, 0, 3, 0, 0, 5.0), vec![pi(3, 0, 0, 5)]);
    for &(x, y, z) in &hits {
        assert_eq!(get_voxel(&v, x, y, z), Some(SEED), "voxel ({x}, {y}, {z})");
    }
    assert_eq!(v.pending_results(), 0);
}

pub(crate) fn set_multipliers_with_compress<T: TestVoxelizer>() {
    let mut v = T::make(1.0, 1, 1, 2, 6, true);
    assert!(from_above(&mut v, (3, 0, 0), 2.0).is_empty());
    // 2 + 2 * 3 = 8 reaches the construct threshold, 2 + 2 would not.
    v.set_multipliers(Multipliers {
        target: 3,
        ..UNSCALED
    });
    assert_eq!(
        from_above(&mut v, (3, 0, 0), 2.0),
        vec![block(3, 0, 0, block_bit(3, 0, 0))]
    );
    assert_eq!(get_voxel(&v, 3, 0, 0), Some(8));
    // A single hit of 6 destroys it, 8 - 1 would not.
    v.set_multipliers(Multipliers { hit: 6, ..UNSCALED });
    assert_eq!(along_x(&mut v, 3, 5, 0, 0, 1.0), vec![block(3, 0, 0, 0)]);
    assert_eq!(get_voxel(&v, 3, 0, 0), Some(2));
    assert!(compressed_bits(&v).is_empty());
}

pub(crate) fn set_multipliers_same_as_constructed<T: TestVoxelizer>() {
    for params in SCENARIO_PARAMS {
        for multipliers in [
            SWITCHED,
            Multipliers {
                intensity: 0.5,
                hit: 1,
                target: 4,
            },
        ] {
            let mut set = T::with_params(params);
            set.set_multipliers(multipliers);
            let constructed = T::with_params(VoxelizerParams {
                multipliers,
                ..params
            });
            assert_snapshot_eq(
                &scenario(set),
                &scenario(constructed),
                &format!("{params:?}, set to {multipliers:?}"),
            );
        }
    }
}

// ---------------------------------------------------------------- scenario

pub(crate) fn scenario_repeatable<T: TestVoxelizer>() {
    for params in SCENARIO_PARAMS {
        let a = scenario(T::with_params(params));
        assert!(a.results.iter().all(|r| !r.is_empty()));
        let b = scenario(T::with_params(params));
        assert_snapshot_eq(&a, &b, &format!("{params:?}"));
    }
}

pub(crate) fn switched_scenario_repeatable<T: TestVoxelizer>() {
    for params in SCENARIO_PARAMS {
        let a = scenario_with(T::with_params(params), Some(SWITCHED));
        assert!(a.results.iter().all(|r| !r.is_empty()));
        // Switching changes the outcome of call 2 only.
        let unswitched = scenario(T::with_params(params));
        assert_eq!(a.results[0], unswitched.results[0], "{params:?}");
        assert_ne!(a.results[1], unswitched.results[1], "{params:?}");
        let b = scenario_with(T::with_params(params), Some(SWITCHED));
        assert_snapshot_eq(&a, &b, &format!("{params:?}"));
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
    use crate::{MultiThreaded, SingleThreaded};
    for switch in [None, Some(SWITCHED)] {
        for params in SCENARIO_PARAMS {
            let expected = scenario_with(SingleThreaded::new(params), switch);
            assert!(expected.results.iter().all(|r| !r.is_empty()));
            for threads in [1, 2, 4, 8] {
                let got = scenario_with(
                    MultiThreaded::new(params, threads).expect("thread pool must build"),
                    switch,
                );
                assert_snapshot_eq(
                    &got,
                    &expected,
                    &format!("{params:?}, switch = {switch:?}, threads = {threads}"),
                );
            }
        }
    }
}

#[cfg(all(feature = "single_threaded", feature = "wgpu"))]
#[test]
fn single_and_wgpu_agree() {
    use crate::{SingleThreaded, WgpuAccelerated};
    for switch in [None, Some(SWITCHED)] {
        for params in SCENARIO_PARAMS {
            let expected = scenario_with(SingleThreaded::new(params), switch);
            assert!(expected.results.iter().all(|r| !r.is_empty()));
            let got = scenario_with(
                pollster::block_on(WgpuAccelerated::new(params, 1 << 12))
                    .expect("a GPU adapter with compute shaders is required"),
                switch,
            );
            assert_snapshot_eq(&got, &expected, &format!("{params:?}, switch = {switch:?}"));
        }
    }
}
