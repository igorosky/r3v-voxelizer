/*
 * Test cases for the C bindings, ported from the shared Rust tests in
 * src/test.rs. Only what the C API exposes is checked: the updates returned
 * by add_points. Where the Rust tests read the voxel grid directly, a
 * follow-up probe ray is used instead.
 *
 * Per implementation cases run against every entry of IMPLS (harness.c).
 */
#include <stdio.h>

#include "harness.h"

static const PointF SENSOR = {0.5f, 0.5f, 0.5f, 1.0f};

/* Seeds voxel (x, y, z) with `value` at resolution 1 (target multiplier 1). */
static void seed_at(void *v, int32_t x, int32_t y, int32_t z, uint32_t value) {
    seed(v, 1.0f, x, y, z, (float)value);
}

static bool same_point(PointI a, PointI b) {
    return a.x == b.x && a.y == b.y && a.z == b.z && a.i == b.i;
}

/* ------------------------------------------------------- lifecycle */

static void create_returns_non_null(const Impl *impl) {
    drop_voxelizer(plain(impl));
    drop_voxelizer(compressed(impl));
    drop_voxelizer(make(impl, 0.05f, 2, 3, 4, 5, true));
}

static void create_drop_cycles(const Impl *impl) {
    for (int k = 0; k < 50; ++k) {
        void *v = make(impl, 1.0f, 1, 1, 0, 1, k % 2 == 0);
        VoxelsUpdate update = add_point(v, SENSOR, pf(3.5f, 0.5f, 0.5f, 1.0f));
        if (k % 2 == 0) {
            EXPECT_VOXELS(update, pi(0, 0, 0, 1u << 3));
        } else {
            EXPECT_VOXELS(update, pi(3, 0, 0, 1));
        }
        drop_voxelizer(v);
    }
}

/* ------------------------------------------------ add_points basics */

static void basic_1d(const Impl *impl) {
    void *v = plain(impl);
    EXPECT_VOXELS(add_point(v, pf(0.0f, 0.0f, 0.0f, 1.0f), pf(5.0f, 0.0f, 0.0f, 1.0f)),
                  pi(5, 0, 0, 1));
    drop_voxelizer(v);
}

static void empty_points_no_change(const Impl *impl) {
    void *v = plain(impl);
    PointF unused = pf(3.5f, 0.5f, 0.5f, 1.0f);
    EXPECT_NO_VOXELS(add_points(v, SENSOR, &unused, 0));
    /* Nothing was stored, so the target still starts from zero. */
    EXPECT_VOXELS(add_point(v, SENSOR, unused), pi(3, 0, 0, 1));
    drop_voxelizer(v);
}

static void returns_absolute_value_not_delta(const Impl *impl) {
    void *v = plain(impl);
    for (int32_t x = 0; x < 5; ++x) {
        seed_at(v, x, 0, 0, SEED);
    }
    seed_at(v, 5, 0, 0, 3);
    EXPECT_VOXELS(add_point(v, SENSOR, pf(5.5f, 0.5f, 0.5f, 2.0f)), pi(0, 0, 0, SEED - 2),
                  pi(1, 0, 0, SEED - 2), pi(2, 0, 0, SEED - 2), pi(3, 0, 0, SEED - 2),
                  pi(4, 0, 0, SEED - 2), pi(5, 0, 0, 5));
    drop_voxelizer(v);
}

static void sensor_intensity_ignored(const Impl *impl) {
    void *v = plain(impl);
    seed_at(v, 0, 0, 0, SEED);
    /* Hits use the point's intensity, not the sensor's. */
    EXPECT_VOXELS(add_point(v, pf(0.5f, 0.5f, 0.5f, 99.0f), pf(3.5f, 0.5f, 0.5f, 1.0f)),
                  pi(0, 0, 0, SEED - 1), pi(3, 0, 0, 1));
    drop_voxelizer(v);
}

static void intensity_truncated_toward_zero(const Impl *impl) {
    void *v = plain(impl);
    EXPECT_VOXELS(add_point(v, SENSOR, pf(3.5f, 0.5f, 0.5f, 2.9f)), pi(3, 0, 0, 2));
    drop_voxelizer(v);
}

static void zero_intensity_reports_nothing(const Impl *impl) {
    void *v = plain(impl);
    seed_at(v, 0, 0, 0, SEED);
    EXPECT_NO_VOXELS(add_point(v, SENSOR, pf(3.5f, 0.5f, 0.5f, 0.0f)));
    /* Probe: both voxels kept their values. */
    EXPECT_VOXELS(add_point(v, SENSOR, pf(3.5f, 0.5f, 0.5f, 1.0f)), pi(0, 0, 0, SEED - 1),
                  pi(3, 0, 0, 1));
    drop_voxelizer(v);
}

static void resolution_scales_coords_not_intensity(const Impl *impl) {
    void *v = make(impl, 0.25f, 1, 1, 0, 1, false);
    for (int32_t x = 0; x < 4; ++x) {
        seed(v, 0.25f, x, 0, 0, (float)SEED);
    }
    EXPECT_VOXELS(add_point(v, pf(0.1f, 0.1f, 0.1f, 1.0f), pf(1.0f, 0.1f, 0.1f, 7.0f)),
                  pi(0, 0, 0, SEED - 7), pi(1, 0, 0, SEED - 7), pi(2, 0, 0, SEED - 7),
                  pi(3, 0, 0, SEED - 7), pi(4, 0, 0, 7));
    drop_voxelizer(v);

    v = make(impl, 2.0f, 1, 1, 0, 1, false);
    EXPECT_VOXELS(add_point(v, pf(0.0f, 0.0f, 0.0f, 1.0f), pf(5.0f, 0.0f, 0.0f, 7.0f)),
                  pi(2, 0, 0, 7));
    drop_voxelizer(v);
}

static void resolution_scales_sensor_too(const Impl *impl) {
    void *v = make(impl, 2.0f, 1, 1, 0, 1, false);
    seed(v, 2.0f, 1, 0, 0, (float)SEED);
    seed(v, 2.0f, 2, 0, 0, (float)SEED);
    /* sensor 2.0 -> 1.0 -> voxel 1, point 7.0 -> 3.5 -> voxel 3 */
    EXPECT_VOXELS(add_point(v, pf(2.0f, 0.0f, 0.0f, 1.0f), pf(7.0f, 0.0f, 0.0f, 1.0f)),
                  pi(1, 0, 0, SEED - 1), pi(2, 0, 0, SEED - 1), pi(3, 0, 0, 1));
    drop_voxelizer(v);
}

static void large_coords(const Impl *impl) {
    void *v = plain(impl);
    PointF sensor = pf(1000.5f, -2000.5f, 29999.5f, 1.0f);
    PointF point = pf(1000.5f, -2000.5f, 30000.5f, 4.0f);
    EXPECT_VOXELS(add_point(v, sensor, point), pi(1000, -2001, 30000, 4));
    /* Probe: the value was stored where it was reported. */
    EXPECT_VOXELS(add_point(v, sensor, point), pi(1000, -2001, 30000, 8));
    drop_voxelizer(v);
}

static void negative_coords_use_floor(const Impl *impl) {
    static const struct {
        float x;
        int32_t voxel;
    } cases[] = {{-0.5f, -1}, {-1.0f, -1}, {-1.5f, -2}};
    for (size_t k = 0; k < ARRAY_LEN(cases); ++k) {
        void *v = plain(impl);
        EXPECT_VOXELS(add_point(v, pf(-3.5f, 0.5f, 0.5f, 1.0f), pf(cases[k].x, 0.5f, 0.5f, 1.0f)),
                      pi(cases[k].voxel, 0, 0, 1));
        drop_voxelizer(v);
    }
}

/* ---------------------------------------- multipliers / saturation */

static void hit_multiplier_scales_hits(const Impl *impl) {
    void *v = make(impl, 1.0f, 3, 1, 0, 1, false);
    seed_at(v, 0, 0, 0, SEED);
    EXPECT_VOXELS(add_point(v, SENSOR, pf(3.5f, 0.5f, 0.5f, 2.0f)), pi(0, 0, 0, SEED - 2 * 3),
                  pi(3, 0, 0, 2));
    drop_voxelizer(v);
}

static void target_multiplier_scales_target(const Impl *impl) {
    void *v = make(impl, 1.0f, 1, 5, 0, 1, false);
    /* Seeding goes through the target path too: 2 * 5 = SEED. */
    seed(v, 1.0f, 0, 0, 0, (float)(SEED / 5));
    EXPECT_VOXELS(add_point(v, SENSOR, pf(3.5f, 0.5f, 0.5f, 2.0f)), pi(0, 0, 0, SEED - 2),
                  pi(3, 0, 0, 2 * 5));
    drop_voxelizer(v);
}

static void both_multipliers(const Impl *impl) {
    void *v = make(impl, 1.0f, 2, 3, 0, 1, false);
    /* Seeds (0,0,0) with 4 * 3 = 12. */
    seed(v, 1.0f, 0, 0, 0, 4.0f);
    PointF point = pf(3.5f, 0.5f, 0.5f, 2.0f);
    EXPECT_VOXELS(add_point(v, SENSOR, point), pi(0, 0, 0, 12 - 2 * 2), pi(3, 0, 0, 2 * 3));
    /* Probe: values were stored. */
    EXPECT_VOXELS(add_point(v, SENSOR, point), pi(0, 0, 0, 12 - 2 * 2 * 2), pi(3, 0, 0, 2 * 3 * 2));
    drop_voxelizer(v);
}

static void hit_saturates_at_zero(const Impl *impl) {
    void *v = plain(impl);
    seed_at(v, 0, 0, 0, 1);
    /* (1,0,0) and (2,0,0) exist (same page) and hold 0. 1 -> 0 is a change
     * and gets reported as 0; 0 -> 0 is not. */
    EXPECT_VOXELS(add_point(v, SENSOR, pf(3.5f, 0.5f, 0.5f, 5.0f)), pi(0, 0, 0, 0),
                  pi(3, 0, 0, 5));
    /* Probe: nothing wrapped around below zero. */
    EXPECT_VOXELS(add_point(v, SENSOR, pf(3.5f, 0.5f, 0.5f, 1.0f)), pi(3, 0, 0, 6));
    drop_voxelizer(v);

    /* The multiplier saturates too. */
    v = make(impl, 1.0f, 100, 1, 0, 1, false);
    seed_at(v, 0, 0, 0, SEED);
    EXPECT_VOXELS(add_point(v, SENSOR, pf(3.5f, 0.5f, 0.5f, 1.0f)), pi(0, 0, 0, 0),
                  pi(3, 0, 0, 1));
    drop_voxelizer(v);
}

static void hit_on_zero_voxel_not_reported(const Impl *impl) {
    void *v = plain(impl);
    /* Creates page (0,0,0) without touching the ray below. */
    seed_at(v, 1, 1, 1, SEED);
    EXPECT_VOXELS(add_point(v, SENSOR, pf(3.5f, 0.5f, 0.5f, 1.0f)), pi(3, 0, 0, 1));
    drop_voxelizer(v);
}

static void thresholds_ignored_without_compress(const Impl *impl) {
    void *v = make(impl, 1.0f, 1, 1, 100, 200, false);
    for (int32_t x = 0; x < 5; ++x) {
        seed_at(v, x, 0, 0, SEED);
    }
    seed_at(v, 5, 0, 0, 3);
    EXPECT_VOXELS(add_point(v, SENSOR, pf(5.5f, 0.5f, 0.5f, 2.0f)), pi(0, 0, 0, SEED - 2),
                  pi(1, 0, 0, SEED - 2), pi(2, 0, 0, SEED - 2), pi(3, 0, 0, SEED - 2),
                  pi(4, 0, 0, SEED - 2), pi(5, 0, 0, 5));
    drop_voxelizer(v);
}

/* ---------------------------------- accumulation / change detection */

static void accumulates_across_calls(const Impl *impl) {
    void *v = plain(impl);
    for (int32_t x = 0; x < 3; ++x) {
        seed_at(v, x, 0, 0, SEED);
    }
    PointF point = pf(3.5f, 0.5f, 0.5f, 1.0f);
    EXPECT_VOXELS(add_point(v, SENSOR, point), pi(0, 0, 0, SEED - 1), pi(1, 0, 0, SEED - 1),
                  pi(2, 0, 0, SEED - 1), pi(3, 0, 0, 1));
    EXPECT_VOXELS(add_point(v, SENSOR, point), pi(0, 0, 0, SEED - 2), pi(1, 0, 0, SEED - 2),
                  pi(2, 0, 0, SEED - 2), pi(3, 0, 0, 2));
    drop_voxelizer(v);
}

static void unchanged_voxel_not_reported(const Impl *impl) {
    void *v = plain(impl);
    seed_at(v, 0, 0, 0, 5);
    seed_at(v, 1, 0, 0, 1);
    /* First ray hits (1,0,0): 1 -> 0, second ray ends there: 0 -> 1.
     * (The other order gives 1 -> 2 -> 1: same outcome.) */
    PointF points[] = {pf(2.5f, 0.5f, 0.5f, 1.0f), pf(1.5f, 0.5f, 0.5f, 1.0f)};
    EXPECT_VOXELS(add_points(v, SENSOR, points, ARRAY_LEN(points)), pi(0, 0, 0, 3),
                  pi(2, 0, 0, 1));
    /* Probe: (1,0,0) holds 1, so one more hit takes it to 0. */
    EXPECT_VOXELS(add_point(v, pf(1.5f, 0.5f, 0.5f, 1.0f), pf(1.5f, 1.5f, 0.5f, 1.0f)),
                  pi(1, 0, 0, 0), pi(1, 1, 0, 1));
    drop_voxelizer(v);
}

static void second_call_reports_only_its_changes(const Impl *impl) {
    void *v = plain(impl);
    for (int32_t x = 0; x < 3; ++x) {
        seed_at(v, x, 0, 0, SEED);
    }
    /* Straight up along z: hits (0,0,0), (0,0,1), (0,0,2). */
    EXPECT_VOXELS(add_point(v, SENSOR, pf(0.5f, 0.5f, 3.5f, 1.0f)), pi(0, 0, 0, SEED - 1),
                  pi(0, 0, 3, 1));
    EXPECT_VOXELS(add_point(v, SENSOR, pf(3.5f, 0.5f, 0.5f, 1.0f)), pi(0, 0, 0, SEED - 2),
                  pi(1, 0, 0, SEED - 1), pi(2, 0, 0, SEED - 1), pi(3, 0, 0, 1));
    drop_voxelizer(v);
}

static void multiple_points_one_call(const Impl *impl) {
    void *v = plain(impl);
    seed_at(v, 0, 0, 0, SEED);
    seed_at(v, 1, 0, 0, SEED);
    seed_at(v, 2, 0, 0, SEED);
    seed_at(v, 0, 1, 0, SEED);
    seed_at(v, 0, 2, 0, SEED);
    PointF points[] = {pf(3.5f, 0.5f, 0.5f, 1.0f), pf(0.5f, 3.5f, 0.5f, 2.0f)};
    EXPECT_VOXELS(add_points(v, SENSOR, points, ARRAY_LEN(points)), pi(0, 0, 0, SEED - 1 - 2),
                  pi(1, 0, 0, SEED - 1), pi(2, 0, 0, SEED - 1), pi(3, 0, 0, 1),
                  pi(0, 1, 0, SEED - 2), pi(0, 2, 0, SEED - 2), pi(0, 3, 0, 2));
    drop_voxelizer(v);
}

/* ---------------------------------------------------------- paging */

static void hits_skip_missing_pages(const Impl *impl) {
    void *v = plain(impl);
    seed_at(v, 0, 0, 0, SEED);
    PointF point = pf(20.5f, 0.5f, 0.5f, 1.0f);
    /* The ray crosses fine pages 0..=5 along x but only page 0 exists. */
    EXPECT_VOXELS(add_point(v, SENSOR, point), pi(0, 0, 0, SEED - 1), pi(20, 0, 0, 1));
    EXPECT_VOXELS(add_point(v, SENSOR, point), pi(0, 0, 0, SEED - 2), pi(20, 0, 0, 2));
    /* Once a page exists on the way its voxels do get hit. */
    seed_at(v, 8, 0, 0, SEED);
    EXPECT_VOXELS(add_point(v, SENSOR, point), pi(0, 0, 0, SEED - 3), pi(8, 0, 0, SEED - 1),
                  pi(20, 0, 0, 3));
    drop_voxelizer(v);
}

static void negative_page(const Impl *impl) {
    void *v = plain(impl);
    PointF sensor = pf(-0.5f, -0.5f, -0.5f, 1.0f);
    PointF point = pf(-3.5f, -0.5f, -0.5f, 3.0f);
    EXPECT_VOXELS(add_point(v, sensor, point), pi(-4, -1, -1, 3));
    /* Probe: the page now exists, the voxels on the way hold 0. */
    EXPECT_VOXELS(add_point(v, sensor, point), pi(-4, -1, -1, 6));
    drop_voxelizer(v);
}

/* -------------------------------------------------- ray march arms */

static void axis_x_negative(const Impl *impl) {
    CHECK_RAY(impl, pf(5.5f, 0.5f, 0.5f, 1.0f), pf(0.5f, 0.5f, 0.5f, 1.0f), ((Coord){0, 0, 0}),
              {5, 0, 0}, {4, 0, 0}, {3, 0, 0}, {2, 0, 0}, {1, 0, 0});
}

static void axis_y_positive(const Impl *impl) {
    CHECK_RAY(impl, pf(0.5f, 0.5f, 0.5f, 1.0f), pf(0.5f, 4.5f, 0.5f, 1.0f), ((Coord){0, 4, 0}),
              {0, 0, 0}, {0, 1, 0}, {0, 2, 0}, {0, 3, 0});
}

static void axis_y_negative(const Impl *impl) {
    CHECK_RAY(impl, pf(0.5f, 4.5f, 0.5f, 1.0f), pf(0.5f, 0.5f, 0.5f, 1.0f), ((Coord){0, 0, 0}),
              {0, 4, 0}, {0, 3, 0}, {0, 2, 0}, {0, 1, 0});
}

static void axis_z_positive(const Impl *impl) {
    CHECK_RAY(impl, pf(0.5f, 0.5f, 0.5f, 1.0f), pf(0.5f, 0.5f, 4.5f, 1.0f), ((Coord){0, 0, 4}),
              {0, 0, 0}, {0, 0, 1}, {0, 0, 2}, {0, 0, 3});
}

static void axis_z_negative(const Impl *impl) {
    CHECK_RAY(impl, pf(0.5f, 0.5f, 4.5f, 1.0f), pf(0.5f, 0.5f, 0.5f, 1.0f), ((Coord){0, 0, 0}),
              {0, 0, 4}, {0, 0, 3}, {0, 0, 2}, {0, 0, 1});
}

static void plane_xy_forward(const Impl *impl) {
    CHECK_RAY(impl, pf(0.5f, 0.5f, 0.5f, 1.0f), pf(3.5f, 1.7f, 0.5f, 1.0f), ((Coord){3, 1, 0}),
              {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {2, 1, 0});
}

static void plane_xy_reverse(const Impl *impl) {
    CHECK_RAY(impl, pf(3.5f, 1.7f, 0.5f, 1.0f), pf(0.5f, 0.5f, 0.5f, 1.0f), ((Coord){0, 0, 0}),
              {3, 1, 0}, {2, 1, 0}, {1, 1, 0}, {1, 0, 0});
}

static void plane_xz(const Impl *impl) {
    CHECK_RAY(impl, pf(0.5f, 0.5f, 0.5f, 1.0f), pf(3.5f, 0.5f, 1.7f, 1.0f), ((Coord){3, 0, 1}),
              {0, 0, 0}, {1, 0, 0}, {1, 0, 1}, {2, 0, 1});
}

static void plane_yz(const Impl *impl) {
    CHECK_RAY(impl, pf(0.5f, 0.5f, 0.5f, 1.0f), pf(0.5f, 3.5f, 1.7f, 1.0f), ((Coord){0, 3, 1}),
              {0, 0, 0}, {0, 1, 0}, {0, 1, 1}, {0, 2, 1});
}

static void full_3d_forward(const Impl *impl) {
    CHECK_RAY(impl, pf(0.5f, 0.5f, 0.5f, 1.0f), pf(2.5f, 1.6f, 3.5f, 1.0f), ((Coord){2, 1, 3}),
              {0, 0, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {1, 1, 2}, {2, 1, 2});
}

static void full_3d_reverse(const Impl *impl) {
    CHECK_RAY(impl, pf(2.5f, 1.6f, 3.5f, 1.0f), pf(0.5f, 0.5f, 0.5f, 1.0f), ((Coord){0, 0, 0}),
              {2, 1, 3}, {2, 1, 2}, {1, 1, 2}, {1, 1, 1}, {1, 0, 1}, {0, 0, 1});
}

static void sensor_on_grid_line_2d(const Impl *impl) {
    /* Line y = 0.6x from the origin: crosses x=1 at t=0.4 (y=0.6), y=1 at
     * t=0.667 (x=1.67), x=2 at t=0.8. */
    CHECK_RAY(impl, pf(0.0f, 0.0f, 0.0f, 1.0f), pf(2.5f, 1.5f, 0.5f, 1.0f), ((Coord){2, 1, 0}),
              {0, 0, 0}, {1, 0, 0}, {1, 1, 0});
}

static void ray_across_page_boundary(const Impl *impl) {
    /* Hits span fine pages (0,0,0) and (1,0,0) along x. */
    CHECK_RAY(impl, pf(2.5f, 0.5f, 0.5f, 1.0f), pf(6.5f, 1.7f, 0.5f, 1.0f), ((Coord){6, 1, 0}),
              {2, 0, 0}, {3, 0, 0}, {4, 0, 0}, {4, 1, 0}, {5, 1, 0});
}

/* ----------------------------------- point inside the sensor voxel */

static void point_in_sensor_voxel_fresh(const Impl *impl) {
    void *v = plain(impl);
    EXPECT_VOXELS(add_point(v, SENSOR, pf(0.7f, 0.5f, 0.5f, 3.0f)), pi(0, 0, 0, 3));
    drop_voxelizer(v);
}

static void point_in_sensor_voxel_accumulates(const Impl *impl) {
    /* The target voxel only ever receives +i in every other arm, so the
     * degenerate arm must add as well instead of cancelling itself out. */
    void *v = plain(impl);
    EXPECT_VOXELS(add_point(v, SENSOR, pf(0.7f, 0.5f, 0.5f, 5.0f)), pi(0, 0, 0, 5));
    EXPECT_VOXELS(add_point(v, SENSOR, pf(0.7f, 0.5f, 0.5f, 3.0f)), pi(0, 0, 0, 8));
    drop_voxelizer(v);
}

/* ---------------------------------------------------- compress mode */

static void compress_construct_sets_bit(const Impl *impl) {
    void *v = compressed(impl);
    EXPECT_VOXELS(along_x(v, 0, 3, 0, 0, 1.0f), pi(0, 0, 0, 1u << 3));
    drop_voxelizer(v);
}

static void compress_no_recross_no_report(const Impl *impl) {
    void *v = compressed(impl);
    EXPECT_VOXELS(along_x(v, 0, 3, 0, 0, 1.0f), block(3, 0, 0, block_bit(3, 0, 0)));
    /* 1 -> 2 stays above the construct threshold: nothing to report, even
     * though the voxel itself changed. */
    EXPECT_NO_VOXELS(along_x(v, 0, 3, 0, 0, 1.0f));
    /* Probe: (3,0,0) holds 2, so one more hit does not destroy it. */
    EXPECT_VOXELS(along_x(v, 0, 7, 0, 0, 1.0f), block(7, 0, 0, block_bit(7, 0, 0)));
    drop_voxelizer(v);
}

static void compress_plain_changes_not_returned(const Impl *impl) {
    /* Below the construct threshold a voxel change is invisible in compress
     * mode, while plain mode reports it. */
    void *c = make(impl, 1.0f, 1, 1, 0, 5, true);
    EXPECT_NO_VOXELS(along_x(c, 0, 3, 0, 0, 1.0f));
    drop_voxelizer(c);
    void *p = make(impl, 1.0f, 1, 1, 0, 5, false);
    EXPECT_VOXELS(along_x(p, 0, 3, 0, 0, 1.0f), pi(3, 0, 0, 1));
    drop_voxelizer(p);
}

static void compress_destroy_clears_bit(const Impl *impl) {
    void *v = compressed(impl);
    EXPECT_VOXELS(along_x(v, 0, 3, 0, 0, 1.0f), block(3, 0, 0, block_bit(3, 0, 0)));
    /* Passes through (3,0,0): 1 -> 0, destroyed. Ends in (7,0,0): constructed. */
    EXPECT_VOXELS(along_x(v, 0, 7, 0, 0, 1.0f), block(3, 0, 0, 0),
                  block(7, 0, 0, block_bit(7, 0, 0)));
    drop_voxelizer(v);
}

static void compress_construct_threshold(const Impl *impl) {
    void *v = make(impl, 1.0f, 1, 1, 0, 3, true);
    EXPECT_NO_VOXELS(along_x(v, 0, 3, 0, 0, 1.0f));
    EXPECT_NO_VOXELS(along_x(v, 0, 3, 0, 0, 1.0f));
    EXPECT_VOXELS(along_x(v, 0, 3, 0, 0, 1.0f), block(3, 0, 0, block_bit(3, 0, 0)));
    drop_voxelizer(v);

    /* Jumping over the threshold in one call constructs as well. */
    v = make(impl, 1.0f, 1, 1, 0, 3, true);
    EXPECT_VOXELS(along_x(v, 0, 3, 0, 0, 7.0f), block(3, 0, 0, block_bit(3, 0, 0)));
    drop_voxelizer(v);
}

static void compress_destroy_threshold(const Impl *impl) {
    void *v = make(impl, 1.0f, 1, 1, 2, 5, true);
    EXPECT_VOXELS(along_x(v, 0, 3, 0, 0, 5.0f), block(3, 0, 0, block_bit(3, 0, 0)));
    /* 5 -> 4 stays above the destroy threshold; (7,0,0) -> 1 stays below
     * the construct threshold. */
    EXPECT_NO_VOXELS(along_x(v, 0, 7, 0, 0, 1.0f));
    /* 4 -> 2 reaches the destroy threshold. */
    EXPECT_VOXELS(along_x(v, 0, 7, 0, 0, 2.0f), block(3, 0, 0, 0));
    /* 2 -> 1 is already below: no second destroy. */
    EXPECT_NO_VOXELS(along_x(v, 0, 7, 0, 0, 1.0f));
    drop_voxelizer(v);
}

static void compress_bit_positions(const Impl *impl) {
    for (int32_t x = 0; x < 4; ++x) {
        for (int32_t y = 0; y < 4; ++y) {
            for (int32_t z = 0; z < 4; ++z) {
                void *v = compressed(impl);
                VoxelsUpdate update = from_above(v, x, y, z, 1.0f);
                PointI expected = pi(0, 0, z & ~1, 1u << (x + 4 * y + 16 * (z & 1)));
                if (update.len != 1 || !same_point(update.data[0], expected)) {
                    fprintf(stderr, "voxel (%d, %d, %d):\n", x, y, z);
                }
                check_update(update, &expected, 1, __FILE__, __LINE__);
                drop_voxelizer(v);
            }
        }
    }
}

static void compress_both_halves_same_page(const Impl *impl) {
    void *v = compressed(impl);
    /* Sensor in (0,0,1); one ray down into (0,0,0), one up into (0,0,2). */
    PointF points[] = {pf(0.5f, 0.5f, 0.5f, 1.0f), pf(0.5f, 0.5f, 2.5f, 1.0f)};
    EXPECT_VOXELS(add_points(v, pf(0.5f, 0.5f, 1.5f, 1.0f), points, ARRAY_LEN(points)),
                  block(0, 0, 0, block_bit(0, 0, 0)), block(0, 0, 2, block_bit(0, 0, 2)));
    drop_voxelizer(v);
}

static void compress_two_voxels_same_block(const Impl *impl) {
    void *v = compressed(impl);
    PointF points[] = {pf(0.5f, 0.5f, 0.5f, 1.0f), pf(1.5f, 1.5f, 1.5f, 1.0f)};
    /* One block, both bits, reported once. */
    EXPECT_VOXELS(add_points(v, pf(0.5f, 0.5f, 1.5f, 1.0f), points, ARRAY_LEN(points)),
                  pi(0, 0, 0, 1u | 1u << 21));
    drop_voxelizer(v);
}

static void compress_low_half_change_with_high_bits_present(const Impl *impl) {
    void *v = compressed(impl);
    EXPECT_VOXELS(from_above(v, 0, 0, 3, 1.0f), pi(0, 0, 2, 1u << 16));
    /* A low half change must report the low half, not the (non-zero) high one. */
    EXPECT_VOXELS(from_above(v, 1, 0, 0, 1.0f), pi(0, 0, 0, 1u << 1));
    drop_voxelizer(v);
}

static void compress_page_keys(const Impl *impl) {
    void *v = compressed(impl);
    /* Fine page (1,0,0) of compressed page (0,0,0). */
    EXPECT_VOXELS(along_x(v, 3, 4, 0, 0, 1.0f), pi(4, 0, 0, 1));
    /* Fine page (16,0,0): first slot of compressed page (1,0,0). */
    EXPECT_VOXELS(along_x(v, 63, 64, 0, 0, 1.0f), pi(64, 0, 0, 1));
    /* Fine page (-1,-1,-1): last slot of compressed page (-1,-1,-1). */
    EXPECT_VOXELS(from_above(v, -1, -1, -1, 1.0f), pi(-4, -4, -2, 1u << 31));
    drop_voxelizer(v);
}

static void compress_block_coords_far(const Impl *impl) {
    void *v = compressed(impl);
    EXPECT_VOXELS(from_above(v, 5, 9, 7, 1.0f), pi(4, 8, 6, 1u << 21));
    EXPECT_VOXELS(from_above(v, -5, -9, -6, 1.0f), pi(-8, -12, -6, 1u << 15));
    drop_voxelizer(v);
}

static void compress_mask_accumulates_across_calls(const Impl *impl) {
    void *v = compressed(impl);
    EXPECT_VOXELS(from_above(v, 0, 0, 0, 1.0f), pi(0, 0, 0, 1));
    /* The pack is the block's full current state, not a delta. */
    EXPECT_VOXELS(from_above(v, 1, 0, 0, 1.0f), pi(0, 0, 0, 0x3));
    EXPECT_VOXELS(from_above(v, 2, 0, 1, 1.0f), pi(0, 0, 0, 0x3 | 1u << 18));
    drop_voxelizer(v);
}

/* --------------------------------------------------------- scenario */

static void scenario_repeatable(const Impl *impl) {
    for (size_t k = 0; k < ARRAY_LEN(SCENARIO_PARAMS); ++k) {
        const ScenarioParams *p = &SCENARIO_PARAMS[k];
        Snapshot a = scenario(
            make(impl, 1.0f, 1, 1, p->destroy_threshold, p->construct_threshold, p->compress));
        for (size_t call = 0; call < SCENARIO_CALLS; ++call) {
            CHECK(a.lens[call] > 0);
        }
        Snapshot b = scenario(
            make(impl, 1.0f, 1, 1, p->destroy_threshold, p->construct_threshold, p->compress));
        check_snapshot_eq(&b, &a, p->compress ? "compress = true" : "compress = false",
                          __FILE__, __LINE__);
        snapshot_free(&a);
        snapshot_free(&b);
    }
}

/* ------------------------------------------------------ NULL inputs */

static void null_points_zero_count(const Impl *impl) {
    void *v = plain(impl);
    VoxelsUpdate update = add_points(v, SENSOR, NULL, 0);
    CHECK(update.len == 0);
    drop_update(update);
    drop_voxelizer(v);
}

static void null_points_nonzero_count(const Impl *impl) {
    void *v = plain(impl);
    VoxelsUpdate update = add_points(v, SENSOR, NULL, 3);
    CHECK(update.len == 0);
    drop_update(update);
    drop_voxelizer(v);
}

/* ------------------------------------------------- global (run once) */

static void single_and_multi_agree(const Impl *unused) {
    (void)unused;
    static const Impl single = {"single", IMPL_SINGLE, 0};
    static const uintptr_t thread_counts[] = {1, 2, 4, 8};
    for (size_t k = 0; k < ARRAY_LEN(SCENARIO_PARAMS); ++k) {
        const ScenarioParams *p = &SCENARIO_PARAMS[k];
        Snapshot expected = scenario(make(&single, 1.0f, 1, 1, p->destroy_threshold,
                                          p->construct_threshold, p->compress));
        for (size_t call = 0; call < SCENARIO_CALLS; ++call) {
            CHECK(expected.lens[call] > 0);
        }
        for (size_t t = 0; t < ARRAY_LEN(thread_counts); ++t) {
            Impl multi = {"multi", IMPL_MULTI, thread_counts[t]};
            Snapshot got = scenario(make(&multi, 1.0f, 1, 1, p->destroy_threshold,
                                         p->construct_threshold, p->compress));
            char ctx[64];
            snprintf(ctx, sizeof ctx, "compress = %s, threads = %zu",
                     p->compress ? "true" : "false", (size_t)thread_counts[t]);
            check_snapshot_eq(&got, &expected, ctx, __FILE__, __LINE__);
            snapshot_free(&got);
        }
        snapshot_free(&expected);
    }
}

static void multi_thread_count_zero(const Impl *unused) {
    (void)unused;
    /* 0 means the default thread count. */
    static const Impl multi_default = {"multi_0", IMPL_MULTI, 0};
    basic_1d(&multi_default);
}

#ifdef R3V_TEST_WGPU
static void single_and_wgpu_agree(const Impl *unused) {
    (void)unused;
    static const Impl single = {"single", IMPL_SINGLE, 0};
    static const Impl wgpu = {"wgpu", IMPL_WGPU, 0};
    for (size_t k = 0; k < ARRAY_LEN(SCENARIO_PARAMS); ++k) {
        const ScenarioParams *p = &SCENARIO_PARAMS[k];
        Snapshot expected = scenario(make(&single, 1.0f, 1, 1, p->destroy_threshold,
                                          p->construct_threshold, p->compress));
        for (size_t call = 0; call < SCENARIO_CALLS; ++call) {
            CHECK(expected.lens[call] > 0);
        }
        Snapshot got = scenario(make(&wgpu, 1.0f, 1, 1, p->destroy_threshold,
                                     p->construct_threshold, p->compress));
        check_snapshot_eq(&got, &expected, p->compress ? "compress = true" : "compress = false",
                          __FILE__, __LINE__);
        snapshot_free(&got);
        snapshot_free(&expected);
    }
}

static void wgpu_zero_buckets_null(const Impl *unused) {
    (void)unused;
    CHECK(create_wgpu_voxelizer(1.0f, 1, 1, 0, 1, false, 0) == NULL);
}

static void wgpu_memory_size_grows(const Impl *unused) {
    (void)unused;
    CHECK(wgpu_memory_size(1) > 0);
    CHECK(wgpu_memory_size(2) > wgpu_memory_size(1));
    CHECK(wgpu_memory_size(WGPU_TEST_BUCKETS) > wgpu_memory_size(2));
}
#else
static void wgpu_null_without_feature(const Impl *unused) {
    (void)unused;
    CHECK(create_wgpu_voxelizer(1.0f, 1, 1, 0, 1, false, WGPU_TEST_BUCKETS) == NULL);
    CHECK(wgpu_memory_size(WGPU_TEST_BUCKETS) == 0);
}
#endif

static void drop_voxelizer_null(const Impl *unused) {
    (void)unused;
    drop_voxelizer(NULL);
}

static void drop_update_empty(const Impl *unused) {
    (void)unused;
    VoxelsUpdate empty = {NULL, 0};
    drop_update(empty);
}

static void null_voxelizer(const Impl *unused) {
    (void)unused;
    PointF point = pf(3.5f, 0.5f, 0.5f, 1.0f);
    VoxelsUpdate update = add_points(NULL, SENSOR, &point, 1);
    CHECK(update.len == 0);
    drop_update(update);
}

static void compress_helpers_self_check(const Impl *unused) {
    (void)unused;
    CHECK(block_bit(3, 3, 1) == 1u << 31);
    CHECK(block_bit(3, 3, 3) == 1u << 31);
    CHECK(same_point(block(3, 0, 0, block_bit(3, 0, 0)), pi(0, 0, 0, 1u << 3)));
    CHECK(same_point(block(5, 9, 7, 0), pi(4, 8, 6, 0)));
    CHECK(same_point(block(-1, -1, -1, block_bit(-1, -1, -1)), pi(-4, -4, -2, 1u << 31)));
    CHECK(same_point(block(-5, -9, -6, 0), pi(-8, -12, -6, 0)));
}

/* ------------------------------------------------------------ table */

#define PER_IMPL(fn) {#fn, fn, true}
#define GLOBAL(fn) {#fn, fn, false}

const TestCase TEST_CASES[] = {
    PER_IMPL(create_returns_non_null),
    PER_IMPL(create_drop_cycles),
    PER_IMPL(basic_1d),
    PER_IMPL(empty_points_no_change),
    PER_IMPL(returns_absolute_value_not_delta),
    PER_IMPL(sensor_intensity_ignored),
    PER_IMPL(intensity_truncated_toward_zero),
    PER_IMPL(zero_intensity_reports_nothing),
    PER_IMPL(resolution_scales_coords_not_intensity),
    PER_IMPL(resolution_scales_sensor_too),
    PER_IMPL(large_coords),
    PER_IMPL(negative_coords_use_floor),
    PER_IMPL(hit_multiplier_scales_hits),
    PER_IMPL(target_multiplier_scales_target),
    PER_IMPL(both_multipliers),
    PER_IMPL(hit_saturates_at_zero),
    PER_IMPL(hit_on_zero_voxel_not_reported),
    PER_IMPL(thresholds_ignored_without_compress),
    PER_IMPL(accumulates_across_calls),
    PER_IMPL(unchanged_voxel_not_reported),
    PER_IMPL(second_call_reports_only_its_changes),
    PER_IMPL(multiple_points_one_call),
    PER_IMPL(hits_skip_missing_pages),
    PER_IMPL(negative_page),
    PER_IMPL(axis_x_negative),
    PER_IMPL(axis_y_positive),
    PER_IMPL(axis_y_negative),
    PER_IMPL(axis_z_positive),
    PER_IMPL(axis_z_negative),
    PER_IMPL(plane_xy_forward),
    PER_IMPL(plane_xy_reverse),
    PER_IMPL(plane_xz),
    PER_IMPL(plane_yz),
    PER_IMPL(full_3d_forward),
    PER_IMPL(full_3d_reverse),
    PER_IMPL(sensor_on_grid_line_2d),
    PER_IMPL(ray_across_page_boundary),
    PER_IMPL(point_in_sensor_voxel_fresh),
    PER_IMPL(point_in_sensor_voxel_accumulates),
    PER_IMPL(compress_construct_sets_bit),
    PER_IMPL(compress_no_recross_no_report),
    PER_IMPL(compress_plain_changes_not_returned),
    PER_IMPL(compress_destroy_clears_bit),
    PER_IMPL(compress_construct_threshold),
    PER_IMPL(compress_destroy_threshold),
    PER_IMPL(compress_bit_positions),
    PER_IMPL(compress_both_halves_same_page),
    PER_IMPL(compress_two_voxels_same_block),
    PER_IMPL(compress_low_half_change_with_high_bits_present),
    PER_IMPL(compress_page_keys),
    PER_IMPL(compress_block_coords_far),
    PER_IMPL(compress_mask_accumulates_across_calls),
    PER_IMPL(scenario_repeatable),
    PER_IMPL(null_points_zero_count),
    PER_IMPL(null_points_nonzero_count),
    GLOBAL(single_and_multi_agree),
    GLOBAL(multi_thread_count_zero),
#ifdef R3V_TEST_WGPU
    GLOBAL(single_and_wgpu_agree),
    GLOBAL(wgpu_zero_buckets_null),
    GLOBAL(wgpu_memory_size_grows),
#else
    GLOBAL(wgpu_null_without_feature),
#endif
    GLOBAL(drop_voxelizer_null),
    GLOBAL(drop_update_empty),
    GLOBAL(null_voxelizer),
    GLOBAL(compress_helpers_self_check),
};
const size_t TEST_CASE_COUNT = ARRAY_LEN(TEST_CASES);
