"""
Test cases for the Python bindings: checks of the Python facing API
(constructors, accessors, argument conversion) followed by a port of
the C binding tests in tests/c/voxelizer_tests.c.

Per implementation cases run against every entry of IMPLS (harness.py).
"""

import gc
import sys

import r3v_voxelizer as rv

from harness import (
    CLASS_OF_FEATURE, FEATURES, SCENARIO_PARAMS, SEED, SINGLE, WGPU, WGPU_TEST_BUCKETS, add_point, along_x, block,
    block_bit, check, check_ray, check_update, compressed, coords, expect_no_voxels, expect_voxels, from_above, main,
    make, multi_threaded, pf, pi, plain, raises, scenario, seed, sensor,
)

U32_MAX = 2**32 - 1

# ------------------------------------------------------------ Python API


def enabled_classes():
    """Names of the classes the module must have with the enabled features."""
    return ["Multipliers", "PointF", "PointI"] + [cls for feature, cls in CLASS_OF_FEATURE.items()
                                                  if feature in FEATURES]


def module_exports_match_features():
    for name in enabled_classes():
        check(isinstance(getattr(rv, name, None), type), f"r3v_voxelizer.{name} is missing")
    for feature, name in CLASS_OF_FEATURE.items():
        if feature not in FEATURES:
            check(not hasattr(rv, name), f"r3v_voxelizer.{name} exists without {feature}")


def classes_belong_to_module():
    for cls in (getattr(rv, name) for name in enabled_classes()):
        check(cls.__module__ == "r3v_voxelizer",
              f"{cls.__name__}.__module__ is {cls.__module__!r}, expected 'r3v_voxelizer'")


def point_f_positional():
    check(coords(rv.PointF(1.5, -2.25, 3.0, 4.0)) == (1.5, -2.25, 3.0, 4.0))


def point_f_keywords():
    check(coords(rv.PointF(x=1.5, y=-2.25, z=3.0, i=4.0)) == (1.5, -2.25, 3.0, 4.0))


def point_f_from_ints():
    point = rv.PointF(1, -2, 3, 4)
    check(coords(point) == (1.0, -2.0, 3.0, 4.0))
    check(all(type(value) is float for value in coords(point)))


def point_f_rejects_non_numbers():
    # The constructor has to work in the first place.
    pf(0.0, 0.0, 0.0, 0.0)
    with raises(TypeError):
        rv.PointF("1", 0.0, 0.0, 0.0)
    with raises(TypeError):
        rv.PointF(0.0, None, 0.0, 0.0)


def point_f_setters():
    point = pf(0.0, 0.0, 0.0, 0.0)
    point.set_x(1.5)
    point.set_y(-2.25)
    point.set_z(3)
    point.set_i(4.0)
    check(coords(point) == (1.5, -2.25, 3.0, 4.0), f"point became {coords(point)}")
    with raises(TypeError):
        point.set_x("1")


def point_i_positional():
    point = rv.PointI(1, -2, 3, 4)
    check(coords(point) == (1, -2, 3, 4))
    check(all(type(value) is int for value in coords(point)))


def point_i_keywords():
    check(coords(rv.PointI(x=1, y=-2, z=3, i=4)) == (1, -2, 3, 4))


def point_i_rejects_out_of_range():
    rv.PointI(-2**31, 2**31 - 1, 0, U32_MAX)
    with raises(OverflowError):
        rv.PointI(2**31, 0, 0, 0)
    with raises(OverflowError):
        rv.PointI(0, -2**31 - 1, 0, 0)
    with raises(OverflowError):
        rv.PointI(0, 0, 0, -1)
    with raises(OverflowError):
        rv.PointI(0, 0, 0, U32_MAX + 1)
    with raises(TypeError):
        rv.PointI(0, 0, 1.5, 0)


def point_i_setters():
    point = rv.PointI(0, 0, 0, 0)
    point.set_x(1)
    point.set_y(-2)
    point.set_z(3)
    point.set_i(U32_MAX)
    check(coords(point) == (1, -2, 3, U32_MAX), f"point became {coords(point)}")
    with raises(OverflowError):
        point.set_i(-1)


def multiplier_values(multipliers):
    return (multipliers.intensity(), multipliers.hit(), multipliers.target())


def multipliers_positional():
    multipliers = rv.Multipliers(2.5, 2, 3)
    check(multiplier_values(multipliers) == (2.5, 2, 3))
    check(type(multipliers.intensity()) is float)
    check(type(multipliers.hit()) is int and type(multipliers.target()) is int)


def multipliers_keywords():
    # Out of order, so a swapped name shows up as swapped values.
    check(multiplier_values(rv.Multipliers(target=3, hit=2, intensity=2.5)) == (2.5, 2, 3))


def multipliers_rejects_out_of_range():
    rv.Multipliers(-1.5, 0, U32_MAX)
    with raises(OverflowError):
        rv.Multipliers(1.0, -1, 0)
    with raises(OverflowError):
        rv.Multipliers(1.0, U32_MAX + 1, 0)
    with raises(OverflowError):
        rv.Multipliers(1.0, 0, -1)
    with raises(OverflowError):
        rv.Multipliers(1.0, 0, U32_MAX + 1)
    with raises(TypeError):
        rv.Multipliers(1.0, 1.5, 0)
    with raises(TypeError):
        rv.Multipliers("1", 1, 1)


def multipliers_setters():
    multipliers = rv.Multipliers(1.0, 1, 1)
    multipliers.set_intensity(2.5)
    multipliers.set_hit(U32_MAX)
    multipliers.set_target(0)
    values = multiplier_values(multipliers)
    check(values == (2.5, U32_MAX, 0), f"multipliers became {values}")
    with raises(OverflowError):
        multipliers.set_hit(-1)
    with raises(OverflowError):
        multipliers.set_target(U32_MAX + 1)


def constructors_reject_out_of_range(impl):
    good = (1.0, 1.0, 1, 1, 0, 1, U32_MAX, False)
    extra = () if impl.extra is None else (impl.extra[1],)
    cls = getattr(rv, impl.cls)
    for k in range(2, 7):
        for bad in (-1, U32_MAX + 1):
            with raises(OverflowError):
                cls(*good[:k], bad, *good[k + 1:], *extra)
    if impl.extra is not None:
        with raises(OverflowError):
            cls(*good, -1)


def constructors_accept_u32_max(impl):
    make(impl, 1.0, U32_MAX, U32_MAX, U32_MAX, U32_MAX, True)


def constructors_accept_keywords(impl):
    # Out of order, so a swapped name shows up as swapped multipliers.
    kwargs = dict(compress=False, max_value=10, construct_threshold=1, destroy_threshold=0,
                  target_multiplier=3, hit_multiplier=2, intensity_multiplier=1.5,
                  resolution=1.0)
    if impl.extra is not None:
        kwargs[impl.extra[0]] = impl.extra[1]
    v = getattr(rv, impl.cls)(**kwargs)
    # Like both_multipliers: seeds (0,0,0) with (4 * 1.5) * 3 = 18, capped at 10.
    seed(v, 1.0, 0, 0, 0, 4.0)
    # 2 * 1.5 = 3
    expect_voxels(add_point(v, sensor(), pf(3.5, 0.5, 0.5, 2.0)),
                  pi(0, 0, 0, 10 - 3 * 2), pi(3, 0, 0, 3 * 3))


def add_points_result_types(impl):
    v = plain(impl)
    update = add_point(v, sensor(), pf(3.5, 0.5, 0.5, 1.0))
    check(type(update) is list, f"add_points returned {type(update).__name__}, not list")
    check(len(update) == 1, f"add_points returned {len(update)} points, expected 1")
    point = update[0]
    check(type(point) is rv.PointI, f"add_points returned a {type(point).__name__}")
    check(all(type(value) is int for value in coords(point)))
    # The result is owned by Python and outlives the voxelizer.
    del v
    gc.collect()
    check(coords(point) == (3, 0, 0, 1))


def add_points_sees_setter_values(impl):
    v = plain(impl)
    point = pf(0.0, 0.0, 0.0, 0.0)
    point.set_x(3.5)
    point.set_y(0.5)
    point.set_z(0.5)
    point.set_i(2.0)
    expect_voxels(add_point(v, sensor(), point), pi(3, 0, 0, 2))


def add_points_accepts_tuple(impl):
    v = plain(impl)
    points = (pf(3.5, 0.5, 0.5, 1.0), pf(0.5, 3.5, 0.5, 2.0))
    expect_voxels(v.add_points(sensor(), points), pi(3, 0, 0, 1), pi(0, 3, 0, 2))


def add_points_rejects_wrong_types(impl):
    v = plain(impl)
    # Built outside of `raises`: a TypeError from PointF itself must not count.
    sensor_pos = sensor()
    point = pf(3.5, 0.5, 0.5, 1.0)
    with raises(TypeError):
        v.add_points(None, [point])
    with raises(TypeError):
        v.add_points(sensor_pos, None)
    with raises(TypeError):
        v.add_points(sensor_pos, [point, None])
    with raises(TypeError):
        v.add_points(sensor_pos, point)
    with raises(TypeError):
        v.add_points(sensor_pos)
    # Probe: none of the failed calls stored anything.
    expect_voxels(add_point(v, sensor_pos, point), pi(3, 0, 0, 1))


def add_points_leaves_inputs_unchanged(impl):
    v = make(impl, 2.0, 1, 1, 0, 1, False)
    sensor_pos = pf(2.0, 0.0, 0.0, 1.0)
    points = [pf(7.0, 0.0, 0.0, 1.0)]
    expect_voxels(v.add_points(sensor_pos, points), pi(3, 0, 0, 1))
    check(coords(sensor_pos) == (2.0, 0.0, 0.0, 1.0), f"sensor became {coords(sensor_pos)}")
    check(len(points) == 1)
    check(coords(points[0]) == (7.0, 0.0, 0.0, 1.0), f"point became {coords(points[0])}")


def instances_are_independent(impl):
    a = plain(impl)
    b = plain(impl)
    point = pf(3.5, 0.5, 0.5, 1.0)
    expect_voxels(add_point(a, sensor(), point), pi(3, 0, 0, 1))
    expect_voxels(add_point(a, sensor(), point), pi(3, 0, 0, 2))
    expect_voxels(add_point(b, sensor(), point), pi(3, 0, 0, 1))


# ------------------------------------------------------------- lifecycle


def create(impl):
    cls = getattr(rv, impl.cls)
    for v in (plain(impl), compressed(impl), make(impl, 0.05, 2, 3, 4, 5, True)):
        check(type(v) is cls, f"created a {type(v).__name__}, expected {cls.__name__}")


def create_drop_cycles(impl):
    for k in range(50):
        v = make(impl, 1.0, 1, 1, 0, 1, k % 2 == 0)
        update = add_point(v, sensor(), pf(3.5, 0.5, 0.5, 1.0))
        if k % 2 == 0:
            expect_voxels(update, pi(0, 0, 0, 1 << 3))
        else:
            expect_voxels(update, pi(3, 0, 0, 1))
        del v


# ------------------------------------------------------ add_points basics


def basic_1d(impl):
    v = plain(impl)
    expect_voxels(add_point(v, pf(0.0, 0.0, 0.0, 1.0), pf(5.0, 0.0, 0.0, 1.0)), pi(5, 0, 0, 1))


def empty_points_no_change(impl):
    v = plain(impl)
    expect_no_voxels(v.add_points(sensor(), []))
    # Nothing was stored, so the target still starts from zero.
    expect_voxels(add_point(v, sensor(), pf(3.5, 0.5, 0.5, 1.0)), pi(3, 0, 0, 1))


def returns_absolute_value_not_delta(impl):
    v = plain(impl)
    for x in range(5):
        seed(v, 1.0, x, 0, 0, SEED)
    seed(v, 1.0, 5, 0, 0, 3)
    expect_voxels(add_point(v, sensor(), pf(5.5, 0.5, 0.5, 2.0)),
                  pi(0, 0, 0, SEED - 2), pi(1, 0, 0, SEED - 2), pi(2, 0, 0, SEED - 2),
                  pi(3, 0, 0, SEED - 2), pi(4, 0, 0, SEED - 2), pi(5, 0, 0, 5))


def sensor_intensity_ignored(impl):
    v = plain(impl)
    seed(v, 1.0, 0, 0, 0, SEED)
    # Hits use the point's intensity, not the sensor's.
    expect_voxels(add_point(v, pf(0.5, 0.5, 0.5, 99.0), pf(3.5, 0.5, 0.5, 1.0)),
                  pi(0, 0, 0, SEED - 1), pi(3, 0, 0, 1))


def intensity_truncated_toward_zero(impl):
    v = plain(impl)
    expect_voxels(add_point(v, sensor(), pf(3.5, 0.5, 0.5, 2.9)), pi(3, 0, 0, 2))


def zero_intensity_reports_nothing(impl):
    v = plain(impl)
    seed(v, 1.0, 0, 0, 0, SEED)
    expect_no_voxels(add_point(v, sensor(), pf(3.5, 0.5, 0.5, 0.0)))
    # Probe: both voxels kept their values.
    expect_voxels(add_point(v, sensor(), pf(3.5, 0.5, 0.5, 1.0)),
                  pi(0, 0, 0, SEED - 1), pi(3, 0, 0, 1))


def resolution_scales_coords_not_intensity(impl):
    v = make(impl, 0.25, 1, 1, 0, 1, False)
    for x in range(4):
        seed(v, 0.25, x, 0, 0, SEED)
    expect_voxels(add_point(v, pf(0.1, 0.1, 0.1, 1.0), pf(1.0, 0.1, 0.1, 7.0)),
                  pi(0, 0, 0, SEED - 7), pi(1, 0, 0, SEED - 7), pi(2, 0, 0, SEED - 7),
                  pi(3, 0, 0, SEED - 7), pi(4, 0, 0, 7))

    v = make(impl, 2.0, 1, 1, 0, 1, False)
    expect_voxels(add_point(v, pf(0.0, 0.0, 0.0, 1.0), pf(5.0, 0.0, 0.0, 7.0)), pi(2, 0, 0, 7))


def resolution_scales_sensor_too(impl):
    v = make(impl, 2.0, 1, 1, 0, 1, False)
    seed(v, 2.0, 1, 0, 0, SEED)
    seed(v, 2.0, 2, 0, 0, SEED)
    # sensor 2.0 -> 1.0 -> voxel 1, point 7.0 -> 3.5 -> voxel 3
    expect_voxels(add_point(v, pf(2.0, 0.0, 0.0, 1.0), pf(7.0, 0.0, 0.0, 1.0)),
                  pi(1, 0, 0, SEED - 1), pi(2, 0, 0, SEED - 1), pi(3, 0, 0, 1))


def large_coords(impl):
    v = plain(impl)
    sensor_pos = pf(1000.5, -2000.5, 29999.5, 1.0)
    point = pf(1000.5, -2000.5, 30000.5, 4.0)
    expect_voxels(add_point(v, sensor_pos, point), pi(1000, -2001, 30000, 4))
    # Probe: the value was stored where it was reported.
    expect_voxels(add_point(v, sensor_pos, point), pi(1000, -2001, 30000, 8))


def negative_coords_use_floor(impl):
    for x, voxel in [(-0.5, -1), (-1.0, -1), (-1.5, -2)]:
        v = plain(impl)
        expect_voxels(add_point(v, pf(-3.5, 0.5, 0.5, 1.0), pf(x, 0.5, 0.5, 1.0)),
                      pi(voxel, 0, 0, 1))


# ------------------------------------------- multipliers / saturation


def hit_multiplier_scales_hits(impl):
    v = make(impl, 1.0, 3, 1, 0, 1, False)
    seed(v, 1.0, 0, 0, 0, SEED)
    expect_voxels(add_point(v, sensor(), pf(3.5, 0.5, 0.5, 2.0)),
                  pi(0, 0, 0, SEED - 2 * 3), pi(3, 0, 0, 2))


def target_multiplier_scales_target(impl):
    v = make(impl, 1.0, 1, 5, 0, 1, False)
    # Seeding goes through the target path too: 2 * 5 = SEED.
    seed(v, 1.0, 0, 0, 0, SEED // 5)
    expect_voxels(add_point(v, sensor(), pf(3.5, 0.5, 0.5, 2.0)),
                  pi(0, 0, 0, SEED - 2), pi(3, 0, 0, 2 * 5))


def both_multipliers(impl):
    v = make(impl, 1.0, 2, 3, 0, 1, False)
    # Seeds (0,0,0) with 4 * 3 = 12.
    seed(v, 1.0, 0, 0, 0, 4.0)
    point = pf(3.5, 0.5, 0.5, 2.0)
    expect_voxels(add_point(v, sensor(), point), pi(0, 0, 0, 12 - 2 * 2), pi(3, 0, 0, 2 * 3))
    # Probe: values were stored.
    expect_voxels(add_point(v, sensor(), point),
                  pi(0, 0, 0, 12 - 2 * 2 * 2), pi(3, 0, 0, 2 * 3 * 2))


def hit_saturates_at_zero(impl):
    v = plain(impl)
    seed(v, 1.0, 0, 0, 0, 1)
    # (1,0,0) and (2,0,0) exist (same page) and hold 0. 1 -> 0 is a change
    # and gets reported as 0; 0 -> 0 is not.
    expect_voxels(add_point(v, sensor(), pf(3.5, 0.5, 0.5, 5.0)), pi(0, 0, 0, 0), pi(3, 0, 0, 5))
    # Probe: nothing wrapped around below zero.
    expect_voxels(add_point(v, sensor(), pf(3.5, 0.5, 0.5, 1.0)), pi(3, 0, 0, 6))

    # The multiplier saturates too.
    v = make(impl, 1.0, 100, 1, 0, 1, False)
    seed(v, 1.0, 0, 0, 0, SEED)
    expect_voxels(add_point(v, sensor(), pf(3.5, 0.5, 0.5, 1.0)), pi(0, 0, 0, 0), pi(3, 0, 0, 1))


def hit_on_zero_voxel_not_reported(impl):
    v = plain(impl)
    # Creates page (0,0,0) without touching the ray below.
    seed(v, 1.0, 1, 1, 1, SEED)
    expect_voxels(add_point(v, sensor(), pf(3.5, 0.5, 0.5, 1.0)), pi(3, 0, 0, 1))


def thresholds_ignored_without_compress(impl):
    v = make(impl, 1.0, 1, 1, 100, 200, False)
    for x in range(5):
        seed(v, 1.0, x, 0, 0, SEED)
    seed(v, 1.0, 5, 0, 0, 3)
    expect_voxels(add_point(v, sensor(), pf(5.5, 0.5, 0.5, 2.0)),
                  pi(0, 0, 0, SEED - 2), pi(1, 0, 0, SEED - 2), pi(2, 0, 0, SEED - 2),
                  pi(3, 0, 0, SEED - 2), pi(4, 0, 0, SEED - 2), pi(5, 0, 0, 5))


# ------------------------------------- accumulation / change detection


def accumulates_across_calls(impl):
    v = plain(impl)
    for x in range(3):
        seed(v, 1.0, x, 0, 0, SEED)
    point = pf(3.5, 0.5, 0.5, 1.0)
    expect_voxels(add_point(v, sensor(), point),
                  pi(0, 0, 0, SEED - 1), pi(1, 0, 0, SEED - 1), pi(2, 0, 0, SEED - 1),
                  pi(3, 0, 0, 1))
    expect_voxels(add_point(v, sensor(), point),
                  pi(0, 0, 0, SEED - 2), pi(1, 0, 0, SEED - 2), pi(2, 0, 0, SEED - 2),
                  pi(3, 0, 0, 2))


def unchanged_voxel_not_reported(impl):
    v = plain(impl)
    seed(v, 1.0, 0, 0, 0, 5)
    seed(v, 1.0, 1, 0, 0, 1)
    # First ray hits (1,0,0): 1 -> 0, second ray ends there: 0 -> 1.
    # (The other order gives 1 -> 2 -> 1: same outcome.)
    points = [pf(2.5, 0.5, 0.5, 1.0), pf(1.5, 0.5, 0.5, 1.0)]
    expect_voxels(v.add_points(sensor(), points), pi(0, 0, 0, 3), pi(2, 0, 0, 1))
    # Probe: (1,0,0) holds 1, so one more hit takes it to 0.
    expect_voxels(add_point(v, pf(1.5, 0.5, 0.5, 1.0), pf(1.5, 1.5, 0.5, 1.0)),
                  pi(1, 0, 0, 0), pi(1, 1, 0, 1))


def second_call_reports_only_its_changes(impl):
    v = plain(impl)
    for x in range(3):
        seed(v, 1.0, x, 0, 0, SEED)
    # Straight up along z: hits (0,0,0), (0,0,1), (0,0,2).
    expect_voxels(add_point(v, sensor(), pf(0.5, 0.5, 3.5, 1.0)),
                  pi(0, 0, 0, SEED - 1), pi(0, 0, 3, 1))
    expect_voxels(add_point(v, sensor(), pf(3.5, 0.5, 0.5, 1.0)),
                  pi(0, 0, 0, SEED - 2), pi(1, 0, 0, SEED - 1), pi(2, 0, 0, SEED - 1),
                  pi(3, 0, 0, 1))


def multiple_points_one_call(impl):
    v = plain(impl)
    for x, y in [(0, 0), (1, 0), (2, 0), (0, 1), (0, 2)]:
        seed(v, 1.0, x, y, 0, SEED)
    points = [pf(3.5, 0.5, 0.5, 1.0), pf(0.5, 3.5, 0.5, 2.0)]
    expect_voxels(v.add_points(sensor(), points),
                  pi(0, 0, 0, SEED - 1 - 2), pi(1, 0, 0, SEED - 1), pi(2, 0, 0, SEED - 1),
                  pi(3, 0, 0, 1), pi(0, 1, 0, SEED - 2), pi(0, 2, 0, SEED - 2), pi(0, 3, 0, 2))


# ------------------------------------------------------------- paging


def hits_skip_missing_pages(impl):
    v = plain(impl)
    seed(v, 1.0, 0, 0, 0, SEED)
    point = pf(20.5, 0.5, 0.5, 1.0)
    # The ray crosses fine pages 0..=5 along x but only page 0 exists.
    expect_voxels(add_point(v, sensor(), point), pi(0, 0, 0, SEED - 1), pi(20, 0, 0, 1))
    expect_voxels(add_point(v, sensor(), point), pi(0, 0, 0, SEED - 2), pi(20, 0, 0, 2))
    # Once a page exists on the way its voxels do get hit.
    seed(v, 1.0, 8, 0, 0, SEED)
    expect_voxels(add_point(v, sensor(), point),
                  pi(0, 0, 0, SEED - 3), pi(8, 0, 0, SEED - 1), pi(20, 0, 0, 3))


def negative_page(impl):
    v = plain(impl)
    sensor_pos = pf(-0.5, -0.5, -0.5, 1.0)
    point = pf(-3.5, -0.5, -0.5, 3.0)
    expect_voxels(add_point(v, sensor_pos, point), pi(-4, -1, -1, 3))
    # Probe: the page now exists, the voxels on the way hold 0.
    expect_voxels(add_point(v, sensor_pos, point), pi(-4, -1, -1, 6))


# ----------------------------------------------------- ray march arms


def axis_x_negative(impl):
    check_ray(impl, pf(5.5, 0.5, 0.5, 1.0), pf(0.5, 0.5, 0.5, 1.0), (0, 0, 0),
              (5, 0, 0), (4, 0, 0), (3, 0, 0), (2, 0, 0), (1, 0, 0))


def axis_y_positive(impl):
    check_ray(impl, pf(0.5, 0.5, 0.5, 1.0), pf(0.5, 4.5, 0.5, 1.0), (0, 4, 0),
              (0, 0, 0), (0, 1, 0), (0, 2, 0), (0, 3, 0))


def axis_y_negative(impl):
    check_ray(impl, pf(0.5, 4.5, 0.5, 1.0), pf(0.5, 0.5, 0.5, 1.0), (0, 0, 0),
              (0, 4, 0), (0, 3, 0), (0, 2, 0), (0, 1, 0))


def axis_z_positive(impl):
    check_ray(impl, pf(0.5, 0.5, 0.5, 1.0), pf(0.5, 0.5, 4.5, 1.0), (0, 0, 4),
              (0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3))


def axis_z_negative(impl):
    check_ray(impl, pf(0.5, 0.5, 4.5, 1.0), pf(0.5, 0.5, 0.5, 1.0), (0, 0, 0),
              (0, 0, 4), (0, 0, 3), (0, 0, 2), (0, 0, 1))


def plane_xy_forward(impl):
    check_ray(impl, pf(0.5, 0.5, 0.5, 1.0), pf(3.5, 1.7, 0.5, 1.0), (3, 1, 0),
              (0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0))


def plane_xy_reverse(impl):
    check_ray(impl, pf(3.5, 1.7, 0.5, 1.0), pf(0.5, 0.5, 0.5, 1.0), (0, 0, 0),
              (3, 1, 0), (2, 1, 0), (1, 1, 0), (1, 0, 0))


def plane_xz(impl):
    check_ray(impl, pf(0.5, 0.5, 0.5, 1.0), pf(3.5, 0.5, 1.7, 1.0), (3, 0, 1),
              (0, 0, 0), (1, 0, 0), (1, 0, 1), (2, 0, 1))


def plane_yz(impl):
    check_ray(impl, pf(0.5, 0.5, 0.5, 1.0), pf(0.5, 3.5, 1.7, 1.0), (0, 3, 1),
              (0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 2, 1))


def full_3d_forward(impl):
    check_ray(impl, pf(0.5, 0.5, 0.5, 1.0), pf(2.5, 1.6, 3.5, 1.0), (2, 1, 3),
              (0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (1, 1, 2), (2, 1, 2))


def full_3d_reverse(impl):
    check_ray(impl, pf(2.5, 1.6, 3.5, 1.0), pf(0.5, 0.5, 0.5, 1.0), (0, 0, 0),
              (2, 1, 3), (2, 1, 2), (1, 1, 2), (1, 1, 1), (1, 0, 1), (0, 0, 1))


def sensor_on_grid_line_2d(impl):
    # Line y = 0.6x from the origin: crosses x=1 at t=0.4 (y=0.6), y=1 at
    # t=0.667 (x=1.67), x=2 at t=0.8.
    check_ray(impl, pf(0.0, 0.0, 0.0, 1.0), pf(2.5, 1.5, 0.5, 1.0), (2, 1, 0),
              (0, 0, 0), (1, 0, 0), (1, 1, 0))


def ray_across_page_boundary(impl):
    # Hits span fine pages (0,0,0) and (1,0,0) along x.
    check_ray(impl, pf(2.5, 0.5, 0.5, 1.0), pf(6.5, 1.7, 0.5, 1.0), (6, 1, 0),
              (2, 0, 0), (3, 0, 0), (4, 0, 0), (4, 1, 0), (5, 1, 0))


# -------------------------------------- point inside the sensor voxel


def point_in_sensor_voxel_fresh(impl):
    v = plain(impl)
    expect_voxels(add_point(v, sensor(), pf(0.7, 0.5, 0.5, 3.0)), pi(0, 0, 0, 3))


def point_in_sensor_voxel_accumulates(impl):
    # The target voxel only ever receives +i in every other arm, so the
    # degenerate arm must add as well instead of cancelling itself out.
    v = plain(impl)
    expect_voxels(add_point(v, sensor(), pf(0.7, 0.5, 0.5, 5.0)), pi(0, 0, 0, 5))
    expect_voxels(add_point(v, sensor(), pf(0.7, 0.5, 0.5, 3.0)), pi(0, 0, 0, 8))


# ------------------------------------------------------- compress mode


def compress_construct_sets_bit(impl):
    v = compressed(impl)
    expect_voxels(along_x(v, 0, 3, 0, 0, 1.0), pi(0, 0, 0, 1 << 3))


def compress_no_recross_no_report(impl):
    v = compressed(impl)
    expect_voxels(along_x(v, 0, 3, 0, 0, 1.0), block(3, 0, 0, block_bit(3, 0, 0)))
    # 1 -> 2 stays above the construct threshold: nothing to report, even
    # though the voxel itself changed.
    expect_no_voxels(along_x(v, 0, 3, 0, 0, 1.0))
    # Probe: (3,0,0) holds 2, so one more hit does not destroy it.
    expect_voxels(along_x(v, 0, 7, 0, 0, 1.0), block(7, 0, 0, block_bit(7, 0, 0)))


def compress_plain_changes_not_returned(impl):
    # Below the construct threshold a voxel change is invisible in compress
    # mode, while plain mode reports it.
    c = make(impl, 1.0, 1, 1, 0, 5, True)
    expect_no_voxels(along_x(c, 0, 3, 0, 0, 1.0))
    p = make(impl, 1.0, 1, 1, 0, 5, False)
    expect_voxels(along_x(p, 0, 3, 0, 0, 1.0), pi(3, 0, 0, 1))


def compress_destroy_clears_bit(impl):
    v = compressed(impl)
    expect_voxels(along_x(v, 0, 3, 0, 0, 1.0), block(3, 0, 0, block_bit(3, 0, 0)))
    # Passes through (3,0,0): 1 -> 0, destroyed. Ends in (7,0,0): constructed.
    expect_voxels(along_x(v, 0, 7, 0, 0, 1.0),
                  block(3, 0, 0, 0), block(7, 0, 0, block_bit(7, 0, 0)))


def compress_construct_threshold(impl):
    v = make(impl, 1.0, 1, 1, 0, 3, True)
    expect_no_voxels(along_x(v, 0, 3, 0, 0, 1.0))
    expect_no_voxels(along_x(v, 0, 3, 0, 0, 1.0))
    expect_voxels(along_x(v, 0, 3, 0, 0, 1.0), block(3, 0, 0, block_bit(3, 0, 0)))

    # Jumping over the threshold in one call constructs as well.
    v = make(impl, 1.0, 1, 1, 0, 3, True)
    expect_voxels(along_x(v, 0, 3, 0, 0, 7.0), block(3, 0, 0, block_bit(3, 0, 0)))


def compress_destroy_threshold(impl):
    v = make(impl, 1.0, 1, 1, 2, 5, True)
    expect_voxels(along_x(v, 0, 3, 0, 0, 5.0), block(3, 0, 0, block_bit(3, 0, 0)))
    # 5 -> 4 stays above the destroy threshold; (7,0,0) -> 1 stays below
    # the construct threshold.
    expect_no_voxels(along_x(v, 0, 7, 0, 0, 1.0))
    # 4 -> 2 reaches the destroy threshold.
    expect_voxels(along_x(v, 0, 7, 0, 0, 2.0), block(3, 0, 0, 0))
    # 2 -> 1 is already below: no second destroy.
    expect_no_voxels(along_x(v, 0, 7, 0, 0, 1.0))


def compress_bit_positions(impl):
    for x in range(4):
        for y in range(4):
            for z in range(4):
                v = compressed(impl)
                expected = pi(0, 0, z & ~1, 1 << (x + 4 * y + 16 * (z & 1)))
                check_update(from_above(v, x, y, z, 1.0), [expected], f"voxel ({x}, {y}, {z})")


def compress_both_halves_same_page(impl):
    v = compressed(impl)
    # Sensor in (0,0,1); one ray down into (0,0,0), one up into (0,0,2).
    points = [pf(0.5, 0.5, 0.5, 1.0), pf(0.5, 0.5, 2.5, 1.0)]
    expect_voxels(v.add_points(pf(0.5, 0.5, 1.5, 1.0), points),
                  block(0, 0, 0, block_bit(0, 0, 0)), block(0, 0, 2, block_bit(0, 0, 2)))


def compress_two_voxels_same_block(impl):
    v = compressed(impl)
    points = [pf(0.5, 0.5, 0.5, 1.0), pf(1.5, 1.5, 1.5, 1.0)]
    # One block, both bits, reported once.
    expect_voxels(v.add_points(pf(0.5, 0.5, 1.5, 1.0), points), pi(0, 0, 0, 1 | 1 << 21))


def compress_low_half_change_with_high_bits_present(impl):
    v = compressed(impl)
    expect_voxels(from_above(v, 0, 0, 3, 1.0), pi(0, 0, 2, 1 << 16))
    # A low half change must report the low half, not the (non-zero) high one.
    expect_voxels(from_above(v, 1, 0, 0, 1.0), pi(0, 0, 0, 1 << 1))


def compress_page_keys(impl):
    v = compressed(impl)
    # Fine page (1,0,0) of compressed page (0,0,0).
    expect_voxels(along_x(v, 3, 4, 0, 0, 1.0), pi(4, 0, 0, 1))
    # Fine page (16,0,0): first slot of compressed page (1,0,0).
    expect_voxels(along_x(v, 63, 64, 0, 0, 1.0), pi(64, 0, 0, 1))
    # Fine page (-1,-1,-1): last slot of compressed page (-1,-1,-1).
    expect_voxels(from_above(v, -1, -1, -1, 1.0), pi(-4, -4, -2, 1 << 31))


def compress_block_coords_far(impl):
    v = compressed(impl)
    expect_voxels(from_above(v, 5, 9, 7, 1.0), pi(4, 8, 6, 1 << 21))
    expect_voxels(from_above(v, -5, -9, -6, 1.0), pi(-8, -12, -6, 1 << 15))


def compress_mask_accumulates_across_calls(impl):
    v = compressed(impl)
    expect_voxels(from_above(v, 0, 0, 0, 1.0), pi(0, 0, 0, 1))
    # The pack is the block's full current state, not a delta.
    expect_voxels(from_above(v, 1, 0, 0, 1.0), pi(0, 0, 0, 0x3))
    expect_voxels(from_above(v, 2, 0, 1, 1.0), pi(0, 0, 0, 0x3 | 1 << 18))


# ----------------------------------------------------------- max value


def max_value_caps_target(impl):
    v = make(impl, 1.0, 1, 1, 0, 1, False, max_value=5)
    expect_voxels(from_above(v, 3, 0, 0, 3.0), pi(3, 0, 0, 3))
    expect_voxels(from_above(v, 3, 0, 0, 3.0), pi(3, 0, 0, 5))
    # Already at the cap: nothing changes, nothing is reported.
    expect_no_voxels(from_above(v, 3, 0, 0, 3.0))


# ------------------------------------------------ intensity multiplier


def intensity_multiplier_scales_hit_and_target(impl):
    v = make(impl, 1.0, 2, 3, 0, 1, False, intensity_multiplier=2.5)
    # Seeds (0,0,0) with (4 * 2.5) * 3 = 30.
    seed(v, 1.0, 0, 0, 0, 4.0)
    # int(3 * 2.5) = 7, times 2 subtracted and times 3 added.
    expect_voxels(add_point(v, sensor(), pf(3.5, 0.5, 0.5, 3.0)),
                  pi(0, 0, 0, 30 - 7 * 2), pi(3, 0, 0, 7 * 3))
    # Scaled before truncation: int(0.5 * 2.5) = 1.
    expect_voxels(from_above(v, 5, 0, 0, 0.5), pi(5, 0, 0, 1 * 3))


# ------------------------------------------------------ set multipliers


def get_multipliers_returns_constructor_values(impl):
    multipliers = plain(impl).get_multipliers()
    check(type(multipliers) is rv.Multipliers,
          f"get_multipliers returned a {type(multipliers).__name__}")
    check(multiplier_values(multipliers) == (1.0, 1, 1))
    v = make(impl, 1.0, 2, 3, 0, 1, False, intensity_multiplier=2.5)
    check(multiplier_values(v.get_multipliers()) == (2.5, 2, 3))


def get_multipliers_returns_set_values(impl):
    v = plain(impl)
    for values in [(2.5, 2, 3), (0.0, 0, 0), (-1.5, U32_MAX, U32_MAX), (1.0, 1, 1)]:
        v.set_multipliers(rv.Multipliers(*values))
        got = multiplier_values(v.get_multipliers())
        check(got == values, f"set {values}, got {got}")
    # The last of several sets wins, adding points leaves it as it is.
    v.set_multipliers(rv.Multipliers(2.5, 2, 3))
    v.set_multipliers(rv.Multipliers(1.5, 2, 3))
    add_point(v, sensor(), pf(3.5, 0.5, 0.5, 1.0))
    check(multiplier_values(v.get_multipliers()) == (1.5, 2, 3))


def get_multipliers_returns_copy(impl):
    v = plain(impl)
    multipliers = rv.Multipliers(2.5, 2, 3)
    v.set_multipliers(multipliers)
    # Changing the object passed in or the one returned leaves the voxelizer
    # as it was.
    multipliers.set_hit(100)
    v.get_multipliers().set_target(100)
    check(multiplier_values(v.get_multipliers()) == (2.5, 2, 3))
    check(multiplier_values(multipliers) == (2.5, 100, 3))


def set_multipliers_applies_to_next_call(impl):
    v = plain(impl)
    seed(v, 1.0, 0, 0, 0, 20.0)
    point = pf(3.5, 0.5, 0.5, 3.0)
    expect_voxels(add_point(v, sensor(), point), pi(0, 0, 0, 17), pi(3, 0, 0, 3))
    v.set_multipliers(rv.Multipliers(2.5, 2, 3))
    # int(3 * 2.5) = 7, times 2 subtracted from and times 3 added to what the
    # voxels hold already.
    expect_voxels(add_point(v, sensor(), point), pi(0, 0, 0, 17 - 7 * 2), pi(3, 0, 0, 3 + 7 * 3))
    # Setting the old ones back restores the old behaviour.
    v.set_multipliers(rv.Multipliers(1.0, 1, 1))
    expect_voxels(add_point(v, sensor(), point), pi(0, 0, 0, 0), pi(3, 0, 0, 24 + 3))


def set_multipliers_rejects_wrong_type(impl):
    v = plain(impl)
    for bad in (None, (2.5, 2, 3), pf(2.5, 2.0, 3.0, 0.0)):
        with raises(TypeError):
            v.set_multipliers(bad)
    with raises(TypeError):
        v.set_multipliers()
    # Probe: none of the failed calls changed anything.
    check(multiplier_values(v.get_multipliers()) == (1.0, 1, 1))
    expect_voxels(add_point(v, sensor(), pf(3.5, 0.5, 0.5, 1.0)), pi(3, 0, 0, 1))


def set_multipliers_same_as_constructed(impl):
    for destroy, construct, compress in SCENARIO_PARAMS:
        for intensity, hit, target in [(1.5, 2, 3), (0.5, 1, 4)]:
            v = make(impl, 1.0, 1, 1, destroy, construct, compress)
            v.set_multipliers(rv.Multipliers(intensity, hit, target))
            a = scenario(v)
            b = scenario(make(impl, 1.0, hit, target, destroy, construct, compress,
                              intensity_multiplier=intensity))
            check(a == b, f"compress = {compress}, set to ({intensity}, {hit}, {target}): "
                          "runs differ")


# ------------------------------------------------------------ scenario


def scenario_repeatable(impl):
    for destroy, construct, compress in SCENARIO_PARAMS:
        a = scenario(make(impl, 1.0, 1, 1, destroy, construct, compress))
        for call, update in enumerate(a):
            check(len(update) > 0, f"compress = {compress}: call {call} changed nothing")
        b = scenario(make(impl, 1.0, 1, 1, destroy, construct, compress))
        check(a == b, f"compress = {compress}: two runs differ")


# ---------------------------------------------------- global (run once)


def single_and_multi_agree():
    for destroy, construct, compress in SCENARIO_PARAMS:
        expected = scenario(make(SINGLE, 1.0, 1, 1, destroy, construct, compress))
        for call, update in enumerate(expected):
            check(len(update) > 0, f"compress = {compress}: call {call} changed nothing")
        for threads in (1, 2, 4, 8):
            multi = multi_threaded("multi", threads)
            got = scenario(make(multi, 1.0, 1, 1, destroy, construct, compress))
            check(got == expected, f"compress = {compress}, threads = {threads}: differs")


def multi_thread_count_zero():
    # 0 means the default thread count.
    basic_1d(multi_threaded("multi_0", 0))


def single_and_wgpu_agree():
    for destroy, construct, compress in SCENARIO_PARAMS:
        expected = scenario(make(SINGLE, 1.0, 1, 1, destroy, construct, compress))
        for call, update in enumerate(expected):
            check(len(update) > 0, f"compress = {compress}: call {call} changed nothing")
        got = scenario(make(WGPU, 1.0, 1, 1, destroy, construct, compress))
        check(got == expected, f"compress = {compress}: differs")


def wgpu_rejects_zero_buckets():
    with raises(RuntimeError):
        rv.WgpuAccelerated(1.0, 1.0, 1, 1, 0, 1, U32_MAX, False, 0)


def wgpu_memory_size_grows():
    sizes = [rv.WgpuAccelerated.memory_size(n) for n in (1, 2, WGPU_TEST_BUCKETS)]
    check(all(type(size) is int for size in sizes), f"memory_size returned {sizes}, not ints")
    check(0 < sizes[0] < sizes[1] < sizes[2], f"memory sizes {sizes} do not grow")


def compress_helpers_self_check():
    check(block_bit(3, 3, 1) == 1 << 31)
    check(block_bit(3, 3, 3) == 1 << 31)
    check(block(3, 0, 0, block_bit(3, 0, 0)) == pi(0, 0, 0, 1 << 3))
    check(block(5, 9, 7, 0) == pi(4, 8, 6, 0))
    check(block(-1, -1, -1, block_bit(-1, -1, -1)) == pi(-4, -4, -2, 1 << 31))
    check(block(-5, -9, -6, 0) == pi(-8, -12, -6, 0))


# --------------------------------------------------------------- table


def per_impl(fn):
    return (fn.__name__, fn, True, ())


def once(fn, *needs):
    """A case run once overall, listed only when the features in needs are
    enabled."""
    return (fn.__name__, fn, False, needs)


TEST_CASES = [
    once(module_exports_match_features),
    once(classes_belong_to_module),
    once(point_f_positional),
    once(point_f_keywords),
    once(point_f_from_ints),
    once(point_f_rejects_non_numbers),
    once(point_f_setters),
    once(point_i_positional),
    once(point_i_keywords),
    once(point_i_rejects_out_of_range),
    once(point_i_setters),
    once(multipliers_positional),
    once(multipliers_keywords),
    once(multipliers_rejects_out_of_range),
    once(multipliers_setters),
    per_impl(constructors_reject_out_of_range),
    per_impl(constructors_accept_u32_max),
    per_impl(constructors_accept_keywords),
    per_impl(add_points_result_types),
    per_impl(add_points_sees_setter_values),
    per_impl(add_points_accepts_tuple),
    per_impl(add_points_rejects_wrong_types),
    per_impl(add_points_leaves_inputs_unchanged),
    per_impl(instances_are_independent),
    per_impl(create),
    per_impl(create_drop_cycles),
    per_impl(basic_1d),
    per_impl(empty_points_no_change),
    per_impl(returns_absolute_value_not_delta),
    per_impl(sensor_intensity_ignored),
    per_impl(intensity_truncated_toward_zero),
    per_impl(zero_intensity_reports_nothing),
    per_impl(resolution_scales_coords_not_intensity),
    per_impl(resolution_scales_sensor_too),
    per_impl(large_coords),
    per_impl(negative_coords_use_floor),
    per_impl(hit_multiplier_scales_hits),
    per_impl(target_multiplier_scales_target),
    per_impl(both_multipliers),
    per_impl(hit_saturates_at_zero),
    per_impl(hit_on_zero_voxel_not_reported),
    per_impl(thresholds_ignored_without_compress),
    per_impl(accumulates_across_calls),
    per_impl(unchanged_voxel_not_reported),
    per_impl(second_call_reports_only_its_changes),
    per_impl(multiple_points_one_call),
    per_impl(hits_skip_missing_pages),
    per_impl(negative_page),
    per_impl(axis_x_negative),
    per_impl(axis_y_positive),
    per_impl(axis_y_negative),
    per_impl(axis_z_positive),
    per_impl(axis_z_negative),
    per_impl(plane_xy_forward),
    per_impl(plane_xy_reverse),
    per_impl(plane_xz),
    per_impl(plane_yz),
    per_impl(full_3d_forward),
    per_impl(full_3d_reverse),
    per_impl(sensor_on_grid_line_2d),
    per_impl(ray_across_page_boundary),
    per_impl(point_in_sensor_voxel_fresh),
    per_impl(point_in_sensor_voxel_accumulates),
    per_impl(compress_construct_sets_bit),
    per_impl(compress_no_recross_no_report),
    per_impl(compress_plain_changes_not_returned),
    per_impl(compress_destroy_clears_bit),
    per_impl(compress_construct_threshold),
    per_impl(compress_destroy_threshold),
    per_impl(compress_bit_positions),
    per_impl(compress_both_halves_same_page),
    per_impl(compress_two_voxels_same_block),
    per_impl(compress_low_half_change_with_high_bits_present),
    per_impl(compress_page_keys),
    per_impl(compress_block_coords_far),
    per_impl(compress_mask_accumulates_across_calls),
    per_impl(max_value_caps_target),
    per_impl(intensity_multiplier_scales_hit_and_target),
    per_impl(get_multipliers_returns_constructor_values),
    per_impl(get_multipliers_returns_set_values),
    per_impl(get_multipliers_returns_copy),
    per_impl(set_multipliers_applies_to_next_call),
    per_impl(set_multipliers_rejects_wrong_type),
    per_impl(set_multipliers_same_as_constructed),
    per_impl(scenario_repeatable),
    once(single_and_multi_agree, "single_threaded", "multi_threaded"),
    once(multi_thread_count_zero, "multi_threaded"),
    once(single_and_wgpu_agree, "single_threaded", "wgpu"),
    once(wgpu_rejects_zero_buckets, "wgpu"),
    once(wgpu_memory_size_grows, "wgpu"),
    once(compress_helpers_self_check),
]

if __name__ == "__main__":
    sys.exit(main(TEST_CASES))
