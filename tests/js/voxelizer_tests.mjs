/**
 * Test cases for the JS bindings: checks of the JS facing API (constructors,
 * properties, argument conversion and ownership) followed by a port of the
 * Python binding tests in tests/python/voxelizer_tests.py.
 *
 * Test cases keep the snake_case names of the C and Python suites. Per
 * implementation cases run against every entry of IMPLS (harness.mjs).
 *
 * Unless a case is about reusing points, every add_points call gets PointF
 * objects of its own.
 *
 * WgpuAccelerated is created and adds points asynchronously, so cases await
 * what voxelizers return (see harness.mjs).
 */

import process from "node:process";

import {
    CLASS_OF_FEATURE, FEATURES, SCENARIO_PARAMS, SEED, SINGLE, UNSUPPORTED_CLASSES, WGPU,
    WGPU_TEST_BUCKETS, addPoint, alongX, block, blockBit, check, checkCoords, checkRay, checkUpdate,
    compressed, coords, expectNoVoxels, expectVoxels, fail, fmt, fromAbove, main, make, pf, pi, plain,
    rv, same, scenario, seed, sensor, throwsError,
} from "./harness.mjs";

const U32_MAX = 2 ** 32 - 1;

// -------------------------------------------------------------- JS API

/** Names of the classes the module must have with the enabled features. */
function enabledClasses() {
    return ["PointF", "PointI", ...Object.entries(CLASS_OF_FEATURE)
        .filter(([feature]) => FEATURES.has(feature))
        .map(([, cls]) => cls)];
}

function module_exports_match_features() {
    for (const name of enabledClasses()) {
        check(typeof rv[name] === "function", `r3v_voxelizer.${name} is missing`);
    }
    for (const [feature, name] of Object.entries(CLASS_OF_FEATURE)) {
        if (!FEATURES.has(feature)) {
            check(!(name in rv), `r3v_voxelizer.${name} exists without ${feature}`);
        }
    }
    for (const name of UNSUPPORTED_CLASSES) {
        check(!(name in rv), `r3v_voxelizer.${name} exists, but it is not supported in JS`);
    }
}

function point_f_constructor() {
    const point = new rv.PointF(1.5, -2.25, 3.0, 4.0);
    check(point instanceof rv.PointF);
    checkCoords(point, [1.5, -2.25, 3, 4], "new PointF(1.5, -2.25, 3, 4)");
}

function point_f_stores_f32() {
    const values = [0.1, 1e-10, 3.4e38, -0.3];
    checkCoords(new rv.PointF(...values), values.map(Math.fround), `new PointF(${values})`);
}

function point_f_properties_writable() {
    const point = pf(0.0, 0.0, 0.0, 0.0);
    point.x = 1.5;
    point.y = -2.25;
    point.z = 3;
    point.i = 4.0;
    checkCoords(point, [1.5, -2.25, 3, 4], "PointF after setting its properties");
}

/** A new PointI, failing with a clear reason when it cannot be built. */
function newPointI(x, y, z, i) {
    const point = new rv.PointI(x, y, z, i);
    check(point instanceof rv.PointI);
    checkCoords(point, [x, y, z, i], `new PointI(${x}, ${y}, ${z}, ${i})`);
    return point;
}

function point_i_constructor() {
    newPointI(1, -2, 3, 4);
    newPointI(-(2 ** 31), 2 ** 31 - 1, 0, U32_MAX);
}

function point_i_properties_writable() {
    const point = newPointI(0, 0, 0, 0);
    point.x = 1;
    point.y = -2;
    point.z = 3;
    point.i = U32_MAX;
    checkCoords(point, [1, -2, 3, U32_MAX], "PointI after setting its properties");
}

function point_i_setter_wraps() {
    // Out of range values wrap like any wasm-bindgen integer (see the README).
    const point = newPointI(0, 0, 0, 0);
    point.x = 2 ** 31;
    point.y = 1.9;
    point.i = -1;
    checkCoords(point, [-(2 ** 31), 1, 0, U32_MAX], "PointI after setting out of range values");
}

async function constructors_accept_u32_max(impl) {
    const v = await make(impl, 1.0, U32_MAX, U32_MAX, U32_MAX, U32_MAX, true);
    check(v instanceof rv[impl.cls]);
}

async function constructors_wrap_out_of_range(impl) {
    // A hit multiplier of 2 ** 32 + 3 wraps to 3 (see the README), so this
    // is hit_multiplier_scales_hits.
    const v = await make(impl, 1.0, 2 ** 32 + 3, 1, 0, 1, false);
    await seed(v, 1.0, 0, 0, 0, SEED);
    await expectVoxels(addPoint(v, sensor(), pf(3.5, 0.5, 0.5, 2.0)),
        pi(0, 0, 0, SEED - 2 * 3), pi(3, 0, 0, 2));
}

async function add_points_result_types(impl) {
    const v = await plain(impl);
    const result = addPoint(v, sensor(), pf(3.5, 0.5, 0.5, 1.0));
    check(result instanceof Promise === impl.isAsync,
        `add_points returned ${impl.isAsync ? "no" : "a"} Promise`);
    const update = await result;
    check(Array.isArray(update), "add_points did not return an Array");
    check(update.length === 1, `add_points returned ${update.length} points, expected 1`);
    const point = update[0];
    check(point instanceof rv.PointI, "add_points did not return PointI objects");
    check(coords(point).every(Number.isInteger), `PointI holds ${fmt(coords(point))}`);
    // The result is owned by JS and outlives the voxelizer.
    v.free();
    checkCoords(point, [3, 0, 0, 1], "PointI after freeing its voxelizer");
}

// The caller keeps owning what it passes to add_points.

async function add_points_leaves_inputs_usable(impl) {
    const v = await make(impl, 2.0, 1, 1, 0, 1, false);
    const sensorPos = pf(2.0, 0.0, 0.0, 1.0);
    const points = [pf(7.0, 0.0, 0.0, 1.0)];
    await expectVoxels(v.add_points(sensorPos, points), pi(3, 0, 0, 1));
    check(points.length === 1, `points array now holds ${points.length} points`);
    checkCoords(sensorPos, [2, 0, 0, 1], "sensor after add_points");
    checkCoords(points[0], [7, 0, 0, 1], "point after add_points");
}

async function add_points_reuses_inputs(impl) {
    const v = await plain(impl);
    const sensorPos = sensor();
    const point = pf(3.5, 0.5, 0.5, 1.0);
    await expectVoxels(addPoint(v, sensorPos, point), pi(3, 0, 0, 1));
    // The same objects once more: the target accumulates.
    await expectVoxels(addPoint(v, sensorPos, point), pi(3, 0, 0, 2));
}

async function add_points_accepts_repeated_point(impl) {
    const v = await plain(impl);
    const point = pf(3.5, 0.5, 0.5, 1.0);
    await expectVoxels(v.add_points(sensor(), [point, point]), pi(3, 0, 0, 2));
}

async function add_points_accepts_point_like_objects(impl) {
    // Anything with numeric x, y, z and i is a point.
    const v = await plain(impl);
    const points = [{ x: 3.5, y: 0.5, z: 0.5, i: 1.0 }, new rv.PointI(0, 3, 0, 2)];
    await expectVoxels(v.add_points(sensor(), points), pi(3, 0, 0, 1), pi(0, 3, 0, 2));
}

// Rejected calls throw an Error, store nothing and leave the voxelizer
// usable. Bad values are built outside of throwsError, so a failure to build
// them does not count as a rejection.

function freedPoint() {
    const point = pf(3.5, 0.5, 0.5, 1.0);
    point.free();
    return point;
}

/** Probe: the voxelizer still works and none of the rejected calls stored
 * anything. */
async function expectUntouched(v) {
    const sensorPos = sensor();
    const point = pf(3.5, 0.5, 0.5, 1.0);
    let update;
    try {
        update = await addPoint(v, sensorPos, point);
    } catch (e) {
        fail(`voxelizer unusable after the rejected calls: ${e.message}`);
    }
    await expectVoxels(update, pi(3, 0, 0, 1));
}

async function add_points_rejects_bad_sensor(impl) {
    const v = await plain(impl);
    const bad = [
        ["null", null],
        ["undefined", undefined],
        ["a plain object", { x: 0.5, y: 0.5, z: 0.5, i: 1.0 }],
        ["a PointI", new rv.PointI(0, 0, 0, 1)],
        ["a freed PointF", freedPoint()],
    ];
    for (const [what, sensorPos] of bad) {
        const points = [pf(3.5, 0.5, 0.5, 1.0)];
        await throwsError(() => v.add_points(sensorPos, points), `add_points with ${what} as sensor`);
    }
    await expectUntouched(v);
}

async function add_points_rejects_non_array_points(impl) {
    const v = await plain(impl);
    const bad = [
        ["null", null],
        ["undefined", undefined],
        ["a single PointF", pf(3.5, 0.5, 0.5, 1.0)],
        ["a number", 42],
    ];
    for (const [what, points] of bad) {
        const sensorPos = sensor();
        await throwsError(() => v.add_points(sensorPos, points), `add_points with ${what} as points`);
    }
    await expectUntouched(v);
}

async function add_points_rejects_bad_array_element(impl) {
    const v = await plain(impl);
    const bad = [
        ["null", () => null],
        ["undefined", () => undefined],
        ["a number", () => 42],
        ["an object without i", () => ({ x: 0.5, y: 3.5, z: 0.5 })],
        ["a string coordinate", () => ({ x: "0.5", y: 3.5, z: 0.5, i: 1.0 })],
        ["a freed PointF", freedPoint],
    ];
    for (const [what, element] of bad) {
        const sensorPos = sensor();
        const points = [pf(3.5, 0.5, 0.5, 1.0), element()];
        await throwsError(() => v.add_points(sensorPos, points), `add_points with ${what} in points`);
    }
    await expectUntouched(v);
}

async function instances_are_independent(impl) {
    const a = await plain(impl);
    const b = await plain(impl);
    await expectVoxels(addPoint(a, sensor(), pf(3.5, 0.5, 0.5, 1.0)), pi(3, 0, 0, 1));
    await expectVoxels(addPoint(a, sensor(), pf(3.5, 0.5, 0.5, 1.0)), pi(3, 0, 0, 2));
    await expectVoxels(addPoint(b, sensor(), pf(3.5, 0.5, 0.5, 1.0)), pi(3, 0, 0, 1));
}

// ------------------------------------------------------------- lifecycle

async function create(impl) {
    const cls = rv[impl.cls];
    for (const v of [await plain(impl), await compressed(impl),
        await make(impl, 0.05, 2, 3, 4, 5, true)]) {
        check(v instanceof cls, `created a ${v.constructor.name}, expected ${cls.name}`);
    }
}

async function create_free_cycles(impl) {
    for (let k = 0; k < 50; k++) {
        const v = await make(impl, 1.0, 1, 1, 0, 1, k % 2 === 0);
        const update = await addPoint(v, sensor(), pf(3.5, 0.5, 0.5, 1.0));
        if (k % 2 === 0) {
            await expectVoxels(update, pi(0, 0, 0, 1 << 3));
        } else {
            await expectVoxels(update, pi(3, 0, 0, 1));
        }
        if (k % 4 < 2) {
            v.free();
        } else {
            v[Symbol.dispose]();
        }
    }
}

async function free_makes_object_unusable(impl) {
    const a = await plain(impl);
    const b = await plain(impl);
    a.free();
    const sensorPos = sensor();
    const points = [pf(3.5, 0.5, 0.5, 1.0)];
    await throwsError(() => a.add_points(sensorPos, points), "add_points on a freed voxelizer");
    // Other instances are not affected.
    await expectVoxels(addPoint(b, sensor(), pf(3.5, 0.5, 0.5, 1.0)), pi(3, 0, 0, 1));
}

// ------------------------------------------------------ add_points basics

async function basic_1d(impl) {
    const v = await plain(impl);
    await expectVoxels(addPoint(v, pf(0.0, 0.0, 0.0, 1.0), pf(5.0, 0.0, 0.0, 1.0)), pi(5, 0, 0, 1));
}

async function empty_points_no_change(impl) {
    const v = await plain(impl);
    await expectNoVoxels(v.add_points(sensor(), []));
    // Nothing was stored, so the target still starts from zero.
    await expectVoxels(addPoint(v, sensor(), pf(3.5, 0.5, 0.5, 1.0)), pi(3, 0, 0, 1));
}

async function returns_absolute_value_not_delta(impl) {
    const v = await plain(impl);
    for (let x = 0; x < 5; x++) {
        await seed(v, 1.0, x, 0, 0, SEED);
    }
    await seed(v, 1.0, 5, 0, 0, 3);
    await expectVoxels(addPoint(v, sensor(), pf(5.5, 0.5, 0.5, 2.0)),
        pi(0, 0, 0, SEED - 2), pi(1, 0, 0, SEED - 2), pi(2, 0, 0, SEED - 2),
        pi(3, 0, 0, SEED - 2), pi(4, 0, 0, SEED - 2), pi(5, 0, 0, 5));
}

async function sensor_intensity_ignored(impl) {
    const v = await plain(impl);
    await seed(v, 1.0, 0, 0, 0, SEED);
    // Hits use the point's intensity, not the sensor's.
    await expectVoxels(addPoint(v, pf(0.5, 0.5, 0.5, 99.0), pf(3.5, 0.5, 0.5, 1.0)),
        pi(0, 0, 0, SEED - 1), pi(3, 0, 0, 1));
}

async function intensity_truncated_toward_zero(impl) {
    const v = await plain(impl);
    await expectVoxels(addPoint(v, sensor(), pf(3.5, 0.5, 0.5, 2.9)), pi(3, 0, 0, 2));
}

async function zero_intensity_reports_nothing(impl) {
    const v = await plain(impl);
    await seed(v, 1.0, 0, 0, 0, SEED);
    await expectNoVoxels(addPoint(v, sensor(), pf(3.5, 0.5, 0.5, 0.0)));
    // Probe: both voxels kept their values.
    await expectVoxels(addPoint(v, sensor(), pf(3.5, 0.5, 0.5, 1.0)),
        pi(0, 0, 0, SEED - 1), pi(3, 0, 0, 1));
}

async function resolution_scales_coords_not_intensity(impl) {
    let v = await make(impl, 0.25, 1, 1, 0, 1, false);
    for (let x = 0; x < 4; x++) {
        await seed(v, 0.25, x, 0, 0, SEED);
    }
    await expectVoxels(addPoint(v, pf(0.1, 0.1, 0.1, 1.0), pf(1.0, 0.1, 0.1, 7.0)),
        pi(0, 0, 0, SEED - 7), pi(1, 0, 0, SEED - 7), pi(2, 0, 0, SEED - 7),
        pi(3, 0, 0, SEED - 7), pi(4, 0, 0, 7));

    v = await make(impl, 2.0, 1, 1, 0, 1, false);
    await expectVoxels(addPoint(v, pf(0.0, 0.0, 0.0, 1.0), pf(5.0, 0.0, 0.0, 7.0)), pi(2, 0, 0, 7));
}

async function resolution_scales_sensor_too(impl) {
    const v = await make(impl, 2.0, 1, 1, 0, 1, false);
    await seed(v, 2.0, 1, 0, 0, SEED);
    await seed(v, 2.0, 2, 0, 0, SEED);
    // sensor 2.0 -> 1.0 -> voxel 1, point 7.0 -> 3.5 -> voxel 3
    await expectVoxels(addPoint(v, pf(2.0, 0.0, 0.0, 1.0), pf(7.0, 0.0, 0.0, 1.0)),
        pi(1, 0, 0, SEED - 1), pi(2, 0, 0, SEED - 1), pi(3, 0, 0, 1));
}

async function large_coords(impl) {
    const v = await plain(impl);
    const sensorPos = () => pf(1000.5, -2000.5, 29999.5, 1.0);
    const point = () => pf(1000.5, -2000.5, 30000.5, 4.0);
    await expectVoxels(addPoint(v, sensorPos(), point()), pi(1000, -2001, 30000, 4));
    // Probe: the value was stored where it was reported.
    await expectVoxels(addPoint(v, sensorPos(), point()), pi(1000, -2001, 30000, 8));
}

async function negative_coords_use_floor(impl) {
    for (const [x, voxel] of [[-0.5, -1], [-1.0, -1], [-1.5, -2]]) {
        const v = await plain(impl);
        await expectVoxels(addPoint(v, pf(-3.5, 0.5, 0.5, 1.0), pf(x, 0.5, 0.5, 1.0)),
            pi(voxel, 0, 0, 1));
    }
}

// ------------------------------------------- multipliers / saturation

async function hit_multiplier_scales_hits(impl) {
    const v = await make(impl, 1.0, 3, 1, 0, 1, false);
    await seed(v, 1.0, 0, 0, 0, SEED);
    await expectVoxels(addPoint(v, sensor(), pf(3.5, 0.5, 0.5, 2.0)),
        pi(0, 0, 0, SEED - 2 * 3), pi(3, 0, 0, 2));
}

async function target_multiplier_scales_target(impl) {
    const v = await make(impl, 1.0, 1, 5, 0, 1, false);
    // Seeding goes through the target path too: 2 * 5 = SEED.
    await seed(v, 1.0, 0, 0, 0, SEED / 5);
    await expectVoxels(addPoint(v, sensor(), pf(3.5, 0.5, 0.5, 2.0)),
        pi(0, 0, 0, SEED - 2), pi(3, 0, 0, 2 * 5));
}

async function both_multipliers(impl) {
    const v = await make(impl, 1.0, 2, 3, 0, 1, false);
    // Seeds (0,0,0) with 4 * 3 = 12.
    await seed(v, 1.0, 0, 0, 0, 4.0);
    const point = () => pf(3.5, 0.5, 0.5, 2.0);
    await expectVoxels(addPoint(v, sensor(), point()), pi(0, 0, 0, 12 - 2 * 2), pi(3, 0, 0, 2 * 3));
    // Probe: values were stored.
    await expectVoxels(addPoint(v, sensor(), point()),
        pi(0, 0, 0, 12 - 2 * 2 * 2), pi(3, 0, 0, 2 * 3 * 2));
}

async function hit_saturates_at_zero(impl) {
    let v = await plain(impl);
    await seed(v, 1.0, 0, 0, 0, 1);
    // (1,0,0) and (2,0,0) exist (same page) and hold 0. 1 -> 0 is a change
    // and gets reported as 0; 0 -> 0 is not.
    await expectVoxels(addPoint(v, sensor(), pf(3.5, 0.5, 0.5, 5.0)), pi(0, 0, 0, 0), pi(3, 0, 0, 5));
    // Probe: nothing wrapped around below zero.
    await expectVoxels(addPoint(v, sensor(), pf(3.5, 0.5, 0.5, 1.0)), pi(3, 0, 0, 6));

    // The multiplier saturates too.
    v = await make(impl, 1.0, 100, 1, 0, 1, false);
    await seed(v, 1.0, 0, 0, 0, SEED);
    await expectVoxels(addPoint(v, sensor(), pf(3.5, 0.5, 0.5, 1.0)), pi(0, 0, 0, 0), pi(3, 0, 0, 1));
}

async function hit_on_zero_voxel_not_reported(impl) {
    const v = await plain(impl);
    // Creates page (0,0,0) without touching the ray below.
    await seed(v, 1.0, 1, 1, 1, SEED);
    await expectVoxels(addPoint(v, sensor(), pf(3.5, 0.5, 0.5, 1.0)), pi(3, 0, 0, 1));
}

async function thresholds_ignored_without_compress(impl) {
    const v = await make(impl, 1.0, 1, 1, 100, 200, false);
    for (let x = 0; x < 5; x++) {
        await seed(v, 1.0, x, 0, 0, SEED);
    }
    await seed(v, 1.0, 5, 0, 0, 3);
    await expectVoxels(addPoint(v, sensor(), pf(5.5, 0.5, 0.5, 2.0)),
        pi(0, 0, 0, SEED - 2), pi(1, 0, 0, SEED - 2), pi(2, 0, 0, SEED - 2),
        pi(3, 0, 0, SEED - 2), pi(4, 0, 0, SEED - 2), pi(5, 0, 0, 5));
}

// ------------------------------------- accumulation / change detection

async function accumulates_across_calls(impl) {
    const v = await plain(impl);
    for (let x = 0; x < 3; x++) {
        await seed(v, 1.0, x, 0, 0, SEED);
    }
    const point = () => pf(3.5, 0.5, 0.5, 1.0);
    await expectVoxels(addPoint(v, sensor(), point()),
        pi(0, 0, 0, SEED - 1), pi(1, 0, 0, SEED - 1), pi(2, 0, 0, SEED - 1),
        pi(3, 0, 0, 1));
    await expectVoxels(addPoint(v, sensor(), point()),
        pi(0, 0, 0, SEED - 2), pi(1, 0, 0, SEED - 2), pi(2, 0, 0, SEED - 2),
        pi(3, 0, 0, 2));
}

async function unchanged_voxel_not_reported(impl) {
    const v = await plain(impl);
    await seed(v, 1.0, 0, 0, 0, 5);
    await seed(v, 1.0, 1, 0, 0, 1);
    // First ray hits (1,0,0): 1 -> 0, second ray ends there: 0 -> 1.
    // (The other order gives 1 -> 2 -> 1: same outcome.)
    const points = [pf(2.5, 0.5, 0.5, 1.0), pf(1.5, 0.5, 0.5, 1.0)];
    await expectVoxels(v.add_points(sensor(), points), pi(0, 0, 0, 3), pi(2, 0, 0, 1));
    // Probe: (1,0,0) holds 1, so one more hit takes it to 0.
    await expectVoxels(addPoint(v, pf(1.5, 0.5, 0.5, 1.0), pf(1.5, 1.5, 0.5, 1.0)),
        pi(1, 0, 0, 0), pi(1, 1, 0, 1));
}

async function second_call_reports_only_its_changes(impl) {
    const v = await plain(impl);
    for (let x = 0; x < 3; x++) {
        await seed(v, 1.0, x, 0, 0, SEED);
    }
    // Straight up along z: hits (0,0,0), (0,0,1), (0,0,2).
    await expectVoxels(addPoint(v, sensor(), pf(0.5, 0.5, 3.5, 1.0)),
        pi(0, 0, 0, SEED - 1), pi(0, 0, 3, 1));
    await expectVoxels(addPoint(v, sensor(), pf(3.5, 0.5, 0.5, 1.0)),
        pi(0, 0, 0, SEED - 2), pi(1, 0, 0, SEED - 1), pi(2, 0, 0, SEED - 1),
        pi(3, 0, 0, 1));
}

async function multiple_points_one_call(impl) {
    const v = await plain(impl);
    for (const [x, y] of [[0, 0], [1, 0], [2, 0], [0, 1], [0, 2]]) {
        await seed(v, 1.0, x, y, 0, SEED);
    }
    const points = [pf(3.5, 0.5, 0.5, 1.0), pf(0.5, 3.5, 0.5, 2.0)];
    await expectVoxels(v.add_points(sensor(), points),
        pi(0, 0, 0, SEED - 1 - 2), pi(1, 0, 0, SEED - 1), pi(2, 0, 0, SEED - 1),
        pi(3, 0, 0, 1), pi(0, 1, 0, SEED - 2), pi(0, 2, 0, SEED - 2), pi(0, 3, 0, 2));
}

// ------------------------------------------------------------- paging

async function hits_skip_missing_pages(impl) {
    const v = await plain(impl);
    await seed(v, 1.0, 0, 0, 0, SEED);
    const point = () => pf(20.5, 0.5, 0.5, 1.0);
    // The ray crosses fine pages 0..=5 along x but only page 0 exists.
    await expectVoxels(addPoint(v, sensor(), point()), pi(0, 0, 0, SEED - 1), pi(20, 0, 0, 1));
    await expectVoxels(addPoint(v, sensor(), point()), pi(0, 0, 0, SEED - 2), pi(20, 0, 0, 2));
    // Once a page exists on the way its voxels do get hit.
    await seed(v, 1.0, 8, 0, 0, SEED);
    await expectVoxels(addPoint(v, sensor(), point()),
        pi(0, 0, 0, SEED - 3), pi(8, 0, 0, SEED - 1), pi(20, 0, 0, 3));
}

async function negative_page(impl) {
    const v = await plain(impl);
    const sensorPos = () => pf(-0.5, -0.5, -0.5, 1.0);
    const point = () => pf(-3.5, -0.5, -0.5, 3.0);
    await expectVoxels(addPoint(v, sensorPos(), point()), pi(-4, -1, -1, 3));
    // Probe: the page now exists, the voxels on the way hold 0.
    await expectVoxels(addPoint(v, sensorPos(), point()), pi(-4, -1, -1, 6));
}

// ----------------------------------------------------- ray march arms

async function axis_x_negative(impl) {
    await checkRay(impl, pf(5.5, 0.5, 0.5, 1.0), pf(0.5, 0.5, 0.5, 1.0), [0, 0, 0],
        [5, 0, 0], [4, 0, 0], [3, 0, 0], [2, 0, 0], [1, 0, 0]);
}

async function axis_y_positive(impl) {
    await checkRay(impl, pf(0.5, 0.5, 0.5, 1.0), pf(0.5, 4.5, 0.5, 1.0), [0, 4, 0],
        [0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 3, 0]);
}

async function axis_y_negative(impl) {
    await checkRay(impl, pf(0.5, 4.5, 0.5, 1.0), pf(0.5, 0.5, 0.5, 1.0), [0, 0, 0],
        [0, 4, 0], [0, 3, 0], [0, 2, 0], [0, 1, 0]);
}

async function axis_z_positive(impl) {
    await checkRay(impl, pf(0.5, 0.5, 0.5, 1.0), pf(0.5, 0.5, 4.5, 1.0), [0, 0, 4],
        [0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]);
}

async function axis_z_negative(impl) {
    await checkRay(impl, pf(0.5, 0.5, 4.5, 1.0), pf(0.5, 0.5, 0.5, 1.0), [0, 0, 0],
        [0, 0, 4], [0, 0, 3], [0, 0, 2], [0, 0, 1]);
}

async function plane_xy_forward(impl) {
    await checkRay(impl, pf(0.5, 0.5, 0.5, 1.0), pf(3.5, 1.7, 0.5, 1.0), [3, 1, 0],
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [2, 1, 0]);
}

async function plane_xy_reverse(impl) {
    await checkRay(impl, pf(3.5, 1.7, 0.5, 1.0), pf(0.5, 0.5, 0.5, 1.0), [0, 0, 0],
        [3, 1, 0], [2, 1, 0], [1, 1, 0], [1, 0, 0]);
}

async function plane_xz(impl) {
    await checkRay(impl, pf(0.5, 0.5, 0.5, 1.0), pf(3.5, 0.5, 1.7, 1.0), [3, 0, 1],
        [0, 0, 0], [1, 0, 0], [1, 0, 1], [2, 0, 1]);
}

async function plane_yz(impl) {
    await checkRay(impl, pf(0.5, 0.5, 0.5, 1.0), pf(0.5, 3.5, 1.7, 1.0), [0, 3, 1],
        [0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 2, 1]);
}

async function full_3d_forward(impl) {
    await checkRay(impl, pf(0.5, 0.5, 0.5, 1.0), pf(2.5, 1.6, 3.5, 1.0), [2, 1, 3],
        [0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [1, 1, 2], [2, 1, 2]);
}

async function full_3d_reverse(impl) {
    await checkRay(impl, pf(2.5, 1.6, 3.5, 1.0), pf(0.5, 0.5, 0.5, 1.0), [0, 0, 0],
        [2, 1, 3], [2, 1, 2], [1, 1, 2], [1, 1, 1], [1, 0, 1], [0, 0, 1]);
}

async function sensor_on_grid_line_2d(impl) {
    // Line y = 0.6x from the origin: crosses x=1 at t=0.4 (y=0.6), y=1 at
    // t=0.667 (x=1.67), x=2 at t=0.8.
    await checkRay(impl, pf(0.0, 0.0, 0.0, 1.0), pf(2.5, 1.5, 0.5, 1.0), [2, 1, 0],
        [0, 0, 0], [1, 0, 0], [1, 1, 0]);
}

async function ray_across_page_boundary(impl) {
    // Hits span fine pages (0,0,0) and (1,0,0) along x.
    await checkRay(impl, pf(2.5, 0.5, 0.5, 1.0), pf(6.5, 1.7, 0.5, 1.0), [6, 1, 0],
        [2, 0, 0], [3, 0, 0], [4, 0, 0], [4, 1, 0], [5, 1, 0]);
}

// -------------------------------------- point inside the sensor voxel

async function point_in_sensor_voxel_fresh(impl) {
    const v = await plain(impl);
    await expectVoxels(addPoint(v, sensor(), pf(0.7, 0.5, 0.5, 3.0)), pi(0, 0, 0, 3));
}

async function point_in_sensor_voxel_accumulates(impl) {
    // The target voxel only ever receives +i in every other arm, so the
    // degenerate arm must add as well instead of cancelling itself out.
    const v = await plain(impl);
    await expectVoxels(addPoint(v, sensor(), pf(0.7, 0.5, 0.5, 5.0)), pi(0, 0, 0, 5));
    await expectVoxels(addPoint(v, sensor(), pf(0.7, 0.5, 0.5, 3.0)), pi(0, 0, 0, 8));
}

// ------------------------------------------------------- compress mode

async function compress_construct_sets_bit(impl) {
    const v = await compressed(impl);
    await expectVoxels(alongX(v, 0, 3, 0, 0, 1.0), pi(0, 0, 0, 1 << 3));
}

async function compress_no_recross_no_report(impl) {
    const v = await compressed(impl);
    await expectVoxels(alongX(v, 0, 3, 0, 0, 1.0), block(3, 0, 0, blockBit(3, 0, 0)));
    // 1 -> 2 stays above the construct threshold: nothing to report, even
    // though the voxel itself changed.
    await expectNoVoxels(alongX(v, 0, 3, 0, 0, 1.0));
    // Probe: (3,0,0) holds 2, so one more hit does not destroy it.
    await expectVoxels(alongX(v, 0, 7, 0, 0, 1.0), block(7, 0, 0, blockBit(7, 0, 0)));
}

async function compress_plain_changes_not_returned(impl) {
    // Below the construct threshold a voxel change is invisible in compress
    // mode, while plain mode reports it.
    const c = await make(impl, 1.0, 1, 1, 0, 5, true);
    await expectNoVoxels(alongX(c, 0, 3, 0, 0, 1.0));
    const p = await make(impl, 1.0, 1, 1, 0, 5, false);
    await expectVoxels(alongX(p, 0, 3, 0, 0, 1.0), pi(3, 0, 0, 1));
}

async function compress_destroy_clears_bit(impl) {
    const v = await compressed(impl);
    await expectVoxels(alongX(v, 0, 3, 0, 0, 1.0), block(3, 0, 0, blockBit(3, 0, 0)));
    // Passes through (3,0,0): 1 -> 0, destroyed. Ends in (7,0,0): constructed.
    await expectVoxels(alongX(v, 0, 7, 0, 0, 1.0),
        block(3, 0, 0, 0), block(7, 0, 0, blockBit(7, 0, 0)));
}

async function compress_construct_threshold(impl) {
    let v = await make(impl, 1.0, 1, 1, 0, 3, true);
    await expectNoVoxels(alongX(v, 0, 3, 0, 0, 1.0));
    await expectNoVoxels(alongX(v, 0, 3, 0, 0, 1.0));
    await expectVoxels(alongX(v, 0, 3, 0, 0, 1.0), block(3, 0, 0, blockBit(3, 0, 0)));

    // Jumping over the threshold in one call constructs as well.
    v = await make(impl, 1.0, 1, 1, 0, 3, true);
    await expectVoxels(alongX(v, 0, 3, 0, 0, 7.0), block(3, 0, 0, blockBit(3, 0, 0)));
}

async function compress_destroy_threshold(impl) {
    const v = await make(impl, 1.0, 1, 1, 2, 5, true);
    await expectVoxels(alongX(v, 0, 3, 0, 0, 5.0), block(3, 0, 0, blockBit(3, 0, 0)));
    // 5 -> 4 stays above the destroy threshold; (7,0,0) -> 1 stays below
    // the construct threshold.
    await expectNoVoxels(alongX(v, 0, 7, 0, 0, 1.0));
    // 4 -> 2 reaches the destroy threshold.
    await expectVoxels(alongX(v, 0, 7, 0, 0, 2.0), block(3, 0, 0, 0));
    // 2 -> 1 is already below: no second destroy.
    await expectNoVoxels(alongX(v, 0, 7, 0, 0, 1.0));
}

async function compress_bit_positions(impl) {
    for (let x = 0; x < 4; x++) {
        for (let y = 0; y < 4; y++) {
            for (let z = 0; z < 4; z++) {
                const v = await compressed(impl);
                const expected = pi(0, 0, z & ~1, (1 << (x + 4 * y + 16 * (z & 1))) >>> 0);
                await checkUpdate(fromAbove(v, x, y, z, 1.0), [expected], `voxel (${x}, ${y}, ${z})`);
            }
        }
    }
}

async function compress_both_halves_same_page(impl) {
    const v = await compressed(impl);
    // Sensor in (0,0,1); one ray down into (0,0,0), one up into (0,0,2).
    const points = [pf(0.5, 0.5, 0.5, 1.0), pf(0.5, 0.5, 2.5, 1.0)];
    await expectVoxels(v.add_points(pf(0.5, 0.5, 1.5, 1.0), points),
        block(0, 0, 0, blockBit(0, 0, 0)), block(0, 0, 2, blockBit(0, 0, 2)));
}

async function compress_two_voxels_same_block(impl) {
    const v = await compressed(impl);
    const points = [pf(0.5, 0.5, 0.5, 1.0), pf(1.5, 1.5, 1.5, 1.0)];
    // One block, both bits, reported once.
    await expectVoxels(v.add_points(pf(0.5, 0.5, 1.5, 1.0), points), pi(0, 0, 0, 1 | 1 << 21));
}

async function compress_low_half_change_with_high_bits_present(impl) {
    const v = await compressed(impl);
    await expectVoxels(fromAbove(v, 0, 0, 3, 1.0), pi(0, 0, 2, 1 << 16));
    // A low half change must report the low half, not the (non-zero) high one.
    await expectVoxels(fromAbove(v, 1, 0, 0, 1.0), pi(0, 0, 0, 1 << 1));
}

async function compress_page_keys(impl) {
    const v = await compressed(impl);
    // Fine page (1,0,0) of compressed page (0,0,0).
    await expectVoxels(alongX(v, 3, 4, 0, 0, 1.0), pi(4, 0, 0, 1));
    // Fine page (16,0,0): first slot of compressed page (1,0,0).
    await expectVoxels(alongX(v, 63, 64, 0, 0, 1.0), pi(64, 0, 0, 1));
    // Fine page (-1,-1,-1): last slot of compressed page (-1,-1,-1).
    await expectVoxels(fromAbove(v, -1, -1, -1, 1.0), pi(-4, -4, -2, 2 ** 31));
}

async function compress_block_coords_far(impl) {
    const v = await compressed(impl);
    await expectVoxels(fromAbove(v, 5, 9, 7, 1.0), pi(4, 8, 6, 1 << 21));
    await expectVoxels(fromAbove(v, -5, -9, -6, 1.0), pi(-8, -12, -6, 1 << 15));
}

async function compress_mask_accumulates_across_calls(impl) {
    const v = await compressed(impl);
    await expectVoxels(fromAbove(v, 0, 0, 0, 1.0), pi(0, 0, 0, 1));
    // The pack is the block's full current state, not a delta.
    await expectVoxels(fromAbove(v, 1, 0, 0, 1.0), pi(0, 0, 0, 0x3));
    await expectVoxels(fromAbove(v, 2, 0, 1, 1.0), pi(0, 0, 0, 0x3 | 1 << 18));
}

// ------------------------------------------------------------ scenario

async function scenario_repeatable(impl) {
    for (const [destroy, construct, compress] of SCENARIO_PARAMS) {
        const a = await scenario(await make(impl, 1.0, 1, 1, destroy, construct, compress));
        a.forEach((update, call) => {
            check(update.length > 0, `compress = ${compress}: call ${call} changed nothing`);
        });
        const b = await scenario(await make(impl, 1.0, 1, 1, destroy, construct, compress));
        check(same(a, b), `compress = ${compress}: two runs differ`);
    }
}

// ---------------------------------------------------- global (run once)

async function single_and_wgpu_agree() {
    for (const [destroy, construct, compress] of SCENARIO_PARAMS) {
        const expected = await scenario(await make(SINGLE, 1.0, 1, 1, destroy, construct, compress));
        expected.forEach((update, call) => {
            check(update.length > 0, `compress = ${compress}: call ${call} changed nothing`);
        });
        const got = await scenario(await make(WGPU, 1.0, 1, 1, destroy, construct, compress));
        check(same(got, expected), `compress = ${compress}: differs`);
    }
}

async function wgpu_create_is_async() {
    const created = rv.WgpuAccelerated.create(1.0, 1, 1, 0, 1, false, WGPU_TEST_BUCKETS);
    check(created instanceof Promise, "WgpuAccelerated.create did not return a Promise");
    const v = await created;
    check(v instanceof rv.WgpuAccelerated, `created a ${v?.constructor?.name}`);
    await expectVoxels(addPoint(v, sensor(), pf(3.5, 0.5, 0.5, 1.0)), pi(3, 0, 0, 1));
}

async function wgpu_rejects_zero_buckets() {
    await throwsError(() => rv.WgpuAccelerated.create(1.0, 1, 1, 0, 1, false, 0),
        "WgpuAccelerated.create with 0 buckets");
}

function wgpu_memory_size_grows() {
    const sizes = [1, 2, WGPU_TEST_BUCKETS].map((n) => rv.WgpuAccelerated.memory_size(n));
    check(sizes.every(Number.isInteger), `memory_size returned ${fmt(sizes)}, not integers`);
    check(0 < sizes[0] && sizes[0] < sizes[1] && sizes[1] < sizes[2],
        `memory sizes ${fmt(sizes)} do not grow`);
}

async function wgpu_overlapping_add_points_rejects() {
    // A call holds the voxelizer until it resolves, a second one meanwhile
    // rejects and changes nothing.
    const v = await plain(WGPU);
    const first = addPoint(v, sensor(), pf(3.5, 0.5, 0.5, 1.0));
    await throwsError(() => addPoint(v, sensor(), pf(3.5, 0.5, 0.5, 1.0)),
        "add_points while another call runs");
    await expectVoxels(first, pi(3, 0, 0, 1));
    await expectVoxels(addPoint(v, sensor(), pf(3.5, 0.5, 0.5, 1.0)), pi(3, 0, 0, 2));
}

function compress_helpers_self_check() {
    check(blockBit(3, 3, 1) === 2 ** 31);
    check(blockBit(3, 3, 3) === 2 ** 31);
    check(same(block(3, 0, 0, blockBit(3, 0, 0)), pi(0, 0, 0, 1 << 3)));
    check(same(block(5, 9, 7, 0), pi(4, 8, 6, 0)));
    check(same(block(-1, -1, -1, blockBit(-1, -1, -1)), pi(-4, -4, -2, 2 ** 31)));
    check(same(block(-5, -9, -6, 0), pi(-8, -12, -6, 0)));
}

// --------------------------------------------------------------- table

function perImpl(fn) {
    return { name: fn.name, fn, perImpl: true, needs: [] };
}

/** A case run once overall, listed only when the features in needs are
 * enabled. */
function once(fn, ...needs) {
    return { name: fn.name, fn, perImpl: false, needs };
}

const TEST_CASES = [
    once(module_exports_match_features),
    once(point_f_constructor),
    once(point_f_stores_f32),
    once(point_f_properties_writable),
    once(point_i_constructor),
    once(point_i_properties_writable),
    once(point_i_setter_wraps),
    perImpl(constructors_accept_u32_max),
    perImpl(constructors_wrap_out_of_range),
    perImpl(add_points_result_types),
    perImpl(add_points_leaves_inputs_usable),
    perImpl(add_points_reuses_inputs),
    perImpl(add_points_accepts_repeated_point),
    perImpl(add_points_accepts_point_like_objects),
    perImpl(add_points_rejects_bad_sensor),
    perImpl(add_points_rejects_non_array_points),
    perImpl(add_points_rejects_bad_array_element),
    perImpl(instances_are_independent),
    perImpl(create),
    perImpl(create_free_cycles),
    perImpl(free_makes_object_unusable),
    perImpl(basic_1d),
    perImpl(empty_points_no_change),
    perImpl(returns_absolute_value_not_delta),
    perImpl(sensor_intensity_ignored),
    perImpl(intensity_truncated_toward_zero),
    perImpl(zero_intensity_reports_nothing),
    perImpl(resolution_scales_coords_not_intensity),
    perImpl(resolution_scales_sensor_too),
    perImpl(large_coords),
    perImpl(negative_coords_use_floor),
    perImpl(hit_multiplier_scales_hits),
    perImpl(target_multiplier_scales_target),
    perImpl(both_multipliers),
    perImpl(hit_saturates_at_zero),
    perImpl(hit_on_zero_voxel_not_reported),
    perImpl(thresholds_ignored_without_compress),
    perImpl(accumulates_across_calls),
    perImpl(unchanged_voxel_not_reported),
    perImpl(second_call_reports_only_its_changes),
    perImpl(multiple_points_one_call),
    perImpl(hits_skip_missing_pages),
    perImpl(negative_page),
    perImpl(axis_x_negative),
    perImpl(axis_y_positive),
    perImpl(axis_y_negative),
    perImpl(axis_z_positive),
    perImpl(axis_z_negative),
    perImpl(plane_xy_forward),
    perImpl(plane_xy_reverse),
    perImpl(plane_xz),
    perImpl(plane_yz),
    perImpl(full_3d_forward),
    perImpl(full_3d_reverse),
    perImpl(sensor_on_grid_line_2d),
    perImpl(ray_across_page_boundary),
    perImpl(point_in_sensor_voxel_fresh),
    perImpl(point_in_sensor_voxel_accumulates),
    perImpl(compress_construct_sets_bit),
    perImpl(compress_no_recross_no_report),
    perImpl(compress_plain_changes_not_returned),
    perImpl(compress_destroy_clears_bit),
    perImpl(compress_construct_threshold),
    perImpl(compress_destroy_threshold),
    perImpl(compress_bit_positions),
    perImpl(compress_both_halves_same_page),
    perImpl(compress_two_voxels_same_block),
    perImpl(compress_low_half_change_with_high_bits_present),
    perImpl(compress_page_keys),
    perImpl(compress_block_coords_far),
    perImpl(compress_mask_accumulates_across_calls),
    perImpl(scenario_repeatable),
    once(single_and_wgpu_agree, "single_threaded", "wgpu"),
    once(wgpu_create_is_async, "wgpu"),
    once(wgpu_rejects_zero_buckets, "wgpu"),
    once(wgpu_memory_size_grows, "wgpu"),
    once(wgpu_overlapping_add_points_rejects, "wgpu"),
    once(compress_helpers_self_check),
];

// Exits right away: a live WebGPU instance would keep Node.js running.
process.exit(await main(TEST_CASES));
