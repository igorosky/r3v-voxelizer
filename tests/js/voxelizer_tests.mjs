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
    CLASS_OF_FEATURE, FEATURES, POINTS_IN_VOXEL_GROUP, SCENARIO_PARAMS, SEED, SINGLE,
    UNSUPPORTED_CLASSES, WGPU, WGPU_TEST_BUCKETS, addPoint, alongX, block, blockBit, check,
    checkConstructed, checkCoords, checkRay, checkUpdate, compressed, coords, decodeGroups, dump,
    expandBlocks, expectNoVoxels, expectVoxels, fail, fmt, fromAbove, main, make, pf, pi, plain,
    replay, rv, same, scenario, seed, sensor, throwsError, typeName,
} from "./harness.mjs";

const U32_MAX = 2 ** 32 - 1;

// -------------------------------------------------------------- JS API

/** Names of the classes the module must have with the enabled features. */
function enabledClasses() {
    return ["CompressedVoxelGroup", "Multipliers", "PointF", "PointI", ...Object.entries(CLASS_OF_FEATURE)
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

/** [intensity, hit, target] of a Multipliers, read through its properties. */
function multiplierValues(multipliers) {
    return [multipliers.intensity, multipliers.hit, multipliers.target];
}

/** Fails unless multipliers holds the expected values. */
function checkMultipliers(multipliers, expected, what) {
    const got = multiplierValues(multipliers);
    check(same(got, expected), `${what}: holds ${fmt(got)}, expected ${fmt(expected)}`);
}

function multipliers_constructor() {
    const multipliers = new rv.Multipliers(2.5, 2, 3);
    check(multipliers instanceof rv.Multipliers);
    checkMultipliers(multipliers, [2.5, 2, 3], "new Multipliers(2.5, 2, 3)");
    // The intensity is an f32.
    checkMultipliers(new rv.Multipliers(0.1, 0, U32_MAX), [Math.fround(0.1), 0, U32_MAX],
        `new Multipliers(0.1, 0, ${U32_MAX})`);
}

function multipliers_properties_writable() {
    const multipliers = new rv.Multipliers(1.0, 1, 1);
    multipliers.intensity = -1.5;
    multipliers.hit = U32_MAX;
    multipliers.target = 0;
    checkMultipliers(multipliers, [-1.5, U32_MAX, 0], "Multipliers after setting its properties");
}

function multipliers_setter_wraps() {
    // Out of range values wrap like any wasm-bindgen integer (see the README).
    const multipliers = new rv.Multipliers(1.0, 1, 1);
    multipliers.hit = -1;
    multipliers.target = 2 ** 32 + 3.9;
    checkMultipliers(multipliers, [1, U32_MAX, 3], "Multipliers after setting out of range values");
    checkMultipliers(new rv.Multipliers(1.0, -1, 1.9), [1, U32_MAX, 1],
        "new Multipliers(1, -1, 1.9)");
}

async function constructors_accept_u32_max(impl) {
    const v = await make(impl, 1.0, U32_MAX, U32_MAX, U32_MAX, U32_MAX, true, U32_MAX);
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
    await throwsError(() => a.get_all_points(), "get_all_points on a freed voxelizer");
    await throwsError(() => a.get_all_points_compressed(),
        "get_all_points_compressed on a freed voxelizer");
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

// ----------------------------------------------------------- max value

async function max_value_caps_target(impl) {
    const v = await make(impl, 1.0, 1, 1, 0, 1, false, 5);
    await expectVoxels(fromAbove(v, 3, 0, 0, 3.0), pi(3, 0, 0, 3));
    await expectVoxels(fromAbove(v, 3, 0, 0, 3.0), pi(3, 0, 0, 5));
    // Already at the cap: nothing changes, nothing is reported.
    await expectNoVoxels(fromAbove(v, 3, 0, 0, 3.0));
}

// ------------------------------------------------ intensity multiplier

async function intensity_multiplier_scales_hit_and_target(impl) {
    const v = await make(impl, 1.0, 2, 3, 0, 1, false, U32_MAX, 2.5);
    // Seeds (0,0,0) with (4 * 2.5) * 3 = 30.
    await seed(v, 1.0, 0, 0, 0, 4.0);
    // Math.trunc(3 * 2.5) = 7, times 2 subtracted and times 3 added.
    await expectVoxels(addPoint(v, sensor(), pf(3.5, 0.5, 0.5, 3.0)),
        pi(0, 0, 0, 30 - 7 * 2), pi(3, 0, 0, 7 * 3));
    // Scaled before truncation: Math.trunc(0.5 * 2.5) = 1.
    await expectVoxels(fromAbove(v, 5, 0, 0, 0.5), pi(5, 0, 0, 1 * 3));
}

// ------------------------------------------------------ set multipliers

async function get_multipliers_returns_constructor_values(impl) {
    const multipliers = (await plain(impl)).get_multipliers();
    check(multipliers instanceof rv.Multipliers, "get_multipliers did not return Multipliers");
    checkMultipliers(multipliers, [1, 1, 1], "get_multipliers of plain");
    const v = await make(impl, 1.0, 2, 3, 0, 1, false, U32_MAX, 2.5);
    checkMultipliers(v.get_multipliers(), [2.5, 2, 3], "get_multipliers after creating with them");
}

async function get_multipliers_returns_set_values(impl) {
    const v = await plain(impl);
    for (const values of [[2.5, 2, 3], [0, 0, 0], [-1.5, U32_MAX, U32_MAX], [1, 1, 1]]) {
        v.set_multipliers(new rv.Multipliers(...values));
        checkMultipliers(v.get_multipliers(), values, `get_multipliers after setting ${fmt(values)}`);
    }
    // The last of several sets wins, adding points leaves it as it is.
    v.set_multipliers(new rv.Multipliers(2.5, 2, 3));
    v.set_multipliers(new rv.Multipliers(1.5, 2, 3));
    await addPoint(v, sensor(), pf(3.5, 0.5, 0.5, 1.0));
    checkMultipliers(v.get_multipliers(), [1.5, 2, 3], "get_multipliers after add_points");
}

async function get_multipliers_returns_copy(impl) {
    const v = await plain(impl);
    const multipliers = new rv.Multipliers(2.5, 2, 3);
    v.set_multipliers(multipliers);
    // Changing the object passed in or the one returned leaves the voxelizer
    // as it was.
    multipliers.hit = 100;
    v.get_multipliers().target = 100;
    checkMultipliers(v.get_multipliers(), [2.5, 2, 3], "get_multipliers after changing copies");
    checkMultipliers(multipliers, [2.5, 100, 3], "Multipliers passed to set_multipliers");
}

async function set_multipliers_applies_to_next_call(impl) {
    const v = await plain(impl);
    await seed(v, 1.0, 0, 0, 0, 20.0);
    await expectVoxels(addPoint(v, sensor(), pf(3.5, 0.5, 0.5, 3.0)),
        pi(0, 0, 0, 17), pi(3, 0, 0, 3));
    v.set_multipliers(new rv.Multipliers(2.5, 2, 3));
    // Math.trunc(3 * 2.5) = 7, times 2 subtracted from and times 3 added to
    // what the voxels hold already.
    await expectVoxels(addPoint(v, sensor(), pf(3.5, 0.5, 0.5, 3.0)),
        pi(0, 0, 0, 17 - 7 * 2), pi(3, 0, 0, 3 + 7 * 3));
    // Setting the old ones back restores the old behaviour.
    v.set_multipliers(new rv.Multipliers(1.0, 1, 1));
    await expectVoxels(addPoint(v, sensor(), pf(3.5, 0.5, 0.5, 3.0)),
        pi(0, 0, 0, 0), pi(3, 0, 0, 24 + 3));
}

async function set_multipliers_leaves_input_usable(impl) {
    // The caller keeps owning the Multipliers it passes, so it can be passed
    // again.
    const a = await plain(impl);
    const b = await plain(impl);
    const multipliers = new rv.Multipliers(2.5, 2, 3);
    a.set_multipliers(multipliers);
    b.set_multipliers(multipliers);
    checkMultipliers(multipliers, [2.5, 2, 3], "Multipliers after set_multipliers");
    checkMultipliers(b.get_multipliers(), [2.5, 2, 3], "get_multipliers of the second voxelizer");
    multipliers.free();
    checkMultipliers(a.get_multipliers(), [2.5, 2, 3], "get_multipliers after freeing the input");
}

async function set_multipliers_rejects_bad_argument(impl) {
    const v = await plain(impl);
    const freed = new rv.Multipliers(2.5, 2, 3);
    freed.free();
    const bad = [
        ["null", null],
        ["undefined", undefined],
        ["a plain object", { intensity: 2.5, hit: 2, target: 3 }],
        ["a PointF", pf(2.5, 2.0, 3.0, 0.0)],
        ["a freed Multipliers", freed],
    ];
    for (const [what, multipliers] of bad) {
        await throwsError(() => v.set_multipliers(multipliers), `set_multipliers with ${what}`);
    }
    checkMultipliers(v.get_multipliers(), [1, 1, 1], "get_multipliers after the rejected calls");
    await expectUntouched(v);
}

async function set_multipliers_same_as_constructed(impl) {
    for (const [destroy, construct, compress] of SCENARIO_PARAMS) {
        for (const [intensity, hit, target] of [[1.5, 2, 3], [0.5, 1, 4]]) {
            const v = await make(impl, 1.0, 1, 1, destroy, construct, compress);
            v.set_multipliers(new rv.Multipliers(intensity, hit, target));
            const a = await scenario(v);
            const b = await scenario(await make(impl, 1.0, hit, target, destroy, construct,
                compress, U32_MAX, intensity));
            check(same(a, b), `compress = ${compress}, set to ${fmt([intensity, hit, target])}: `
                + "runs differ");
        }
    }
}

// --------------------------------------------------------------- dumps

async function get_all_points_result_types(impl) {
    const v = await plain(impl);
    await fromAbove(v, 5, 6, 7, 3.0);
    const pointsResult = v.get_all_points();
    check(pointsResult instanceof Promise === impl.isAsync,
        `get_all_points returned ${impl.isAsync ? "no" : "a"} Promise`);
    const points = await pointsResult;
    check(Array.isArray(points) && points.length === 1,
        `get_all_points returned ${typeName(points)} of ${points?.length}`);
    check(points[0] instanceof rv.PointI, "get_all_points did not return PointI objects");
    const groupsResult = v.get_all_points_compressed();
    check(groupsResult instanceof Promise === impl.isAsync,
        `get_all_points_compressed returned ${impl.isAsync ? "no" : "a"} Promise`);
    const groups = await groupsResult;
    check(Array.isArray(groups) && groups.length === 1,
        `get_all_points_compressed returned ${typeName(groups)} of ${groups?.length}`);
    const [group] = groups;
    check(group instanceof rv.CompressedVoxelGroup,
        "get_all_points_compressed did not return CompressedVoxelGroup objects");
    check(group.point instanceof rv.PointI, `group key is ${typeName(group.point)}`);
    check(group.values instanceof BigUint64Array && group.values.length === POINTS_IN_VOXEL_GROUP,
        `group words are ${typeName(group.values)} of ${group.values?.length}`);
    // The results are owned by JS and outlive the voxelizer.
    v.free();
    checkCoords(points[0], [5, 6, 7, 3], "PointI after freeing its voxelizer");
    checkCoords(group.point, [0, 0, 0, 0], "group key after freeing its voxelizer");
    check(same(decodeGroups(groups), [pi(5, 6, 7, 1)]), "group after freeing its voxelizer");
}

async function compressed_voxel_group_returns_copies(impl) {
    const v = await compressed(impl);
    await fromAbove(v, 5, 6, 7, 1.0);
    const [group] = await v.get_all_points_compressed();
    // Both properties give copies, changing them leaves the group as it was.
    group.point.x = 9;
    group.values[7] = 0n;
    checkCoords(group.point, [0, 0, 0, 0], "group key after changing a copy");
    check(group.values[7] === 1n << 57n, "group words changed through a copy");
    await throwsError(() => {
        group.point = new rv.PointI(1, 2, 3, 0);
    }, "setting the read only point");
    checkCoords(group.point, [0, 0, 0, 0], "group key after setting it");
    group.free();
    await throwsError(() => group.values, "reading the words of a freed group");
}

async function get_all_points_empty(impl) {
    await expectNoVoxels((await plain(impl)).get_all_points());
    await expectNoVoxels((await compressed(impl)).get_all_points());
}

async function get_all_points_single_target(impl) {
    const v = await plain(impl);
    await fromAbove(v, 5, 6, 7, 3.0);
    await expectVoxels(v.get_all_points(), pi(5, 6, 7, 3));
}

async function get_all_points_zero_intensity_target(impl) {
    const v = await plain(impl);
    await expectNoVoxels(fromAbove(v, 5, 6, 7, 0.0));
    await expectNoVoxels(v.get_all_points());
}

async function get_all_points_emptied_page_gone(impl) {
    const v = await plain(impl);
    await fromAbove(v, 0, 0, 0, 2.0);
    // Drops (0, 0, 0) to 0, which empties its page.
    await alongX(v, 0, 5, 0, 0, 2.0);
    await expectVoxels(v.get_all_points(), pi(5, 0, 0, 2));
}

async function get_all_points_negative_and_large_coords(impl) {
    const voxels = [pi(-1, -1, -1, 1), pi(-4, 3, -5, 2), pi(3, 4, -4, 3), pi(63, -64, 64, 4),
        pi(-70000, 70000, 0, 5), pi(1000000, -1000000, -1, 6)];
    const v = await plain(impl);
    for (const [x, y, z, i] of voxels) {
        await seed(v, 1.0, x, y, z, i);
    }
    await checkUpdate(v.get_all_points(), voxels);
}

async function get_all_points_with_compress_returns_values(impl) {
    const v = await make(impl, 1.0, 1, 1, 2, 3, true);
    // 2 stays between the thresholds, so only (-5, 6, 7) is constructed.
    await seed(v, 1.0, 5, 6, 7, 2.0);
    await seed(v, 1.0, -5, 6, 7, 4.0);
    await expectVoxels(v.get_all_points(), pi(5, 6, 7, 2), pi(-5, 6, 7, 4));
    await checkConstructed(v, pi(-5, 6, 7, 1));
}

async function get_all_points_does_not_change_state(impl) {
    for (const [destroy, construct, compress] of SCENARIO_PARAMS) {
        const v = await make(impl, 1.0, 1, 1, destroy, construct, compress);
        const untouched = await make(impl, 1.0, 1, 1, destroy, construct, compress);
        const first = await scenario(v);
        await scenario(untouched, false);
        check(first.dump.points.length > 0 && first.dump.constructed.length > 0,
            `compress = ${compress}: empty dumps`);
        check(same(await dump(v), first.dump), `compress = ${compress}: dumps differ`);
        // The following calls change the same as without the dumps.
        check(same(await scenario(v, false), await scenario(untouched, false)),
            `compress = ${compress}: dumping changed later calls`);
    }
}

async function get_all_points_compressed_empty(impl) {
    for (const v of [await plain(impl), await compressed(impl)]) {
        const groups = await v.get_all_points_compressed();
        check(Array.isArray(groups) && groups.length === 0, `dumped ${groups?.length} groups`);
    }
}

async function get_all_points_compressed_single_voxel(impl) {
    const v = await compressed(impl);
    await fromAbove(v, 5, 6, 7, 1.0);
    const groups = await v.get_all_points_compressed();
    check(groups.length === 1, `${groups.length} groups`);
    checkCoords(groups[0].point, [0, 0, 0, 0], "group key");
    // Page (1, 1, 1) is word 0b111 of the group, voxel (1, 2, 3) of it is
    // bit 1 + 4 * 2 + 16 * 3.
    const words = [];
    groups[0].values.forEach((word, k) => {
        if (word !== 0n) {
            words.push([k, word]);
        }
    });
    check(words.length === 1 && words[0][0] === 0b111 && words[0][1] === 1n << 57n,
        `words ${words.map(([k, word]) => `${k}: ${word}`)}`);
}

async function get_all_points_compressed_group_keys(impl) {
    // A group spans 16 fine pages, 64 voxels, along every axis.
    for (const [voxel, key] of [[[-1, -1, -1], [-1, -1, -1]], [[64, 0, 0], [1, 0, 0]],
        [[63, 63, 63], [0, 0, 0]], [[-65, 0, 0], [-2, 0, 0]], [[0, -64, 128], [0, -1, 2]]]) {
        const v = await compressed(impl);
        await fromAbove(v, ...voxel, 1.0);
        const groups = await v.get_all_points_compressed();
        const keys = groups.map((group) => coords(group.point));
        check(same(keys, [[...key, 0]]), `voxel ${fmt(voxel)}: keys ${fmt(keys)}`);
        check(same(decodeGroups(groups), [pi(...voxel, 1)]), `voxel ${fmt(voxel)}`);
    }
}

async function get_all_points_compressed_destroyed_group_removed(impl) {
    const v = await compressed(impl);
    await fromAbove(v, 5, 6, 7, 1.0);
    // Destroys (5, 6, 7) and constructs (70, 6, 7) in the next group.
    await alongX(v, 5, 70, 6, 7, 1.0);
    const groups = await v.get_all_points_compressed();
    check(same(groups.map((group) => coords(group.point)), [[1, 0, 0, 0]]), "group keys");
    check(same(decodeGroups(groups), [pi(70, 6, 7, 1)]), "constructed voxels");
}

async function get_all_points_compressed_plain_threshold(impl) {
    // Without compress a voxel counts as constructed once it reaches
    // construct_threshold.
    const voxels = [[1, 2, 3, 2.0], [5, 6, 7, 3.0], [70, 0, 0, 4.0]];
    let v = await make(impl, 1.0, 1, 1, 0, 3, false);
    for (const voxel of voxels) {
        await seed(v, 1.0, ...voxel);
    }
    await checkConstructed(v, pi(5, 6, 7, 1), pi(70, 0, 0, 1));
    // A threshold of 0 still leaves out voxels that are 0.
    v = await make(impl, 1.0, 1, 1, 0, 0, false);
    for (const voxel of voxels) {
        await seed(v, 1.0, ...voxel);
    }
    await checkConstructed(v, pi(1, 2, 3, 1), pi(5, 6, 7, 1), pi(70, 0, 0, 1));
}

async function get_all_points_compressed_plain_matches_compress_mode(impl) {
    // With thresholds 0 and 1 no voxel is between them, so both modes agree
    // on what is constructed.
    const groups = async (compress) => {
        const v = await make(impl, 1.0, 1, 1, 0, 1, compress);
        await scenario(v, false);
        return (await v.get_all_points_compressed())
            .map((group) => [coords(group.point), group.values.join()])
            .sort((a, b) => (a[0].join() < b[0].join() ? -1 : 1));
    };
    const plainGroups = await groups(false);
    check(plainGroups.length > 1, `${plainGroups.length} groups`);
    check(same(plainGroups, await groups(true)), "the modes dump different groups");
}

async function dump_matches_replayed_results(impl) {
    for (const [destroy, construct, compress] of SCENARIO_PARAMS) {
        const { results, dump: dumped } =
            await scenario(await make(impl, 1.0, 1, 1, destroy, construct, compress));
        const replayed = replay(results);
        check(replayed.length > 0, `compress = ${compress}: nothing to replay`);
        if (compress) {
            // The results are blocks, the dump has every constructed voxel of
            // them.
            check(same(dumped.constructed, expandBlocks(replayed)), "compress = true: differs");
        } else {
            check(same(dumped.points, replayed), "compress = false: get_all_points differs");
            // Construct threshold 1: every voxel that is not 0 is
            // constructed.
            check(same(dumped.constructed, replayed.map(([x, y, z]) => pi(x, y, z, 1))),
                "compress = false: get_all_points_compressed differs");
        }
    }
}

// ------------------------------------------------------------ scenario

async function scenario_repeatable(impl) {
    for (const [destroy, construct, compress] of SCENARIO_PARAMS) {
        const a = await scenario(await make(impl, 1.0, 1, 1, destroy, construct, compress));
        a.results.forEach((update, call) => {
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
        expected.results.forEach((update, call) => {
            check(update.length > 0, `compress = ${compress}: call ${call} changed nothing`);
        });
        const got = await scenario(await make(WGPU, 1.0, 1, 1, destroy, construct, compress));
        check(same(got, expected), `compress = ${compress}: differs`);
    }
}

async function wgpu_create_is_async() {
    const created = rv.WgpuAccelerated.create(1.0, 1.0, 1, 1, 0, 1, U32_MAX, false,
        WGPU_TEST_BUCKETS);
    check(created instanceof Promise, "WgpuAccelerated.create did not return a Promise");
    const v = await created;
    check(v instanceof rv.WgpuAccelerated, `created a ${v?.constructor?.name}`);
    await expectVoxels(addPoint(v, sensor(), pf(3.5, 0.5, 0.5, 1.0)), pi(3, 0, 0, 1));
}

async function wgpu_rejects_zero_buckets() {
    await throwsError(() => rv.WgpuAccelerated.create(1.0, 1.0, 1, 1, 0, 1, U32_MAX, false, 0),
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

async function wgpu_multipliers_while_running_rejects() {
    // A call holds the voxelizer until it resolves, setting or reading the
    // multipliers meanwhile throws and changes nothing.
    const v = await make(WGPU, 1.0, 2, 3, 0, 1, false, U32_MAX, 2.5);
    const first = addPoint(v, sensor(), pf(3.5, 0.5, 0.5, 1.0));
    // Async calls start from a microtask, this lets the first one take the
    // voxelizer.
    await null;
    const multipliers = new rv.Multipliers(1.0, 1, 1);
    await throwsError(() => v.set_multipliers(multipliers), "set_multipliers while add_points runs");
    await throwsError(() => v.get_multipliers(), "get_multipliers while add_points runs");
    // Math.trunc(1 * 2.5) = 2, times 3.
    await expectVoxels(first, pi(3, 0, 0, 6));
    checkMultipliers(v.get_multipliers(), [2.5, 2, 3], "get_multipliers after the call");
    v.set_multipliers(multipliers);
    await expectVoxels(addPoint(v, sensor(), pf(3.5, 0.5, 0.5, 1.0)), pi(3, 0, 0, 7));
}

async function wgpu_dump_while_running_rejects() {
    // A call holds the voxelizer until it resolves: dumping while points are
    // added rejects, as does any other call while a dump runs, and none of
    // them changes anything.
    const v = await plain(WGPU);
    const adding = addPoint(v, sensor(), pf(3.5, 0.5, 0.5, 1.0));
    await throwsError(() => v.get_all_points(), "get_all_points while add_points runs");
    await throwsError(() => v.get_all_points_compressed(),
        "get_all_points_compressed while add_points runs");
    await expectVoxels(adding, pi(3, 0, 0, 1));
    const dumping = v.get_all_points();
    // Async calls start from a microtask, this lets the dump take the
    // voxelizer.
    await null;
    await throwsError(() => v.get_multipliers(), "get_multipliers while get_all_points runs");
    await throwsError(() => addPoint(v, sensor(), pf(3.5, 0.5, 0.5, 1.0)),
        "add_points while get_all_points runs");
    await throwsError(() => v.get_all_points_compressed(),
        "get_all_points_compressed while get_all_points runs");
    await expectVoxels(dumping, pi(3, 0, 0, 1));
    await expectVoxels(addPoint(v, sensor(), pf(3.5, 0.5, 0.5, 1.0)), pi(3, 0, 0, 2));
    await checkConstructed(v, pi(3, 0, 0, 1));
}

function compress_helpers_self_check() {
    check(blockBit(3, 3, 1) === 2 ** 31);
    check(blockBit(3, 3, 3) === 2 ** 31);
    check(same(block(3, 0, 0, blockBit(3, 0, 0)), pi(0, 0, 0, 1 << 3)));
    check(same(block(5, 9, 7, 0), pi(4, 8, 6, 0)));
    check(same(block(-1, -1, -1, blockBit(-1, -1, -1)), pi(-4, -4, -2, 2 ** 31)));
    check(same(block(-5, -9, -6, 0), pi(-8, -12, -6, 0)));
}

function dump_helpers_self_check() {
    // decodeGroups only reads point and values, as a CompressedVoxelGroup
    // cannot be built from JS this stands in for one.
    const values = new BigUint64Array(POINTS_IN_VOXEL_GROUP);
    // Word 7 = 0b111 is page (1, 1, 1), 41 = 0b101001 is page (3, 0, 2).
    values[7] = 1n << 57n;
    values[41] = 1n | 1n << 63n;
    const group = Object.create(rv.CompressedVoxelGroup.prototype, {
        point: { value: new rv.PointI(-1, 0, 2, 0) },
        values: { value: values },
    });
    check(same(decodeGroups([group]), [pi(-64 + 5, 6, 128 + 7, 1), pi(-64 + 12, 0, 128 + 8, 1),
        pi(-64 + 15, 3, 128 + 11, 1)]), "decodeGroups");
    const blocks = [block(-1, -1, -1, blockBit(-1, -1, -1)),
        block(5, 6, 7, blockBit(5, 6, 7) | blockBit(4, 4, 6))];
    check(same(expandBlocks(blocks), [pi(-1, -1, -1, 1), pi(4, 4, 6, 1), pi(5, 6, 7, 1)]),
        "expandBlocks");
    check(same(replay([[pi(1, 0, 0, 3), pi(2, 0, 0, 4)], [pi(1, 0, 0, 0), pi(3, 0, 0, 5)]]),
        [pi(2, 0, 0, 4), pi(3, 0, 0, 5)]), "replay");
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
    once(multipliers_constructor),
    once(multipliers_properties_writable),
    once(multipliers_setter_wraps),
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
    perImpl(max_value_caps_target),
    perImpl(intensity_multiplier_scales_hit_and_target),
    perImpl(get_multipliers_returns_constructor_values),
    perImpl(get_multipliers_returns_set_values),
    perImpl(get_multipliers_returns_copy),
    perImpl(set_multipliers_applies_to_next_call),
    perImpl(set_multipliers_leaves_input_usable),
    perImpl(set_multipliers_rejects_bad_argument),
    perImpl(set_multipliers_same_as_constructed),
    perImpl(get_all_points_result_types),
    perImpl(compressed_voxel_group_returns_copies),
    perImpl(get_all_points_empty),
    perImpl(get_all_points_single_target),
    perImpl(get_all_points_zero_intensity_target),
    perImpl(get_all_points_emptied_page_gone),
    perImpl(get_all_points_negative_and_large_coords),
    perImpl(get_all_points_with_compress_returns_values),
    perImpl(get_all_points_does_not_change_state),
    perImpl(get_all_points_compressed_empty),
    perImpl(get_all_points_compressed_single_voxel),
    perImpl(get_all_points_compressed_group_keys),
    perImpl(get_all_points_compressed_destroyed_group_removed),
    perImpl(get_all_points_compressed_plain_threshold),
    perImpl(get_all_points_compressed_plain_matches_compress_mode),
    perImpl(dump_matches_replayed_results),
    perImpl(scenario_repeatable),
    once(single_and_wgpu_agree, "single_threaded", "wgpu"),
    once(wgpu_create_is_async, "wgpu"),
    once(wgpu_rejects_zero_buckets, "wgpu"),
    once(wgpu_memory_size_grows, "wgpu"),
    once(wgpu_overlapping_add_points_rejects, "wgpu"),
    once(wgpu_multipliers_while_running_rejects, "wgpu"),
    once(wgpu_dump_while_running_rejects, "wgpu"),
    once(compress_helpers_self_check),
    once(dump_helpers_self_check),
];

// Exits right away: a live WebGPU instance would keep Node.js running.
process.exit(await main(TEST_CASES));
