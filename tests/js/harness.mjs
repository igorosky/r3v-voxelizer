/**
 * Minimal test harness for the JS bindings.
 *
 * Every test case runs in its own Node.js process (see tests/js_bindings.rs),
 * so a failed check throws and ends the process. Only Node.js built-ins are
 * used, except for the wgpu feature: Node.js has no WebGPU, so its tests take
 * it from the `webgpu` package (Dawn), installed into tests/js/node_modules
 * by `npm ci` in tests/js.
 *
 * WgpuAccelerated is created and adds points asynchronously, so the helpers
 * creating voxelizers or adding points may return Promises and every test
 * awaits them (a no-op for the synchronous classes).
 *
 * usage: voxelizer_tests.mjs list | run <impl>::<test> | run <test>
 */

import { readFileSync } from "node:fs";
import { join } from "node:path";
import process from "node:process";
import { fileURLToPath, pathToFileURL } from "node:url";

const MODULE = "r3v_voxelizer";

/** Directory of the wasm-pack output (`--target web`), as passed in by
 * tests/js_bindings.rs. When run by hand without it, the README's build
 * output (pkg/) is used. */
const PKG_DIR =
    process.env.R3V_VOXELIZER_PKG ?? fileURLToPath(new URL("../../pkg", import.meta.url));

export const rv = await import(pathToFileURL(join(PKG_DIR, `${MODULE}.js`)).href);
rv.initSync({ module: readFileSync(join(PKG_DIR, `${MODULE}_bg.wasm`)) });

// Hits saturate at zero, so a voxel must hold something before a hit on it
// becomes observable. Hit voxels are seeded with this value.
export const SEED = 10;

// The voxelizer class each voxelizer feature adds to the module.
export const CLASS_OF_FEATURE = { single_threaded: "SingleThreaded", wgpu: "WgpuAccelerated" };

// Hashmap size of the wgpu voxelizers the tests create: 8192 pages, as in the
// wgpu unit tests (src/wgpu.rs).
export const WGPU_TEST_BUCKETS = 4096;

// Voxelizer classes not supported in JS, the module must never have them.
export const UNSUPPORTED_CLASSES = ["MultiThreaded"];

/** Voxelizer features the module was built with, as passed in by
 * tests/js_bindings.rs. When run by hand without it, whatever classes the
 * module has are taken as enabled. */
function enabledFeatures() {
    const features = process.env.R3V_VOXELIZER_FEATURES;
    if (features === undefined) {
        return new Set(Object.keys(CLASS_OF_FEATURE).filter((f) => CLASS_OF_FEATURE[f] in rv));
    }
    return new Set(features.split(",").filter(Boolean));
}

export const FEATURES = enabledFeatures();

if (FEATURES.has("wgpu")) {
    const { create, globals } = await import("webgpu");
    Object.assign(globalThis, globals);
    Object.defineProperty(globalThis.navigator, "gpu", { value: create([]), configurable: true });
    // wgpu only looks for navigator.gpu when the global object is a Window
    // or a WorkerGlobalScope, which it finds by those constructors.
    globalThis.Window ??= class Window {};
}

/** One voxelizer type the per implementation tests run against: cls is the
 * name of its class in the module, isAsync tells whether it is created and
 * adds points asynchronously. */
export class Impl {
    constructor(name, cls, isAsync = false) {
        this.name = name;
        this.cls = cls;
        this.isAsync = isAsync;
    }
}

export const SINGLE = new Impl("single", "SingleThreaded");
export const WGPU = new Impl("wgpu", "WgpuAccelerated", true);
export const IMPLS = [
    ...(FEATURES.has("single_threaded") ? [SINGLE] : []),
    ...(FEATURES.has("wgpu") ? [WGPU] : []),
];

// ---------------------------------------------------------------- checks

export function fail(message) {
    throw new Error(message);
}

export function check(cond, message = "check failed") {
    if (!cond) {
        fail(message);
    }
}

export function fmt(value) {
    return JSON.stringify(value);
}

export function same(a, b) {
    return fmt(a) === fmt(b);
}

/** Fails unless fn throws an Error or returns a Promise rejecting with one.
 * A wasm trap (a Rust panic, after which the module is in an undefined state)
 * does not count. Resolves to the Error. */
export async function throwsError(fn, what) {
    try {
        await fn();
    } catch (e) {
        check(!(e instanceof WebAssembly.RuntimeError),
            `${what}: trapped (Rust panic) instead of throwing: ${e.message}`);
        check(e instanceof Error, `${what}: threw ${String(e)}, not an Error`);
        return e;
    }
    fail(`${what}: expected an Error, nothing was thrown`);
}

/** [x, y, z, i] of a PointF or PointI, read through its properties. */
export function coords(point) {
    return [point.x, point.y, point.z, point.i];
}

/** Fails unless point reads back as expected (exactly). */
export function checkCoords(point, expected, what) {
    let got;
    try {
        got = coords(point);
    } catch (e) {
        fail(`${what}: reading the point failed: ${e.message}`);
    }
    check(same(got, expected), `${what}: is ${fmt(got)}, expected ${fmt(expected)}`);
}

export function typeName(value) {
    return value === null || value === undefined ? String(value) : value.constructor?.name ?? typeof value;
}

function compareTuples(a, b) {
    for (let k = 0; k < a.length; k++) {
        if (a[k] !== b[k]) {
            return a[k] - b[k];
        }
    }
    return 0;
}

/** Checks the types of what add_points (or get_all_points) returned (once
 * resolved) and gives its points as sorted [x, y, z, i] arrays. */
export function voxels(update, context = "") {
    const prefix = context ? `${context}: ` : "";
    check(Array.isArray(update), `${prefix}returned ${typeName(update)}, not an Array`);
    for (const point of update) {
        check(point instanceof rv.PointI, `${prefix}update holds ${typeName(point)}, not PointI`);
    }
    return update.map(coords).sort(compareTuples);
}

/** Compares what add_points (or get_all_points) returned (or the Promise of
 * it) with [x, y, z, i] arrays, ignoring order. */
export async function checkUpdate(update, expected, context = "") {
    const got = voxels(await update, context);
    const want = [...expected].sort(compareTuples);
    if (!same(got, want)) {
        const prefix = context ? `${context}: ` : "";
        fail(`${prefix}update differs\n     got: ${fmt(got)}\nexpected: ${fmt(want)}`);
    }
}

export async function expectVoxels(update, ...expected) {
    await checkUpdate(update, expected);
}

export async function expectNoVoxels(update) {
    await checkUpdate(update, []);
}

// --------------------------------------------------------------- helpers

/** A new PointF. Fails right here, with one clear reason, when the bindings
 * cannot build a point from JS. */
export function pf(x, y, z, i) {
    const what = `new PointF(${x}, ${y}, ${z}, ${i}) does not build a usable point`;
    let point;
    try {
        point = new rv.PointF(x, y, z, i);
    } catch (e) {
        fail(`${what}: ${e.message}`);
    }
    checkCoords(point, [x, y, z, i].map(Math.fround), what);
    return point;
}

/** An expected PointI, as compared by checkUpdate. */
export function pi(x, y, z, i) {
    return [x, y, z, i];
}

/** The sensor most tests fire from: the middle of voxel (0, 0, 0). A new
 * PointF on every call. */
export function sensor() {
    return pf(0.5, 0.5, 0.5, 1.0);
}

/** Resolves to a new voxelizer of the given type, without a cap on voxel
 * values unless maxValue is given and without scaling of intensities unless
 * intensityMultiplier is given. */
export async function make(impl, resolution, hitMultiplier, targetMultiplier, destroyThreshold,
    constructThreshold, compress, maxValue = 0xFFFFFFFF, intensityMultiplier = 1.0) {
    const args = [resolution, intensityMultiplier, hitMultiplier, targetMultiplier,
        destroyThreshold, constructThreshold, maxValue, compress];
    if (impl.cls === "WgpuAccelerated") {
        return await rv.WgpuAccelerated.create(...args, WGPU_TEST_BUCKETS);
    }
    return new rv[impl.cls](...args);
}

/** Resolution 1, multipliers 1, destroy at 0, construct at 1, no compression. */
export function plain(impl) {
    return make(impl, 1.0, 1, 1, 0, 1, false);
}

/** Same as plain but with compression on. */
export function compressed(impl) {
    return make(impl, 1.0, 1, 1, 0, 1, true);
}

/** add_points for a single point. */
export function addPoint(voxelizer, sensorPos, point) {
    return voxelizer.add_points(sensorPos, [point]);
}

/** Adds i * target_multiplier to voxel (x, y, z) (creating its page) by
 * firing a point that stays inside the sensor voxel, so nothing else gets
 * hit. The update is dropped. */
export async function seed(voxelizer, resolution, x, y, z, i) {
    const fy = (y + 0.5) * resolution;
    const fz = (z + 0.5) * resolution;
    await addPoint(voxelizer, pf((x + 0.5) * resolution, fy, fz, 1.0),
        pf((x + 0.75) * resolution, fy, fz, i));
}

/** Fires one ray of intensity i into voxel (x, y, z) from the voxel right
 * above it, so the only voxel hit on the way is (x, y, z + 1). */
export function fromAbove(voxelizer, x, y, z, i) {
    const [fx, fy, fz] = [x + 0.5, y + 0.5, z + 0.5];
    return addPoint(voxelizer, pf(fx, fy, fz + 1.0, 1.0), pf(fx, fy, fz, i));
}

/** Fires one ray of intensity i along x from voxel fromX to voxel toX,
 * hitting every voxel in between (including fromX). */
export function alongX(voxelizer, fromX, toX, y, z, i) {
    const [fy, fz] = [y + 0.5, z + 0.5];
    return addPoint(voxelizer, pf(fromX + 0.5, fy, fz, 1.0), pf(toX + 0.5, fy, fz, i));
}

/** Seeds every hit voxel with SEED, fires one ray of intensity 1 and checks
 * that exactly those voxels dropped to SEED - 1 and that target went to 1. */
export async function checkRay(impl, sensorPos, point, target, ...hit) {
    const v = await plain(impl);
    for (const [x, y, z] of hit) {
        await seed(v, 1.0, x, y, z, SEED);
    }
    const expected = [...hit.map(([x, y, z]) => pi(x, y, z, SEED - 1)), pi(...target, 1)];
    await checkUpdate(addPoint(v, sensorPos, point), expected);
}

// ------------------------------------------------------ compress helpers
//
// The output of compress mode is one PointI per changed 4x4x2 block (half of
// a 4x4x4 fine page): x, y, z are the voxel origin of the block, i is its
// 32 bit occupancy pack (unsigned, so bit 31 reads as 2 ** 31).

/** Bit of voxel (x, y, z) inside the 32 bit pack of its block. */
export function blockBit(x, y, z) {
    return (1 << ((x & 3) + 4 * (y & 3) + 16 * (z & 1))) >>> 0;
}

/** Expected compress mode output for the block containing (x, y, z). */
export function block(x, y, z, mask) {
    return pi(x & ~3, y & ~3, z & ~1, mask);
}

/** Every voxel set in the 32 bit packs of blocks, as sorted [x, y, z, 1]
 * arrays. */
export function expandBlocks(blocks) {
    const points = [];
    for (const [x, y, z, mask] of blocks) {
        for (let bit = 0; bit < 32; bit++) {
            if ((mask >>> bit) & 1) {
                points.push(pi(x + (bit & 3), y + ((bit >> 2) & 3), z + (bit >> 4), 1));
            }
        }
    }
    return points.sort(compareTuples);
}

// -------------------------------------------------------- dump helpers
//
// get_all_points_compressed returns one CompressedVoxelGroup per 64x64x64
// voxel cube holding a constructed voxel: a word per 4x4x4 fine page in
// z-order, bit x + 4 * y + 16 * z of it set when voxel (x, y, z) of the page
// is constructed.

// ALL_POINTS_GROUP_EDGE_BITS of the crate.
const GROUP_EDGE_BITS = 4;
export const POINTS_IN_VOXEL_GROUP = 2 ** (3 * GROUP_EDGE_BITS);

/** Checks the types of what get_all_points_compressed returned (once
 * resolved) and gives every voxel set in it as sorted [x, y, z, 1] arrays
 * (port of `decode_groups` in src/test.rs). */
export function decodeGroups(groups) {
    check(Array.isArray(groups),
        `get_all_points_compressed returned ${typeName(groups)}, not an Array`);
    const pages = 2 ** GROUP_EDGE_BITS;
    const points = [];
    for (const group of groups) {
        check(group instanceof rv.CompressedVoxelGroup,
            `get_all_points_compressed holds ${typeName(group)}, not CompressedVoxelGroup`);
        const [x, y, z, i] = coords(group.point);
        check(i === 0, `group key ${fmt([x, y, z, i])} has i != 0`);
        const values = group.values;
        check(values instanceof BigUint64Array && values.length === POINTS_IN_VOXEL_GROUP,
            `group words are ${typeName(values)} of ${values?.length}`);
        values.forEach((word, k) => {
            if (word === 0n) {
                return;
            }
            // Bits of x, y and z of the page offset interleave from the
            // lowest one.
            const offset = [0, 0, 0];
            for (let b = 0; b < 3 * GROUP_EDGE_BITS; b++) {
                offset[b % 3] |= ((k >> b) & 1) << Math.floor(b / 3);
            }
            // Multiplied, shifting would wrap around at 32 bits.
            const page = [x * pages + offset[0], y * pages + offset[1], z * pages + offset[2]];
            for (let bit = 0; bit < 64; bit++) {
                if ((word >> BigInt(bit)) & 1n) {
                    points.push(pi(page[0] * 4 + (bit & 3), page[1] * 4 + ((bit >> 2) & 3),
                        page[2] * 4 + (bit >> 4), 1));
                }
            }
        });
    }
    return points.sort(compareTuples);
}

/** Compares the decoded compressed dump of voxelizer with [x, y, z, 1]
 * arrays, ignoring order. */
export async function checkConstructed(voxelizer, ...expected) {
    const got = decodeGroups(await voxelizer.get_all_points_compressed());
    const want = [...expected].sort(compareTuples);
    check(same(got, want), `constructed voxels differ\n     got: ${fmt(got)}\nexpected: ${fmt(want)}`);
}

/** Resolves to both dumps of voxelizer: get_all_points as sorted
 * [x, y, z, i] arrays and the voxels set in get_all_points_compressed (see
 * decodeGroups). */
export async function dump(voxelizer) {
    return {
        points: voxels(await voxelizer.get_all_points(), "get_all_points"),
        constructed: decodeGroups(await voxelizer.get_all_points_compressed()),
    };
}

// -------------------------------------------------------------- scenario
//
// A deterministic workload used to compare runs (port of `scenario` in
// src/test.rs).

// [destroyThreshold, constructThreshold, compress]. With compression
// destroy = construct - 1, so a voxel above the destroy threshold has always
// been constructed.
export const SCENARIO_PARAMS = [[0, 1, false], [2, 3, true]];

/** Tiny LCG, same as in src/test.rs. */
class Lcg {
    constructor(state) {
        this.state = BigInt(state);
    }

    next() {
        this.state = BigInt.asUintN(64, this.state * 6364136223846793005n + 1442695040888963407n);
        return Number(this.state >> 33n);
    }
}

/** Voxel coordinates on the six faces of the cube |coord| = dist, spaced
 * step apart within -half..=half on the other two axes. */
function cubeFaces(dist, half, step) {
    const coords = [];
    for (let a = -half; a <= half; a += step) {
        for (let b = -half; b <= half; b += step) {
            coords.push([dist, a, b], [-dist, a, b], [a, dist, b],
                [a, -dist, b], [a, b, dist], [a, b, -dist]);
        }
    }
    return coords;
}

/** Runs the scenario and resolves to {results, dump}: the sorted output of
 * every call and the dumps after the last call (null unless withDump). */
export async function scenario(voxelizer, withDump = true) {
    const rng = new Lcg(0x5EED);
    const results = [];
    for (const [dist, half, step, maxI] of [[50, 20, 4, 5], [100, 40, 8, 3]]) {
        const points = cubeFaces(dist, half, step)
            .map(([x, y, z]) => pf(x + 0.5, y + 0.5, z + 0.5, 1 + rng.next() % maxI));
        results.push(voxels(await voxelizer.add_points(sensor(), points)));
    }
    return { results, dump: withDump ? await dump(voxelizer) : null };
}

/** Last value every voxel (or block in compress mode) got in results,
 * without the zeroed ones, as sorted [x, y, z, i] arrays. */
export function replay(results) {
    const last = new Map();
    for (const result of results) {
        for (const [x, y, z, i] of result) {
            last.set(`${x},${y},${z}`, pi(x, y, z, i));
        }
    }
    return [...last.values()].filter(([, , , i]) => i !== 0).sort(compareTuples);
}

// ------------------------------------------------------------------ main

/** testCases: {name, fn, perImpl, needs} objects. perImpl cases run once
 * per entry of IMPLS and get it as their argument, the others run once.
 * Cases needing a feature (from needs) that is not enabled are left out.
 * Resolves to the exit code. */
export async function main(testCases) {
    testCases = testCases.filter(({ needs }) => needs.every((feature) => FEATURES.has(feature)));
    const args = process.argv.slice(2);
    if (args.length === 1 && args[0] === "list") {
        const names = [];
        for (const { name, perImpl } of testCases) {
            if (perImpl) {
                names.push(...IMPLS.map((impl) => `${impl.name}::${name}`));
            } else {
                names.push(name);
            }
        }
        process.stdout.write(names.map((name) => `${name}\n`).join(""));
        return 0;
    }
    if (args.length === 2 && args[0] === "run") {
        let name = args[1];
        let impl = null;
        const sep = name.indexOf("::");
        if (sep >= 0) {
            const implName = name.slice(0, sep);
            name = name.slice(sep + 2);
            impl = IMPLS.find((impl) => impl.name === implName) ?? null;
            if (impl === null) {
                console.error(`unknown voxelizer type in ${args[1]}`);
                return 2;
            }
        }
        for (const testCase of testCases) {
            if (testCase.name === name && testCase.perImpl === (impl !== null)) {
                if (testCase.perImpl) {
                    await testCase.fn(impl);
                } else {
                    await testCase.fn();
                }
                return 0;
            }
        }
        console.error(`unknown test ${args[1]}`);
        return 2;
    }
    console.error(`usage: ${process.argv[1]} list | run <impl>::<test> | run <test>`);
    return 2;
}
