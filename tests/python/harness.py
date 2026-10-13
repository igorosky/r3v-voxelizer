"""
Minimal test harness for the Python bindings.

Every test case runs in its own interpreter process (see
tests/python_bindings.rs), so a failed check raises and ends the process.

usage: voxelizer_tests.py list | run <impl>::<test> | run <test>
"""

import contextlib
import os
import sys

import r3v_voxelizer as rv

# Hits saturate at zero, so a voxel must hold something before a hit on it
# becomes observable. Hit voxels are seeded with this value.
SEED = 10

# The voxelizer class each voxelizer feature adds to the module.
CLASS_OF_FEATURE = {"single_threaded": "SingleThreaded", "multi_threaded": "MultiThreaded",
                    "wgpu": "WgpuAccelerated"}

# Hashmap size of the wgpu voxelizers the tests create: 8192 pages, as in the
# wgpu unit tests (src/wgpu.rs).
WGPU_TEST_BUCKETS = 4096


def enabled_features():
    """Voxelizer features the module was built with, as passed in by
    tests/python_bindings.rs. When run by hand without it, whatever classes
    the module has are taken as enabled."""
    features = os.environ.get("R3V_VOXELIZER_FEATURES")
    if features is None:
        return {feature for feature, cls in CLASS_OF_FEATURE.items() if hasattr(rv, cls)}
    return set(filter(None, features.split(",")))


FEATURES = enabled_features()


class Impl:
    """One voxelizer type the per implementation tests run against: cls is the
    name of its class in the module, extra the (keyword, value) of the
    argument its constructor takes after the common ones, if any."""

    def __init__(self, name, cls, extra=None):
        self.name = name
        self.cls = cls
        self.extra = extra


def multi_threaded(name, threads):
    return Impl(name, "MultiThreaded", ("thread_count", threads))


SINGLE = Impl("single", "SingleThreaded")
WGPU = Impl("wgpu", "WgpuAccelerated", ("bucket_count", WGPU_TEST_BUCKETS))
IMPLS = ([SINGLE] if "single_threaded" in FEATURES else []) + (
    [multi_threaded("multi_1", 1), multi_threaded("multi_4", 4)]
    if "multi_threaded" in FEATURES else []) + ([WGPU] if "wgpu" in FEATURES else [])

# ---------------------------------------------------------------- checks


def fail(message):
    raise AssertionError(message)


def check(cond, message="check failed"):
    if not cond:
        fail(message)


@contextlib.contextmanager
def raises(exc_type):
    """Fails unless the block raises exc_type. Other exceptions propagate."""
    try:
        yield
    except exc_type:
        return
    fail(f"expected {exc_type.__name__}, nothing was raised")


def coords(point):
    """(x, y, z, i) of a PointF or PointI, read through its accessor methods."""
    return (point.x(), point.y(), point.z(), point.i())


def voxels(update, context=""):
    """Checks the types of what add_points returned and gives its points as
    sorted (x, y, z, i) tuples."""
    prefix = f"{context}: " if context else ""
    check(type(update) is list, f"{prefix}add_points returned {type(update).__name__}, not list")
    for point in update:
        check(type(point) is rv.PointI, f"{prefix}update holds {type(point).__name__}, not PointI")
    return sorted(coords(point) for point in update)


def check_update(update, expected, context=""):
    """Compares what add_points returned with (x, y, z, i) tuples, ignoring order."""
    got = voxels(update, context)
    expected = sorted(expected)
    if got != expected:
        prefix = f"{context}: " if context else ""
        fail(f"{prefix}update differs\n     got: {got}\nexpected: {expected}")


def expect_voxels(update, *expected):
    check_update(update, expected)


def expect_no_voxels(update):
    check_update(update, ())


# --------------------------------------------------------------- helpers


def pf(x, y, z, i):
    return rv.PointF(x, y, z, i)


def pi(x, y, z, i):
    """An expected PointI, as compared by check_update."""
    return (x, y, z, i)


def sensor():
    """The sensor most tests fire from: the middle of voxel (0, 0, 0)."""
    return pf(0.5, 0.5, 0.5, 1.0)


def make(impl, resolution, hit_multiplier, target_multiplier, destroy_threshold,
         construct_threshold, compress, max_value=0xFFFFFFFF):
    """A new voxelizer of the given type, without a cap on voxel values unless
    max_value is given."""
    args = (resolution, hit_multiplier, target_multiplier, destroy_threshold,
            construct_threshold, max_value, compress)
    if impl.extra is not None:
        args += (impl.extra[1],)
    return getattr(rv, impl.cls)(*args)


def plain(impl):
    """Resolution 1, multipliers 1, destroy at 0, construct at 1, no compression."""
    return make(impl, 1.0, 1, 1, 0, 1, False)


def compressed(impl):
    """Same as plain but with compression on."""
    return make(impl, 1.0, 1, 1, 0, 1, True)


def add_point(voxelizer, sensor_pos, point):
    """add_points for a single point."""
    return voxelizer.add_points(sensor_pos, [point])


def seed(voxelizer, resolution, x, y, z, i):
    """Adds i * target_multiplier to voxel (x, y, z) (creating its page) by
    firing a point that stays inside the sensor voxel, so nothing else gets
    hit. The update is dropped."""
    fy, fz = (y + 0.5) * resolution, (z + 0.5) * resolution
    add_point(voxelizer, pf((x + 0.5) * resolution, fy, fz, 1.0),
              pf((x + 0.75) * resolution, fy, fz, i))


def from_above(voxelizer, x, y, z, i):
    """Fires one ray of intensity i into voxel (x, y, z) from the voxel right
    above it, so the only voxel hit on the way is (x, y, z + 1)."""
    fx, fy, fz = x + 0.5, y + 0.5, z + 0.5
    return add_point(voxelizer, pf(fx, fy, fz + 1.0, 1.0), pf(fx, fy, fz, i))


def along_x(voxelizer, from_x, to_x, y, z, i):
    """Fires one ray of intensity i along x from voxel from_x to voxel to_x,
    hitting every voxel in between (including from_x)."""
    fy, fz = y + 0.5, z + 0.5
    return add_point(voxelizer, pf(from_x + 0.5, fy, fz, 1.0), pf(to_x + 0.5, fy, fz, i))


def check_ray(impl, sensor_pos, point, target, *hit):
    """Seeds every hit voxel with SEED, fires one ray of intensity 1 and checks
    that exactly those voxels dropped to SEED - 1 and that target went to 1."""
    v = plain(impl)
    for x, y, z in hit:
        seed(v, 1.0, x, y, z, SEED)
    expected = [pi(x, y, z, SEED - 1) for x, y, z in hit] + [pi(*target, 1)]
    check_update(add_point(v, sensor_pos, point), expected)


# ------------------------------------------------------ compress helpers
#
# The output of compress mode is one PointI per changed 4x4x2 block (half of
# a 4x4x4 fine page): x, y, z are the voxel origin of the block, i is its
# 32 bit occupancy pack.


def block_bit(x, y, z):
    """Bit of voxel (x, y, z) inside the 32 bit pack of its block."""
    return 1 << ((x & 3) + 4 * (y & 3) + 16 * (z & 1))


def block(x, y, z, mask):
    """Expected compress mode output for the block containing (x, y, z)."""
    return pi(x & ~3, y & ~3, z & ~1, mask)


# -------------------------------------------------------------- scenario
#
# A deterministic, schedule independent workload used to compare
# implementations and thread counts (port of `scenario` in src/test.rs).

# (destroy_threshold, construct_threshold, compress). With compression
# destroy = construct - 1, so a voxel above the destroy threshold has always
# been constructed.
SCENARIO_PARAMS = [(0, 1, False), (2, 3, True)]


class Lcg:
    """Tiny LCG, same as in src/test.rs."""

    def __init__(self, state):
        self.state = state

    def next(self):
        self.state = (self.state * 6364136223846793005 + 1442695040888963407) % 2**64
        return self.state >> 33


def cube_faces(dist, half, step):
    """Voxel coordinates on the six faces of the cube |coord| = dist, spaced
    step apart within -half..=half on the other two axes."""
    coords = []
    for a in range(-half, half + 1, step):
        for b in range(-half, half + 1, step):
            coords += [(dist, a, b), (-dist, a, b), (a, dist, b),
                       (a, -dist, b), (a, b, dist), (a, b, -dist)]
    return coords


def scenario(voxelizer):
    """Runs the scenario and returns the sorted output of every call."""
    rng = Lcg(0x5EED)
    snapshot = []
    for dist, half, step, max_i in [(50, 20, 4, 5), (100, 40, 8, 3)]:
        points = [pf(x + 0.5, y + 0.5, z + 0.5, 1 + rng.next() % max_i)
                  for x, y, z in cube_faces(dist, half, step)]
        snapshot.append(voxels(voxelizer.add_points(sensor(), points)))
    return snapshot


# ------------------------------------------------------------------ main


def main(test_cases):
    """test_cases: (name, fn, per_impl, needs) tuples. per_impl cases run once
    per entry of IMPLS and get it as their argument, the others run once.
    Cases needing a feature (from needs) that is not enabled are left out."""
    test_cases = [(name, fn, per_impl) for name, fn, per_impl, needs in test_cases
                  if FEATURES.issuperset(needs)]
    args = sys.argv[1:]
    if args == ["list"]:
        for name, _, per_impl in test_cases:
            if per_impl:
                for impl in IMPLS:
                    print(f"{impl.name}::{name}")
            else:
                print(name)
        return 0
    if len(args) == 2 and args[0] == "run":
        name, impl = args[1], None
        if "::" in name:
            impl_name, name = name.split("::", 1)
            impl = next((impl for impl in IMPLS if impl.name == impl_name), None)
            if impl is None:
                print(f"unknown voxelizer type in {args[1]}", file=sys.stderr)
                return 2
        for case_name, fn, per_impl in test_cases:
            if case_name == name and per_impl == (impl is not None):
                if per_impl:
                    fn(impl)
                else:
                    fn()
                return 0
        print(f"unknown test {args[1]}", file=sys.stderr)
        return 2
    print(f"usage: {sys.argv[0]} list | run <impl>::<test> | run <test>", file=sys.stderr)
    return 2
