/*
 * Minimal test harness for the C bindings.
 *
 * Every test case runs in its own process (see tests/c_bindings.rs), so a
 * failed check prints a message and exits right away.
 */
#ifndef VOXELIZER_C_TEST_HARNESS_H
#define VOXELIZER_C_TEST_HARNESS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "r3v-voxelizer.h"

#define ARRAY_LEN(a) (sizeof(a) / sizeof((a)[0]))

/* Hits saturate at zero, so a voxel must hold something before a hit on it
 * becomes observable. Hit voxels are seeded with this value. */
#define SEED 10u

/* Hashmap size of the wgpu voxelizers the tests create: 8192 pages, as in
 * the wgpu unit tests (src/wgpu.rs). */
#define WGPU_TEST_BUCKETS 4096u

typedef enum ImplKind
{
    IMPL_SINGLE,
    IMPL_MULTI,
    IMPL_WGPU,
} ImplKind;

/* One voxelizer type the per implementation tests run against. The wgpu one
 * is there when the tests are compiled with R3V_TEST_WGPU (the crate's wgpu
 * feature). */
typedef struct Impl
{
    const char *name;
    ImplKind kind;
    /* Thread count of IMPL_MULTI. */
    uintptr_t threads;
} Impl;

typedef void (*TestFn)(const Impl *impl);

typedef struct TestCase
{
    const char *name;
    TestFn fn;
    /* Run once per entry of IMPLS when true, once overall otherwise. */
    bool per_impl;
} TestCase;

extern const Impl IMPLS[];
extern const size_t IMPL_COUNT;

/* Defined by the test file. */
extern const TestCase TEST_CASES[];
extern const size_t TEST_CASE_COUNT;

/* ------------------------------------------------------------- checks */

void fail(const char *file, int line, const char *fmt, ...);

#define CHECK(cond)                                              \
    do                                                           \
    {                                                            \
        if (!(cond))                                             \
        {                                                        \
            fail(__FILE__, __LINE__, "check failed: %s", #cond); \
        }                                                        \
    } while (0)

/* Same order as the derived Ord of PointI: x, y, z, i. qsort comparator. */
int cmp_point_i(const void *lhs, const void *rhs);
void sort_points(PointI *points, size_t len);

/* Sorts both lists (got in place) and compares them. */
void check_points(PointI *got, size_t got_len, const PointI *expected, size_t expected_len,
                  const char *file, int line);

/* Sorts both lists, compares them and frees the update. */
void check_update(VoxelsUpdate update, const PointI *expected, size_t expected_len,
                  const char *file, int line);

#define VOXELS(...) ((PointI[]){__VA_ARGS__})
#define EXPECT_VOXELS(update, ...)                                              \
    check_update((update), VOXELS(__VA_ARGS__), ARRAY_LEN(VOXELS(__VA_ARGS__)), \
                 __FILE__, __LINE__)
#define EXPECT_NO_VOXELS(update) check_update((update), NULL, 0, __FILE__, __LINE__)

/* ------------------------------------------------------------- helpers */

PointF pf(float x, float y, float z, float i);
PointI pi(int32_t x, int32_t y, int32_t z, uint32_t i);
Multipliers mult(float intensity, uint32_t hit, uint32_t target);

/* Creates a voxelizer of the given type. Fails the test on NULL. */
void *make_full(const Impl *impl, float resolution, float intensity_multiplier,
                uint32_t hit_multiplier, uint32_t target_multiplier, uint32_t destroy_threshold,
                uint32_t construct_threshold, uint32_t max_value, bool compress);
/* make_full without scaling of intensities. */
void *make_capped(const Impl *impl, float resolution, uint32_t hit_multiplier,
                  uint32_t target_multiplier, uint32_t destroy_threshold,
                  uint32_t construct_threshold, uint32_t max_value, bool compress);
/* make_capped without a cap on voxel values. */
void *make(const Impl *impl, float resolution, uint32_t hit_multiplier,
           uint32_t target_multiplier, uint32_t destroy_threshold,
           uint32_t construct_threshold, bool compress);
/* Resolution 1, multipliers 1, destroy at 0, construct at 1, no compression. */
void *plain(const Impl *impl);
/* Same as plain but with compression on. */
void *compressed(const Impl *impl);

/* add_points for a single point. */
VoxelsUpdate add_point(void *voxelizer, PointF sensor, PointF point);

/* Adds i * target_multiplier to voxel (x, y, z) (creating its page) by
 * firing a point that stays inside the sensor voxel, so nothing else gets
 * hit. The update is dropped. */
void seed(void *voxelizer, float resolution, int32_t x, int32_t y, int32_t z, float i);

/* Fires one ray of intensity i into voxel (x, y, z) from the voxel right
 * above it, so the only voxel hit on the way is (x, y, z + 1). */
VoxelsUpdate from_above(void *voxelizer, int32_t x, int32_t y, int32_t z, float i);

/* Fires one ray of intensity i along x from voxel from_x to voxel to_x,
 * hitting every voxel in between (including from_x). */
VoxelsUpdate along_x(void *voxelizer, int32_t from_x, int32_t to_x, int32_t y, int32_t z,
                     float i);

typedef struct Coord
{
    int32_t x, y, z;
} Coord;

/* Seeds every hit voxel with SEED, fires one ray of intensity 1 and checks
 * that exactly those voxels dropped to SEED - 1 and that target went to 1. */
void check_ray_at(const Impl *impl, PointF sensor, PointF point, const Coord *hit,
                  size_t hit_len, Coord target, const char *file, int line);
#define CHECK_RAY(impl, sensor, point, target, ...)                   \
    check_ray_at((impl), (sensor), (point), ((Coord[]){__VA_ARGS__}), \
                 ARRAY_LEN(((Coord[]){__VA_ARGS__})), (target), __FILE__, __LINE__)

/* ----------------------------------------------------- compress helpers
 *
 * The output of compress mode is one PointI per changed 4x4x2 block (half of
 * a 4x4x4 fine page): x, y, z are the voxel origin of the block, i is its
 * 32 bit occupancy pack. */

/* Bit of voxel (x, y, z) inside the 32 bit pack of its block. */
uint32_t block_bit(int32_t x, int32_t y, int32_t z);
/* Expected compress mode output for the block containing (x, y, z). */
PointI block(int32_t x, int32_t y, int32_t z, uint32_t mask);
/* Every voxel set in the 32 bit packs of blocks, with i = 1, sorted. The
 * returned list is malloc'ed. */
PointI *expand_blocks(const PointI *blocks, size_t len, size_t *out_len);

/* ---------------------------------------------------------- dump helpers
 *
 * get_all_points_compressed returns one VoxelGroup per 64x64x64 voxel cube
 * holding a constructed voxel: a word per 4x4x4 fine page in z-order, bit
 * x + 4 * y + 16 * z of it set when voxel (x, y, z) of the page is
 * constructed. */

/* Every voxel set in groups, with i = 1, sorted (port of `decode_groups` in
 * src/test.rs). The returned list is malloc'ed. */
PointI *decode_groups(VoxelGroups groups, size_t *out_len);

/* Both dumps of a voxelizer, sorted. */
typedef struct Dump
{
    /* get_all_points. */
    PointI *points;
    size_t points_len;
    /* Decoded get_all_points_compressed. */
    PointI *constructed;
    size_t constructed_len;
} Dump;

Dump dump(const void *voxelizer);
void dump_free(Dump *dump);
void check_dump_eq(const Dump *got, const Dump *expected, const char *ctx, const char *file,
                   int line);

/* Decodes the compressed dump of voxelizer and compares it with expected
 * voxels (i = 1), ignoring order. */
void check_constructed(const void *voxelizer, const PointI *expected, size_t expected_len,
                       const char *file, int line);
#define EXPECT_CONSTRUCTED(voxelizer, ...)                                               \
    check_constructed((voxelizer), VOXELS(__VA_ARGS__), ARRAY_LEN(VOXELS(__VA_ARGS__)), \
                      __FILE__, __LINE__)
#define EXPECT_NONE_CONSTRUCTED(voxelizer) \
    check_constructed((voxelizer), NULL, 0, __FILE__, __LINE__)

/* ------------------------------------------------------------ scenario
 *
 * A deterministic, schedule independent workload used to compare
 * implementations and thread counts (port of `scenario` in src/test.rs). */

typedef struct ScenarioParams
{
    uint32_t destroy_threshold;
    uint32_t construct_threshold;
    bool compress;
} ScenarioParams;

extern const ScenarioParams SCENARIO_PARAMS[2];

#define SCENARIO_CALLS 2

typedef struct Snapshot
{
    /* Sorted output of every call. */
    PointI *results[SCENARIO_CALLS];
    size_t lens[SCENARIO_CALLS];
    /* Dumps after the last call, empty unless asked for. */
    Dump dump;
} Snapshot;

/* Runs the scenario on voxelizer, dumping it after the last call when
 * with_dump. */
Snapshot run_scenario(void *voxelizer, bool with_dump);
/* Runs the scenario with dumps on a fresh voxelizer and drops it. */
Snapshot scenario(void *voxelizer);
void snapshot_free(Snapshot *snapshot);
/* Last value every voxel (or block in compress mode) got in the results of
 * snapshot, without the zeroed ones, sorted. The returned list is
 * malloc'ed. */
PointI *replay(const Snapshot *snapshot, size_t *out_len);
void check_snapshot_eq(const Snapshot *got, const Snapshot *expected, const char *ctx,
                       const char *file, int line);

#endif /* VOXELIZER_C_TEST_HARNESS_H */
