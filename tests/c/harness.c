#include "harness.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const Impl IMPLS[] = {
    {"single", IMPL_SINGLE, 0},
    {"multi_1", IMPL_MULTI, 1},
    {"multi_4", IMPL_MULTI, 4},
#ifdef R3V_TEST_WGPU
    {"wgpu", IMPL_WGPU, 0},
#endif
};
const size_t IMPL_COUNT = ARRAY_LEN(IMPLS);

/* -------------------------------------------------------------- checks */

void fail(const char *file, int line, const char *fmt, ...)
{
    va_list args;
    fprintf(stderr, "%s:%d: ", file, line);
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fputc('\n', stderr);
    exit(1);
}

int cmp_point_i(const void *lhs, const void *rhs)
{
    const PointI *a = lhs;
    const PointI *b = rhs;
    if (a->x != b->x)
        return a->x < b->x ? -1 : 1;
    if (a->y != b->y)
        return a->y < b->y ? -1 : 1;
    if (a->z != b->z)
        return a->z < b->z ? -1 : 1;
    if (a->i != b->i)
        return a->i < b->i ? -1 : 1;
    return 0;
}

void sort_points(PointI *points, size_t len)
{
    if (len > 1)
    {
        qsort(points, len, sizeof(PointI), cmp_point_i);
    }
}

static void print_points(const char *label, const PointI *points, size_t len)
{
    fprintf(stderr, "  %s (%zu):", label, len);
    for (size_t k = 0; k < len; ++k)
    {
        fprintf(stderr, " (%d, %d, %d, %u)", points[k].x, points[k].y, points[k].z,
                points[k].i);
    }
    fputc('\n', stderr);
}

static bool points_eq(const PointI *a, size_t a_len, const PointI *b, size_t b_len)
{
    if (a_len != b_len)
        return false;
    for (size_t k = 0; k < a_len; ++k)
    {
        if (cmp_point_i(&a[k], &b[k]) != 0)
            return false;
    }
    return true;
}

void check_points(PointI *got, size_t got_len, const PointI *expected, size_t expected_len,
                  const char *file, int line)
{
    PointI *want = NULL;
    if (expected_len > 0)
    {
        want = malloc(expected_len * sizeof(PointI));
        CHECK(want != NULL);
        memcpy(want, expected, expected_len * sizeof(PointI));
        sort_points(want, expected_len);
    }
    sort_points(got, got_len);
    if (!points_eq(got, got_len, want, expected_len))
    {
        fprintf(stderr, "%s:%d: unexpected voxels\n", file, line);
        print_points("got", got, got_len);
        print_points("expected", want, expected_len);
        exit(1);
    }
    free(want);
}

void check_update(VoxelsUpdate update, const PointI *expected, size_t expected_len,
                  const char *file, int line)
{
    check_points(update.data, update.len, expected, expected_len, file, line);
    drop_update(update);
}

/* Compares two sorted lists, what names them in the failure message. */
static void check_list_eq(const PointI *got, size_t got_len, const PointI *expected,
                          size_t expected_len, const char *what, const char *ctx,
                          const char *file, int line)
{
    if (got_len != expected_len)
    {
        fail(file, line, "%s: %s has %zu voxels, expected %zu", ctx, what, got_len,
             expected_len);
    }
    for (size_t k = 0; k < got_len; ++k)
    {
        const PointI *g = &got[k];
        const PointI *e = &expected[k];
        if (cmp_point_i(g, e) != 0)
        {
            fail(file, line,
                 "%s: %s differs at %zu: got (%d, %d, %d, %u), expected (%d, %d, %d, %u)", ctx,
                 what, k, g->x, g->y, g->z, g->i, e->x, e->y, e->z, e->i);
        }
    }
}

/* ------------------------------------------------------------- helpers */

PointF pf(float x, float y, float z, float i)
{
    PointF p = {x, y, z, i};
    return p;
}

PointI pi(int32_t x, int32_t y, int32_t z, uint32_t i)
{
    PointI p = {x, y, z, i};
    return p;
}

Multipliers mult(float intensity, uint32_t hit, uint32_t target)
{
    Multipliers m = {intensity, hit, target};
    return m;
}

void *make_full(const Impl *impl, float resolution, float intensity_multiplier,
                uint32_t hit_multiplier, uint32_t target_multiplier, uint32_t destroy_threshold,
                uint32_t construct_threshold, uint32_t max_value, bool compress)
{
    void *voxelizer = NULL;
    switch (impl->kind)
    {
    case IMPL_SINGLE:
        voxelizer = create_single_threaded_voxelizer(resolution, intensity_multiplier,
                                                     hit_multiplier, target_multiplier,
                                                     destroy_threshold, construct_threshold,
                                                     max_value, compress);
        break;
    case IMPL_MULTI:
        voxelizer = create_multi_threaded_voxelizer(resolution, intensity_multiplier,
                                                    hit_multiplier, target_multiplier,
                                                    destroy_threshold, construct_threshold,
                                                    max_value, compress, impl->threads);
        break;
    case IMPL_WGPU:
        voxelizer = create_wgpu_voxelizer(resolution, intensity_multiplier, hit_multiplier,
                                          target_multiplier, destroy_threshold,
                                          construct_threshold, max_value, compress,
                                          WGPU_TEST_BUCKETS);
        break;
    }
    CHECK(voxelizer != NULL);
    return voxelizer;
}

void *make_capped(const Impl *impl, float resolution, uint32_t hit_multiplier,
                  uint32_t target_multiplier, uint32_t destroy_threshold,
                  uint32_t construct_threshold, uint32_t max_value, bool compress)
{
    return make_full(impl, resolution, 1.0f, hit_multiplier, target_multiplier,
                     destroy_threshold, construct_threshold, max_value, compress);
}

void *make(const Impl *impl, float resolution, uint32_t hit_multiplier,
           uint32_t target_multiplier, uint32_t destroy_threshold,
           uint32_t construct_threshold, bool compress)
{
    return make_capped(impl, resolution, hit_multiplier, target_multiplier, destroy_threshold,
                       construct_threshold, UINT32_MAX, compress);
}

void *plain(const Impl *impl) { return make(impl, 1.0f, 1, 1, 0, 1, false); }

void *compressed(const Impl *impl) { return make(impl, 1.0f, 1, 1, 0, 1, true); }

VoxelsUpdate add_point(void *voxelizer, PointF sensor, PointF point)
{
    return add_points(voxelizer, sensor, &point, 1);
}

void seed(void *voxelizer, float resolution, int32_t x, int32_t y, int32_t z, float i)
{
    PointF sensor = pf(((float)x + 0.5f) * resolution, ((float)y + 0.5f) * resolution,
                       ((float)z + 0.5f) * resolution, 1.0f);
    PointF point = pf(((float)x + 0.75f) * resolution, sensor.y, sensor.z, i);
    drop_update(add_point(voxelizer, sensor, point));
}

VoxelsUpdate from_above(void *voxelizer, int32_t x, int32_t y, int32_t z, float i)
{
    float fx = (float)x + 0.5f, fy = (float)y + 0.5f, fz = (float)z + 0.5f;
    return add_point(voxelizer, pf(fx, fy, fz + 1.0f, 1.0f), pf(fx, fy, fz, i));
}

VoxelsUpdate along_x(void *voxelizer, int32_t from_x, int32_t to_x, int32_t y, int32_t z,
                     float i)
{
    float fy = (float)y + 0.5f, fz = (float)z + 0.5f;
    return add_point(voxelizer, pf((float)from_x + 0.5f, fy, fz, 1.0f),
                     pf((float)to_x + 0.5f, fy, fz, i));
}

void check_ray_at(const Impl *impl, PointF sensor, PointF point, const Coord *hit,
                  size_t hit_len, Coord target, const char *file, int line)
{
    void *v = plain(impl);
    PointI *expected = malloc((hit_len + 1) * sizeof(PointI));
    CHECK(expected != NULL);
    for (size_t k = 0; k < hit_len; ++k)
    {
        seed(v, 1.0f, hit[k].x, hit[k].y, hit[k].z, (float)SEED);
        expected[k] = pi(hit[k].x, hit[k].y, hit[k].z, SEED - 1);
    }
    expected[hit_len] = pi(target.x, target.y, target.z, 1);
    check_update(add_point(v, sensor, point), expected, hit_len + 1, file, line);
    free(expected);
    drop_voxelizer(v);
}

/* ---------------------------------------------------- compress helpers */

uint32_t block_bit(int32_t x, int32_t y, int32_t z)
{
    return 1u << ((x & 3) + 4 * (y & 3) + 16 * (z & 1));
}

PointI block(int32_t x, int32_t y, int32_t z, uint32_t mask)
{
    return pi(x & ~3, y & ~3, z & ~1, mask);
}

/* Appends p to the malloc'ed list points, growing it as needed. */
static void push_point(PointI **points, size_t *len, size_t *cap, PointI p)
{
    if (*len == *cap)
    {
        *cap = *cap == 0 ? 64 : *cap * 2;
        *points = realloc(*points, *cap * sizeof(PointI));
        CHECK(*points != NULL);
    }
    (*points)[(*len)++] = p;
}

PointI *expand_blocks(const PointI *blocks, size_t len, size_t *out_len)
{
    PointI *points = NULL;
    size_t count = 0, cap = 0;
    for (size_t k = 0; k < len; ++k)
    {
        for (int32_t bit = 0; bit < 32; ++bit)
        {
            if (blocks[k].i & (1u << bit))
            {
                push_point(&points, &count, &cap,
                           pi(blocks[k].x + (bit & 3), blocks[k].y + ((bit >> 2) & 3),
                              blocks[k].z + (bit >> 4), 1));
            }
        }
    }
    sort_points(points, count);
    *out_len = count;
    return points;
}

/* -------------------------------------------------------- dump helpers */

PointI *decode_groups(VoxelGroups groups, size_t *out_len)
{
    PointI *points = NULL;
    size_t count = 0, cap = 0;
    for (size_t g = 0; g < groups.len; ++g)
    {
        const VoxelGroup *group = &groups.data[g];
        for (size_t k = 0; k < POINTS_IN_VOXEL_GROUP; ++k)
        {
            uint64_t word = group->values[k];
            if (word == 0)
            {
                continue;
            }
            /* Bits of x, y and z of the page offset interleave from the
             * lowest one. */
            int32_t offset[3] = {0, 0, 0};
            for (size_t b = 0; b < 3 * ALL_POINTS_GROUP_EDGE_BITS; ++b)
            {
                offset[b % 3] |= (int32_t)((k >> b) & 1) << (b / 3);
            }
            /* Multiplied, a negative value must not be shifted left. */
            int32_t page_x = group->point.x * (1 << ALL_POINTS_GROUP_EDGE_BITS) + offset[0];
            int32_t page_y = group->point.y * (1 << ALL_POINTS_GROUP_EDGE_BITS) + offset[1];
            int32_t page_z = group->point.z * (1 << ALL_POINTS_GROUP_EDGE_BITS) + offset[2];
            for (int32_t bit = 0; bit < 64; ++bit)
            {
                if (word & (1ull << bit))
                {
                    push_point(&points, &count, &cap,
                               pi(page_x * 4 + (bit & 3), page_y * 4 + ((bit >> 2) & 3),
                                  page_z * 4 + (bit >> 4), 1));
                }
            }
        }
    }
    sort_points(points, count);
    *out_len = count;
    return points;
}

Dump dump(const void *voxelizer)
{
    Dump out = {NULL, 0, NULL, 0};
    VoxelsUpdate points = get_all_points(voxelizer);
    out.points_len = points.len;
    out.points = malloc((points.len + 1) * sizeof(PointI));
    CHECK(out.points != NULL);
    if (points.len > 0)
    {
        memcpy(out.points, points.data, points.len * sizeof(PointI));
    }
    sort_points(out.points, out.points_len);
    drop_update(points);
    VoxelGroups groups = get_all_points_compressed(voxelizer);
    out.constructed = decode_groups(groups, &out.constructed_len);
    drop_voxel_groups(groups);
    return out;
}

void dump_free(Dump *dump)
{
    free(dump->points);
    free(dump->constructed);
    dump->points = NULL;
    dump->points_len = 0;
    dump->constructed = NULL;
    dump->constructed_len = 0;
}

void check_dump_eq(const Dump *got, const Dump *expected, const char *ctx, const char *file,
                   int line)
{
    check_list_eq(got->points, got->points_len, expected->points, expected->points_len,
                  "get_all_points", ctx, file, line);
    check_list_eq(got->constructed, got->constructed_len, expected->constructed,
                  expected->constructed_len, "get_all_points_compressed", ctx, file, line);
}

void check_constructed(const void *voxelizer, const PointI *expected, size_t expected_len,
                       const char *file, int line)
{
    VoxelGroups groups = get_all_points_compressed(voxelizer);
    size_t len;
    PointI *got = decode_groups(groups, &len);
    drop_voxel_groups(groups);
    check_points(got, len, expected, expected_len, file, line);
    free(got);
}

/* ------------------------------------------------------------ scenario */

/* With compression destroy = construct - 1, so a voxel above the destroy
 * threshold has always been constructed. */
const ScenarioParams SCENARIO_PARAMS[2] = {{0, 1, false}, {2, 3, true}};

/* Tiny LCG, same as in src/test.rs. */
static uint32_t lcg(uint64_t *state)
{
    *state = *state * 6364136223846793005ull + 1442695040888963407ull;
    return (uint32_t)(*state >> 33);
}

/* Voxel coordinates on the six faces of the cube |coord| = dist, spaced
 * step apart within -half..=half on the other two axes. Returns the count. */
static size_t cube_faces(int32_t dist, int32_t half, int32_t step, Coord **out)
{
    size_t per_axis = (size_t)(2 * half / step + 1);
    Coord *coords = malloc(per_axis * per_axis * 6 * sizeof(Coord));
    CHECK(coords != NULL);
    size_t len = 0;
    for (int32_t a = -half; a <= half; a += step)
    {
        for (int32_t b = -half; b <= half; b += step)
        {
            coords[len++] = (Coord){dist, a, b};
            coords[len++] = (Coord){-dist, a, b};
            coords[len++] = (Coord){a, dist, b};
            coords[len++] = (Coord){a, -dist, b};
            coords[len++] = (Coord){a, b, dist};
            coords[len++] = (Coord){a, b, -dist};
        }
    }
    *out = coords;
    return len;
}

Snapshot run_scenario(void *voxelizer, bool with_dump)
{
    static const struct
    {
        int32_t dist, half, step;
        uint32_t max_i;
    } calls[SCENARIO_CALLS] = {{50, 20, 4, 5}, {100, 40, 8, 3}};
    PointF sensor = pf(0.5f, 0.5f, 0.5f, 1.0f);
    uint64_t rng = 0x5EED;
    Snapshot snapshot = {{NULL}, {0}, {NULL, 0, NULL, 0}};
    for (size_t call = 0; call < SCENARIO_CALLS; ++call)
    {
        Coord *coords;
        size_t len = cube_faces(calls[call].dist, calls[call].half, calls[call].step, &coords);
        PointF *points = malloc(len * sizeof(PointF));
        CHECK(points != NULL);
        for (size_t k = 0; k < len; ++k)
        {
            uint32_t i = 1 + lcg(&rng) % calls[call].max_i;
            points[k] = pf((float)coords[k].x + 0.5f, (float)coords[k].y + 0.5f,
                           (float)coords[k].z + 0.5f, (float)i);
        }
        VoxelsUpdate update = add_points(voxelizer, sensor, points, len);
        snapshot.lens[call] = update.len;
        snapshot.results[call] = malloc((update.len + 1) * sizeof(PointI));
        CHECK(snapshot.results[call] != NULL);
        if (update.len > 0)
        {
            memcpy(snapshot.results[call], update.data, update.len * sizeof(PointI));
        }
        sort_points(snapshot.results[call], update.len);
        drop_update(update);
        free(points);
        free(coords);
    }
    if (with_dump)
    {
        snapshot.dump = dump(voxelizer);
    }
    return snapshot;
}

Snapshot scenario(void *voxelizer)
{
    Snapshot snapshot = run_scenario(voxelizer, true);
    drop_voxelizer(voxelizer);
    return snapshot;
}

void snapshot_free(Snapshot *snapshot)
{
    for (size_t call = 0; call < SCENARIO_CALLS; ++call)
    {
        free(snapshot->results[call]);
        snapshot->results[call] = NULL;
        snapshot->lens[call] = 0;
    }
    dump_free(&snapshot->dump);
}

void check_snapshot_eq(const Snapshot *got, const Snapshot *expected, const char *ctx,
                       const char *file, int line)
{
    for (size_t call = 0; call < SCENARIO_CALLS; ++call)
    {
        char what[32];
        snprintf(what, sizeof what, "result of call %zu", call);
        check_list_eq(got->results[call], got->lens[call], expected->results[call],
                      expected->lens[call], what, ctx, file, line);
    }
    check_dump_eq(&got->dump, &expected->dump, ctx, file, line);
}

/* Whether points holds one at the coordinates of p, whatever its i. */
static bool has_coords(const PointI *points, size_t len, PointI p)
{
    for (size_t k = 0; k < len; ++k)
    {
        if (points[k].x == p.x && points[k].y == p.y && points[k].z == p.z)
        {
            return true;
        }
    }
    return false;
}

PointI *replay(const Snapshot *snapshot, size_t *out_len)
{
    PointI *points = NULL;
    size_t count = 0, cap = 0;
    for (size_t call = 0; call < SCENARIO_CALLS; ++call)
    {
        for (size_t k = 0; k < snapshot->lens[call]; ++k)
        {
            PointI p = snapshot->results[call][k];
            /* A later call reporting the voxel overrides this value. */
            bool overridden = false;
            for (size_t next = call + 1; next < SCENARIO_CALLS && !overridden; ++next)
            {
                overridden = has_coords(snapshot->results[next], snapshot->lens[next], p);
            }
            if (!overridden && p.i != 0)
            {
                push_point(&points, &count, &cap, p);
            }
        }
    }
    sort_points(points, count);
    *out_len = count;
    return points;
}

/* ---------------------------------------------------------------- main */

static int usage(const char *prog)
{
    fprintf(stderr, "usage: %s list | run <impl>::<test> | run <test>\n", prog);
    return 2;
}

static void list_cases(void)
{
    for (size_t t = 0; t < TEST_CASE_COUNT; ++t)
    {
        if (TEST_CASES[t].per_impl)
        {
            for (size_t k = 0; k < IMPL_COUNT; ++k)
            {
                printf("%s::%s\n", IMPLS[k].name, TEST_CASES[t].name);
            }
        }
        else
        {
            printf("%s\n", TEST_CASES[t].name);
        }
    }
}

static int run_case(const char *name)
{
    const Impl *impl = NULL;
    const char *sep = strstr(name, "::");
    if (sep != NULL)
    {
        size_t impl_len = (size_t)(sep - name);
        for (size_t k = 0; k < IMPL_COUNT; ++k)
        {
            if (strlen(IMPLS[k].name) == impl_len && strncmp(IMPLS[k].name, name, impl_len) == 0)
            {
                impl = &IMPLS[k];
            }
        }
        if (impl == NULL)
        {
            fprintf(stderr, "unknown voxelizer type in %s\n", name);
            return 2;
        }
        name = sep + 2;
    }
    for (size_t t = 0; t < TEST_CASE_COUNT; ++t)
    {
        if (strcmp(TEST_CASES[t].name, name) == 0 && TEST_CASES[t].per_impl == (impl != NULL))
        {
            TEST_CASES[t].fn(impl);
            return 0;
        }
    }
    fprintf(stderr, "unknown test %s\n", name);
    return 2;
}

int main(int argc, char **argv)
{
    if (argc == 2 && strcmp(argv[1], "list") == 0)
    {
        list_cases();
        return 0;
    }
    if (argc == 3 && strcmp(argv[1], "run") == 0)
    {
        return run_case(argv[2]);
    }
    return usage(argv[0]);
}
