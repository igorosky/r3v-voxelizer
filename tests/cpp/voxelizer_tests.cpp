/*
 * Smoke tests for the C++ header bindings/r3v-voxelizer.hpp. The behaviour
 * of the C ABI is covered by tests/c, these check that the header compiles
 * as C++ and that what it declares reaches the library.
 *
 * Every test case runs in its own process (see tests/c_bindings.rs), so a
 * failed check prints a message and exits right away. Voxelizers built with
 * the crate's wgpu feature are tested when compiled with R3V_TEST_WGPU.
 *
 * usage: cpp_tests list | run <impl>::<test> | run <test>
 */
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <tuple>
#include <vector>

#include "r3v-voxelizer.hpp"

/* Next to PointI, so that std::sort and std::vector find them. */
namespace r3v::voxelizer {

bool operator==(const PointI &a, const PointI &b) {
    return std::tie(a.x, a.y, a.z, a.i) == std::tie(b.x, b.y, b.z, b.i);
}

bool operator<(const PointI &a, const PointI &b) {
    return std::tie(a.x, a.y, a.z, a.i) < std::tie(b.x, b.y, b.z, b.i);
}

} // namespace r3v::voxelizer

namespace rv = r3v::voxelizer;

namespace {

[[noreturn]] void fail(const char *file, int line, const char *what) {
    std::fprintf(stderr, "%s:%d: %s\n", file, line, what);
    std::exit(1);
}

#define CHECK(cond)                                                \
    do {                                                           \
        if (!(cond)) {                                             \
            fail(__FILE__, __LINE__, "check failed: " #cond);      \
        }                                                          \
    } while (0)

enum class ImplKind { Single, Multi, Wgpu };

struct Impl {
    const char *name;
    ImplKind kind;
};

const Impl IMPLS[] = {
    {"single", ImplKind::Single},
    {"multi_4", ImplKind::Multi},
#ifdef R3V_TEST_WGPU
    {"wgpu", ImplKind::Wgpu},
#endif
};

/* Hashmap size of the wgpu voxelizers, as in tests/c/harness.h. */
constexpr uint32_t WGPU_TEST_BUCKETS = 4096;

constexpr rv::PointF SENSOR{0.5f, 0.5f, 0.5f, 1.0f};

/* A voxelizer of the given type with resolution 1, multipliers 1, destroy
 * at 0, construct at 1 and no cap, dropped with the object. */
class Voxelizer {
  public:
    Voxelizer(const Impl &impl, bool compress) {
        switch (impl.kind) {
        case ImplKind::Single:
            ptr_ = rv::create_single_threaded_voxelizer(1.0f, 1.0f, 1, 1, 0, 1, UINT32_MAX,
                                                        compress);
            break;
        case ImplKind::Multi:
            ptr_ = rv::create_multi_threaded_voxelizer(1.0f, 1.0f, 1, 1, 0, 1, UINT32_MAX,
                                                       compress, 4);
            break;
        case ImplKind::Wgpu:
            ptr_ = rv::create_wgpu_voxelizer(1.0f, 1.0f, 1, 1, 0, 1, UINT32_MAX, compress,
                                             WGPU_TEST_BUCKETS);
            break;
        }
        CHECK(ptr_ != nullptr);
    }
    ~Voxelizer() { rv::drop_voxelizer(ptr_); }
    Voxelizer(const Voxelizer &) = delete;
    Voxelizer &operator=(const Voxelizer &) = delete;

    void *get() const { return ptr_; }

  private:
    void *ptr_ = nullptr;
};

/* The points of update, sorted. Drops the update. */
std::vector<rv::PointI> take(rv::VoxelsUpdate update) {
    std::vector<rv::PointI> points(update.data, update.data + update.len);
    rv::drop_update(update);
    std::sort(points.begin(), points.end());
    return points;
}

std::vector<rv::PointI> add_points(const Voxelizer &v, std::vector<rv::PointF> points) {
    return take(rv::add_points(v.get(), SENSOR, points.data(), points.size()));
}

/* Every voxel set in groups, with i = 1, sorted. Drops the groups. */
std::vector<rv::PointI> decode(rv::VoxelGroups groups) {
    constexpr int32_t pages = 1 << rv::ALL_POINTS_GROUP_EDGE_BITS;
    std::vector<rv::PointI> points;
    for (size_t g = 0; g < groups.len; ++g) {
        const rv::VoxelGroup &group = groups.data[g];
        for (size_t k = 0; k < rv::POINTS_IN_VOXEL_GROUP; ++k) {
            int32_t offset[3] = {0, 0, 0};
            for (size_t b = 0; b < 3 * rv::ALL_POINTS_GROUP_EDGE_BITS; ++b) {
                offset[b % 3] |= static_cast<int32_t>((k >> b) & 1) << (b / 3);
            }
            for (int32_t bit = 0; bit < 64; ++bit) {
                if (group.values[k] & (uint64_t{1} << bit)) {
                    points.push_back({(group.point.x * pages + offset[0]) * 4 + (bit & 3),
                                      (group.point.y * pages + offset[1]) * 4 + ((bit >> 2) & 3),
                                      (group.point.z * pages + offset[2]) * 4 + (bit >> 4), 1});
                }
            }
        }
    }
    rv::drop_voxel_groups(groups);
    std::sort(points.begin(), points.end());
    return points;
}

/* -------------------------------------------------------- per impl */

void add_points_and_multipliers(const Impl &impl) {
    Voxelizer v(impl, false);
    CHECK((add_points(v, {{3.5f, 0.5f, 0.5f, 1.0f}}) == std::vector<rv::PointI>{{3, 0, 0, 1}}));
    rv::set_multipliers(v.get(), {2.5f, 2, 3});
    rv::Multipliers m = rv::get_multipliers(v.get());
    CHECK(m.intensity == 2.5f && m.hit == 2 && m.target == 3);
    /* (1 * 2.5) as u32 = 2, times 3 added. */
    CHECK((add_points(v, {{3.5f, 0.5f, 0.5f, 1.0f}}) == std::vector<rv::PointI>{{3, 0, 0, 7}}));
}

void get_all_points(const Impl &impl) {
    Voxelizer v(impl, false);
    CHECK(take(rv::get_all_points(v.get())).empty());
    const std::vector<rv::PointI> expected = {{-3, 0, 0, 3}, {3, 0, 0, 2}};
    CHECK(add_points(v, {{3.5f, 0.5f, 0.5f, 2.0f}, {-2.5f, 0.5f, 0.5f, 3.0f}}) == expected);
    CHECK(take(rv::get_all_points(v.get())) == expected);
}

void get_all_points_compressed(const Impl &impl) {
    Voxelizer v(impl, true);
    rv::VoxelGroups empty = rv::get_all_points_compressed(v.get());
    CHECK(empty.len == 0);
    rv::drop_voxel_groups(empty);
    rv::PointF sensor{5.5f, 6.5f, 8.5f, 1.0f}, point{5.5f, 6.5f, 7.5f, 1.0f};
    take(rv::add_points(v.get(), sensor, &point, 1));

    rv::VoxelGroups groups = rv::get_all_points_compressed(v.get());
    CHECK(groups.len == 1);
    CHECK((groups.data[0].point == rv::PointI{0, 0, 0, 0}));
    /* Page (1, 1, 1) is word 0b111 of the group, voxel (1, 2, 3) of it is
     * bit 1 + 4 * 2 + 16 * 3. */
    for (size_t k = 0; k < rv::POINTS_IN_VOXEL_GROUP; ++k) {
        CHECK(groups.data[0].values[k] == (k == 7 ? uint64_t{1} << 57 : 0));
    }
    CHECK((decode(groups) == std::vector<rv::PointI>{{5, 6, 7, 1}}));
}

/* ----------------------------------------------------------- global */

void voxel_group_layout(const Impl *) {
    static_assert(rv::POINTS_IN_VOXEL_GROUP == 1u << (3 * rv::ALL_POINTS_GROUP_EDGE_BITS),
                  "a group holds a word per fine page");
    static_assert(sizeof(rv::VoxelGroup) ==
                      sizeof(rv::PointI) + sizeof(uint64_t) * rv::POINTS_IN_VOXEL_GROUP,
                  "a group is its key and its words");
    CHECK(rv::POINTS_IN_VOXEL_GROUP == 4096);
}

void null_voxelizer(const Impl *) {
    rv::VoxelsUpdate points = rv::get_all_points(nullptr);
    CHECK(points.data == nullptr && points.len == 0);
    rv::drop_update(points);
    rv::VoxelGroups groups = rv::get_all_points_compressed(nullptr);
    CHECK(groups.data == nullptr && groups.len == 0);
    rv::drop_voxel_groups(groups);
    rv::Multipliers m = rv::get_multipliers(nullptr);
    CHECK(m.intensity == 0.0f && m.hit == 0 && m.target == 0);
    rv::drop_voxelizer(nullptr);
}

/* ------------------------------------------------------------ table */

struct TestCase {
    const char *name;
    void (*per_impl)(const Impl &);
    void (*global)(const Impl *);
};

const TestCase TEST_CASES[] = {
    {"add_points_and_multipliers", add_points_and_multipliers, nullptr},
    {"get_all_points", get_all_points, nullptr},
    {"get_all_points_compressed", get_all_points_compressed, nullptr},
    {"voxel_group_layout", nullptr, voxel_group_layout},
    {"null_voxelizer", nullptr, null_voxelizer},
};

int usage(const char *prog) {
    std::fprintf(stderr, "usage: %s list | run <impl>::<test> | run <test>\n", prog);
    return 2;
}

} // namespace

int main(int argc, char **argv) {
    if (argc == 2 && std::strcmp(argv[1], "list") == 0) {
        for (const TestCase &test : TEST_CASES) {
            if (test.per_impl != nullptr) {
                for (const Impl &impl : IMPLS) {
                    std::printf("%s::%s\n", impl.name, test.name);
                }
            } else {
                std::printf("%s\n", test.name);
            }
        }
        return 0;
    }
    if (argc != 3 || std::strcmp(argv[1], "run") != 0) {
        return usage(argv[0]);
    }
    std::string name = argv[2];
    const Impl *impl = nullptr;
    size_t sep = name.find("::");
    if (sep != std::string::npos) {
        for (const Impl &candidate : IMPLS) {
            if (name.compare(0, sep, candidate.name) == 0 && std::strlen(candidate.name) == sep) {
                impl = &candidate;
            }
        }
        if (impl == nullptr) {
            std::fprintf(stderr, "unknown voxelizer type in %s\n", argv[2]);
            return 2;
        }
        name = name.substr(sep + 2);
    }
    for (const TestCase &test : TEST_CASES) {
        if (name == test.name && (test.per_impl != nullptr) == (impl != nullptr)) {
            if (impl != nullptr) {
                test.per_impl(*impl);
            } else {
                test.global(nullptr);
            }
            return 0;
        }
    }
    std::fprintf(stderr, "unknown test %s\n", argv[2]);
    return 2;
}
