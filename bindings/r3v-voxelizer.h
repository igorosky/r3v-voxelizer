#ifndef R3V_VOXELIZER_GUARD
#define R3V_VOXELIZER_GUARD

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

/**
 * Count of bits that encode one coordinate in all points group
 */
#define ALL_POINTS_GROUP_EDGE_BITS 4

/**
 * Count of words in a compressed voxel group, one per fine page
 */
#define POINTS_IN_VOXEL_GROUP (1 << (3 * ALL_POINTS_GROUP_EDGE_BITS))

typedef struct PointI {
  int32_t x;
  int32_t y;
  int32_t z;
  uint32_t i;
} PointI;

typedef struct VoxelsUpdate {
  struct PointI *data;
  uintptr_t len;
} VoxelsUpdate;

typedef struct PointF {
  float x;
  float y;
  float z;
  float i;
} PointF;

/**
 * Scale the intensities points add to and subtract from voxels. A point of
 * intensity `i` adds `trunc(i * intensity) * target` to the voxel it ends in
 * and subtracts `trunc(i * intensity) * hit` from every voxel its ray passes
 * through.
 */
typedef struct Multipliers {
  /**
   * Scales the intensity of every point before it is truncated to an
   * integer and multiplied by `hit` or `target`. `1.0` leaves intensities
   * as they are.
   */
  float intensity;
  /**
   * Scales the intensity subtracted from voxels a ray passes through.
   */
  uint32_t hit;
  /**
   * Scales the intensity added to the voxel a ray ends in.
   */
  uint32_t target;
} Multipliers;

/**
 * Constructed voxels of a cube of 16x16x16 fine pages (64x64x64 voxels).
 *
 * `point` is the key of the group: voxel coordinates shifted right by
 * `2 + ALL_POINTS_GROUP_EDGE_BITS`, `point.i` is 0. `values[k]` is the
 * occupancy of the fine page whose offset in the group, bits of x, y and z
 * interleaved from the lowest one, is `k`. Bit `x + 4 * y + 16 * z` of it is
 * set when voxel (x, y, z) of the page is constructed.
 */
typedef struct VoxelGroup {
  struct PointI point;
  uint64_t values[POINTS_IN_VOXEL_GROUP];
} VoxelGroup;

typedef struct VoxelGroups {
  struct VoxelGroup *data;
  uintptr_t len;
} VoxelGroups;

void *create_single_threaded_voxelizer(float _resolution,
                                       float _intensity_multiplier,
                                       uint32_t _hit_multiplier,
                                       uint32_t _target_multiplier,
                                       uint32_t _destroy_threshold,
                                       uint32_t _construct_threshold,
                                       uint32_t _max_value,
                                       bool _compress);

void *create_multi_threaded_voxelizer(float _resolution,
                                      float _intensity_multiplier,
                                      uint32_t _hit_multiplier,
                                      uint32_t _target_multiplier,
                                      uint32_t _destroy_threshold,
                                      uint32_t _construct_threshold,
                                      uint32_t _max_value,
                                      bool _compress,
                                      uintptr_t _thread_count);

/**
 * Returns NULL when there is no GPU adapter capable of running compute
 * shaders, `bucket_count` is 0 or the hashmap does not fit the adapter's
 * limits.
 */
void *create_wgpu_voxelizer(float _resolution,
                            float _intensity_multiplier,
                            uint32_t _hit_multiplier,
                            uint32_t _target_multiplier,
                            uint32_t _destroy_threshold,
                            uint32_t _construct_threshold,
                            uint32_t _max_value,
                            bool _compress,
                            uint32_t _bucket_count);

/**
 * GPU memory taken by the hashmap of a wgpu voxelizer with `bucket_count`
 * buckets, in bytes. 0 when built without wgpu.
 */
uint64_t wgpu_memory_size(uint32_t _bucket_count);

struct VoxelsUpdate add_points(void *voxelizer,
                               struct PointF sensor_pos,
                               const struct PointF *points,
                               uintptr_t count);

/**
 * Multipliers used by the following `add_points` calls of `voxelizer`.
 * Voxels keep the values they got so far. Does nothing when `voxelizer` is
 * NULL.
 */
void set_multipliers(void *voxelizer, struct Multipliers multipliers);

/**
 * Multipliers used by `add_points` of `voxelizer`. All 0 when `voxelizer`
 * is NULL.
 */
struct Multipliers get_multipliers(const void *voxelizer);

/**
 * Every voxel of `voxelizer` that is not 0, with its value. Free it with
 * `drop_update`. Empty when `voxelizer` is NULL.
 */
struct VoxelsUpdate get_all_points(const void *voxelizer);

/**
 * Every constructed voxel of `voxelizer`, as groups holding at least one.
 * Without `compress` a voxel counts as constructed once it reaches
 * `construct_threshold`. Free it with `drop_voxel_groups`. Empty when
 * `voxelizer` is NULL.
 */
struct VoxelGroups get_all_points_compressed(const void *voxelizer);

void drop_update(struct VoxelsUpdate update);

void drop_voxel_groups(struct VoxelGroups groups);

void drop_voxelizer(void *voxelizer);

#endif  /* R3V_VOXELIZER_GUARD */
