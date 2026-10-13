#ifndef R3V_VOXELIZER_GUARD
#define R3V_VOXELIZER_GUARD

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

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

void *create_single_threaded_voxelizer(float _resolution,
                                       uint32_t _hit_multiplier,
                                       uint32_t _target_multiplier,
                                       uint32_t _destroy_threshold,
                                       uint32_t _construct_threshold,
                                       uint32_t _max_value,
                                       bool _compress);

void *create_multi_threaded_voxelizer(float _resolution,
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

void drop_update(struct VoxelsUpdate update);

void drop_voxelizer(void *voxelizer);

#endif  /* R3V_VOXELIZER_GUARD */
