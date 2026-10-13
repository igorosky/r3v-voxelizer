#pragma once

#include <cstdarg>
#include <cstdint>
#include <cstdlib>
#include <ostream>
#include <new>

namespace r3v {
namespace voxelizer {

struct PointI {
  int32_t x;
  int32_t y;
  int32_t z;
  uint32_t i;
};

struct VoxelsUpdate {
  PointI *data;
  uintptr_t len;
};

struct PointF {
  float x;
  float y;
  float z;
  float i;
};

extern "C" {

void *create_single_threaded_voxelizer(float _resolution,
                                       uint32_t _hit_multiplier,
                                       uint32_t _target_multiplier,
                                       uint32_t _destroy_threshold,
                                       uint32_t _construct_threshold,
                                       bool _compress);

void *create_multi_threaded_voxelizer(float _resolution,
                                      uint32_t _hit_multiplier,
                                      uint32_t _target_multiplier,
                                      uint32_t _destroy_threshold,
                                      uint32_t _construct_threshold,
                                      bool _compress,
                                      uintptr_t _thread_count);

/// Returns NULL when there is no GPU adapter capable of running compute
/// shaders, `bucket_count` is 0 or the hashmap does not fit the adapter's
/// limits.
void *create_wgpu_voxelizer(float _resolution,
                            uint32_t _hit_multiplier,
                            uint32_t _target_multiplier,
                            uint32_t _destroy_threshold,
                            uint32_t _construct_threshold,
                            bool _compress,
                            uint32_t _bucket_count);

/// GPU memory taken by the hashmap of a wgpu voxelizer with `bucket_count`
/// buckets, in bytes. 0 when built without wgpu.
uint64_t wgpu_memory_size(uint32_t _bucket_count);

VoxelsUpdate add_points(void *voxelizer, PointF sensor_pos, const PointF *points, uintptr_t count);

void drop_update(VoxelsUpdate update);

void drop_voxelizer(void *voxelizer);

}  // extern "C"

}  // namespace voxelizer
}  // namespace r3v
