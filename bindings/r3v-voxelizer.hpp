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

void *create_single_threaded_voxelizer(float resolution,
                                       uint32_t hit_multiplayer,
                                       uint32_t target_multiplayer,
                                       uint32_t destroy_threshold,
                                       uint32_t construct_threshold,
                                       bool compress);

void *create_multi_threaded_voxelizer(float resolution,
                                      uint32_t hit_multiplayer,
                                      uint32_t target_multiplayer,
                                      uint32_t destroy_threshold,
                                      uint32_t construct_threshold,
                                      bool compress,
                                      uintptr_t thread_count);

VoxelsUpdate add_points(void *voxelizer, PointF sensor_pos, const PointF *points, uintptr_t count);

void drop_update(VoxelsUpdate update);

void drop_voxelizer(void *voxelizer);

}  // extern "C"

}  // namespace voxelizer
}  // namespace r3v
