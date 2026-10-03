#pragma once

#include <cstdarg>
#include <cstdint>
#include <cstdlib>
#include <ostream>
#include <new>

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

void add_points(void *voxelizer, PointF sensor_pos, const PointF *points, uintptr_t count);

void drop_voxelizer(void *voxelizer);

}  // extern "C"
