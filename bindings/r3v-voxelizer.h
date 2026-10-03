#ifndef R3V_VOXELIZER_GUARD
#define R3V_VOXELIZER_GUARD

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct PointF {
  float x;
  float y;
  float z;
  float i;
} PointF;

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

void add_points(void *voxelizer,
                struct PointF sensor_pos,
                const struct PointF *points,
                uintptr_t count);

void drop_voxelizer(void *voxelizer);

#endif  /* R3V_VOXELIZER_GUARD */
