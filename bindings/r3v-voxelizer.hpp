#pragma once

#include <cstdarg>
#include <cstdint>
#include <cstdlib>
#include <ostream>
#include <new>

namespace r3v {
namespace voxelizer {

/// Count of bits that encode one coordinate in all points group
constexpr static const uint32_t ALL_POINTS_GROUP_EDGE_BITS = 4;

/// Count of words in a compressed voxel group, one per fine page
constexpr static const uintptr_t POINTS_IN_VOXEL_GROUP = (1 << (3 * ALL_POINTS_GROUP_EDGE_BITS));

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

/// Scale the intensities points add to and subtract from voxels. A point of
/// intensity `i` adds `trunc(i * intensity) * target` to the voxel it ends in
/// and subtracts `trunc(i * intensity) * hit` from every voxel its ray passes
/// through.
struct Multipliers {
  /// Scales the intensity of every point before it is truncated to an
  /// integer and multiplied by `hit` or `target`. `1.0` leaves intensities
  /// as they are.
  float intensity;
  /// Scales the intensity subtracted from voxels a ray passes through.
  uint32_t hit;
  /// Scales the intensity added to the voxel a ray ends in.
  uint32_t target;
};

/// Constructed voxels of a cube of 16x16x16 fine pages (64x64x64 voxels).
///
/// `point` is the key of the group: voxel coordinates shifted right by
/// `2 + ALL_POINTS_GROUP_EDGE_BITS`, `point.i` is 0. `values[k]` is the
/// occupancy of the fine page whose offset in the group, bits of x, y and z
/// interleaved from the lowest one, is `k`. Bit `x + 4 * y + 16 * z` of it is
/// set when voxel (x, y, z) of the page is constructed.
struct VoxelGroup {
  PointI point;
  uint64_t values[POINTS_IN_VOXEL_GROUP];
};

struct VoxelGroups {
  VoxelGroup *data;
  uintptr_t len;
};

extern "C" {

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

/// Returns NULL when there is no GPU adapter capable of running compute
/// shaders, `bucket_count` is 0 or the hashmap does not fit the adapter's
/// limits.
void *create_wgpu_voxelizer(float _resolution,
                            float _intensity_multiplier,
                            uint32_t _hit_multiplier,
                            uint32_t _target_multiplier,
                            uint32_t _destroy_threshold,
                            uint32_t _construct_threshold,
                            uint32_t _max_value,
                            bool _compress,
                            uint32_t _bucket_count);

/// GPU memory taken by the hashmap of a wgpu voxelizer with `bucket_count`
/// buckets, in bytes. 0 when built without wgpu.
uint64_t wgpu_memory_size(uint32_t _bucket_count);

VoxelsUpdate add_points(void *voxelizer, PointF sensor_pos, const PointF *points, uintptr_t count);

/// Multipliers used by the following `add_points` calls of `voxelizer`.
/// Voxels keep the values they got so far. Does nothing when `voxelizer` is
/// NULL.
void set_multipliers(void *voxelizer, Multipliers multipliers);

/// Multipliers used by `add_points` of `voxelizer`. All 0 when `voxelizer`
/// is NULL.
Multipliers get_multipliers(const void *voxelizer);

/// Every voxel of `voxelizer` that is not 0, with its value. Free it with
/// `drop_update`. Empty when `voxelizer` is NULL.
VoxelsUpdate get_all_points(const void *voxelizer);

/// Every constructed voxel of `voxelizer`, as groups holding at least one.
/// Without `compress` a voxel counts as constructed once it reaches
/// `construct_threshold`. Free it with `drop_voxel_groups`. Empty when
/// `voxelizer` is NULL.
VoxelGroups get_all_points_compressed(const void *voxelizer);

/// Turns downsampling on: one downsampled voxel stands for a cube of
/// `downsampled_voxel_size`^3 voxels and is present while at least
/// `downsampled_threshold` of them are constructed. Drops the downsampled
/// state, the next downsampling call builds it again.
///
/// Returns false and changes nothing when `voxelizer` is NULL or was created
/// without `compress`, `downsampled_voxel_size` is not a power of two in
/// 2..=128 or `downsampled_threshold` is not in 1..=size^3.
bool set_downsampling_params(void *voxelizer,
                             uint8_t downsampled_voxel_size,
                             uint32_t downsampled_threshold);

/// Turns downsampling off. Does nothing when `voxelizer` is NULL.
void disable_downsampling(void *voxelizer);

/// Downsampled voxels that changed since the previous call, as 4x4x2 blocks
/// of downsampled voxels in the format of compress mode `add_points`. Every
/// present block after the downsampled state got dropped. Free it with
/// `drop_update`. Empty when `voxelizer` is NULL or downsampling is off.
VoxelsUpdate get_downsampled(void *voxelizer);

/// Every present downsampled voxel, grouped like `get_all_points_compressed`.
/// Free it with `drop_voxel_groups`. Empty when `voxelizer` is NULL or
/// downsampling is off.
VoxelGroups get_all_downsampled(const void *voxelizer);

/// Removes every voxel of `voxelizer`. Downsampling keeps its params, its
/// state is dropped. Does nothing when `voxelizer` is NULL.
void clear_voxelizer(void *voxelizer);

/// Drops the downsampled state of `voxelizer`, keeping the params: the next
/// `get_downsampled` returns every present block. Does nothing when
/// `voxelizer` is NULL.
void clear_downsampled(void *voxelizer);

void drop_update(VoxelsUpdate update);

void drop_voxel_groups(VoxelGroups groups);

void drop_voxelizer(void *voxelizer);

}  // extern "C"

}  // namespace voxelizer
}  // namespace r3v
