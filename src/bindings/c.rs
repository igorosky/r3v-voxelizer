use core::ffi::c_void;

use crate::{
    Multipliers, POINTS_IN_VOXEL_GROUP, Voxelizer,
    geometry::{PointF, PointI},
};

#[repr(C)]
pub struct VoxelsUpdate {
    data: *mut PointI,
    len: usize,
}

/// Constructed voxels of a cube of 16x16x16 fine pages (64x64x64 voxels).
///
/// `point` is the key of the group: voxel coordinates shifted right by
/// `2 + ALL_POINTS_GROUP_EDGE_BITS`, `point.i` is 0. `values[k]` is the
/// occupancy of the fine page whose offset in the group, bits of x, y and z
/// interleaved from the lowest one, is `k`. Bit `x + 4 * y + 16 * z` of it is
/// set when voxel (x, y, z) of the page is constructed.
#[repr(C)]
pub struct VoxelGroup {
    point: PointI,
    values: [u64; POINTS_IN_VOXEL_GROUP],
}

#[repr(C)]
pub struct VoxelGroups {
    data: *mut VoxelGroup,
    len: usize,
}

#[cfg(any(
    feature = "single_threaded",
    feature = "multi_threaded",
    feature = "wgpu"
))]
#[expect(clippy::too_many_arguments)]
fn params(
    resolution: f32,
    intensity_multiplier: f32,
    hit_multiplier: u32,
    target_multiplier: u32,
    destroy_threshold: u32,
    construct_threshold: u32,
    max_value: u32,
    compress: bool,
) -> crate::VoxelizerParams {
    crate::VoxelizerParams {
        resolution,
        multipliers: Multipliers {
            intensity: intensity_multiplier,
            hit: hit_multiplier,
            target: target_multiplier,
        },
        destroy_threshold,
        construct_threshold,
        max_value,
        compress,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn create_single_threaded_voxelizer(
    _resolution: f32,
    _intensity_multiplier: f32,
    _hit_multiplier: u32,
    _target_multiplier: u32,
    _destroy_threshold: u32,
    _construct_threshold: u32,
    _max_value: u32,
    _compress: bool,
) -> *mut c_void {
    #[cfg(not(feature = "single_threaded"))]
    return core::ptr::null_mut();
    #[cfg(feature = "single_threaded")]
    {
        let voxelizer: Box<dyn Voxelizer> = Box::new(crate::SingleThreaded::new(params(
            _resolution,
            _intensity_multiplier,
            _hit_multiplier,
            _target_multiplier,
            _destroy_threshold,
            _construct_threshold,
            _max_value,
            _compress,
        )));
        Box::into_raw(Box::new(voxelizer)) as *mut c_void
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn create_multi_threaded_voxelizer(
    _resolution: f32,
    _intensity_multiplier: f32,
    _hit_multiplier: u32,
    _target_multiplier: u32,
    _destroy_threshold: u32,
    _construct_threshold: u32,
    _max_value: u32,
    _compress: bool,
    _thread_count: usize,
) -> *mut c_void {
    #[cfg(not(feature = "multi_threaded"))]
    return core::ptr::null_mut();
    #[cfg(feature = "multi_threaded")]
    {
        let voxelizer: Box<dyn Voxelizer> = Box::new(
            match crate::MultiThreaded::new(
                params(
                    _resolution,
                    _intensity_multiplier,
                    _hit_multiplier,
                    _target_multiplier,
                    _destroy_threshold,
                    _construct_threshold,
                    _max_value,
                    _compress,
                ),
                _thread_count,
            ) {
                Ok(v) => v,
                Err(_) => return core::ptr::null_mut(),
            },
        );
        Box::into_raw(Box::new(voxelizer)) as *mut c_void
    }
}

/// Returns NULL when there is no GPU adapter capable of running compute
/// shaders, `bucket_count` is 0 or the hashmap does not fit the adapter's
/// limits.
#[unsafe(no_mangle)]
pub extern "C" fn create_wgpu_voxelizer(
    _resolution: f32,
    _intensity_multiplier: f32,
    _hit_multiplier: u32,
    _target_multiplier: u32,
    _destroy_threshold: u32,
    _construct_threshold: u32,
    _max_value: u32,
    _compress: bool,
    _bucket_count: u32,
) -> *mut c_void {
    #[cfg(not(feature = "wgpu"))]
    return core::ptr::null_mut();
    #[cfg(feature = "wgpu")]
    {
        let voxelizer: Box<dyn Voxelizer> = Box::new(
            match pollster::block_on(crate::WgpuAccelerated::new(
                params(
                    _resolution,
                    _intensity_multiplier,
                    _hit_multiplier,
                    _target_multiplier,
                    _destroy_threshold,
                    _construct_threshold,
                    _max_value,
                    _compress,
                ),
                _bucket_count,
            )) {
                Some(v) => v,
                None => return core::ptr::null_mut(),
            },
        );
        Box::into_raw(Box::new(voxelizer)) as *mut c_void
    }
}

/// GPU memory taken by the hashmap of a wgpu voxelizer with `bucket_count`
/// buckets, in bytes. 0 when built without wgpu.
#[unsafe(no_mangle)]
pub extern "C" fn wgpu_memory_size(_bucket_count: u32) -> u64 {
    #[cfg(not(feature = "wgpu"))]
    return 0;
    #[cfg(feature = "wgpu")]
    {
        crate::WgpuAccelerated::memory_size(_bucket_count)
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn add_points(
    voxelizer: *mut c_void,
    sensor_pos: PointF,
    points: *const PointF,
    count: usize,
) -> VoxelsUpdate {
    if voxelizer.is_null() || points.is_null() || count == 0 {
        return VoxelsUpdate {
            data: core::ptr::null_mut(),
            len: 0,
        };
    }
    let mut voxelizer = unsafe { Box::from_raw(voxelizer as *mut Box<dyn Voxelizer>) };
    let points = unsafe { std::slice::from_raw_parts(points, count) };
    let mut result = voxelizer.add_points(sensor_pos, points).into_boxed_slice();
    std::mem::forget(voxelizer);
    let len = result.len();
    let data = result.as_mut_ptr();
    std::mem::forget(result);
    VoxelsUpdate { data, len }
}

/// Multipliers used by the following `add_points` calls of `voxelizer`.
/// Voxels keep the values they got so far. Does nothing when `voxelizer` is
/// NULL.
#[unsafe(no_mangle)]
pub extern "C" fn set_multipliers(voxelizer: *mut c_void, multipliers: Multipliers) {
    if voxelizer.is_null() {
        return;
    }
    let voxelizer = unsafe { &mut *(voxelizer as *mut Box<dyn Voxelizer>) };
    voxelizer.set_multipliers(multipliers);
}

/// Multipliers used by `add_points` of `voxelizer`. All 0 when `voxelizer`
/// is NULL.
#[unsafe(no_mangle)]
pub extern "C" fn get_multipliers(voxelizer: *const c_void) -> Multipliers {
    if voxelizer.is_null() {
        return Multipliers {
            intensity: 0.0,
            hit: 0,
            target: 0,
        };
    }
    let voxelizer = unsafe { &*(voxelizer as *const Box<dyn Voxelizer>) };
    voxelizer.multipliers()
}

/// Every voxel of `voxelizer` that is not 0, with its value. Free it with
/// `drop_update`. Empty when `voxelizer` is NULL.
#[unsafe(no_mangle)]
pub extern "C" fn get_all_points(voxelizer: *const c_void) -> VoxelsUpdate {
    if voxelizer.is_null() {
        return VoxelsUpdate {
            data: core::ptr::null_mut(),
            len: 0,
        };
    }
    let voxelizer = unsafe { &*(voxelizer as *const Box<dyn Voxelizer>) };
    let mut result = voxelizer.get_all_points().into_boxed_slice();
    let len = result.len();
    let data = result.as_mut_ptr();
    std::mem::forget(result);
    VoxelsUpdate { data, len }
}

/// Every constructed voxel of `voxelizer`, as groups holding at least one.
/// Without `compress` a voxel counts as constructed once it reaches
/// `construct_threshold`. Free it with `drop_voxel_groups`. Empty when
/// `voxelizer` is NULL.
#[unsafe(no_mangle)]
pub extern "C" fn get_all_points_compressed(voxelizer: *const c_void) -> VoxelGroups {
    if voxelizer.is_null() {
        return VoxelGroups {
            data: core::ptr::null_mut(),
            len: 0,
        };
    }
    let voxelizer = unsafe { &*(voxelizer as *const Box<dyn Voxelizer>) };
    let mut result: Box<[VoxelGroup]> = voxelizer
        .get_all_points_compressed()
        .into_iter()
        .map(|group| VoxelGroup {
            point: group.point,
            values: *group.values,
        })
        .collect();
    let len = result.len();
    let data = result.as_mut_ptr();
    std::mem::forget(result);
    VoxelGroups { data, len }
}

#[unsafe(no_mangle)]
pub extern "C" fn drop_update(update: VoxelsUpdate) {
    if !update.data.is_null() && update.len != 0 {
        drop(unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(update.data, update.len)) });
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn drop_voxel_groups(groups: VoxelGroups) {
    if !groups.data.is_null() && groups.len != 0 {
        drop(unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(groups.data, groups.len)) });
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn drop_voxelizer(voxelizer: *mut c_void) {
    if !voxelizer.is_null() {
        drop(unsafe { Box::from_raw(voxelizer as *mut Box<dyn Voxelizer>) });
    }
}
