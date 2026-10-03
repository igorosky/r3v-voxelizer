use core::ffi::c_void;

use crate::{
    Voxelizer,
    geometry::{PointF, PointI},
};

#[repr(C)]
pub struct VoxelsUpdate {
    data: *mut PointI,
    len: usize,
}

#[unsafe(no_mangle)]
pub extern "C" fn create_single_threaded_voxelizer(
    resolution: f32,
    hit_multiplayer: u32,
    target_multiplayer: u32,
    destroy_threshold: u32,
    construct_threshold: u32,
    compress: bool,
) -> *mut c_void {
    #[cfg(not(feature = "multi_threaded"))]
    return core::ptr::null_mut();
    #[cfg(feature = "single_threaded")]
    {
        let voxelizer: Box<dyn Voxelizer> = Box::new(crate::SingleThreaded::new(
            resolution,
            hit_multiplayer,
            target_multiplayer,
            destroy_threshold,
            construct_threshold,
            compress,
        ));
        Box::into_raw(Box::new(voxelizer)) as *mut c_void
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn create_multi_threaded_voxelizer(
    resolution: f32,
    hit_multiplayer: u32,
    target_multiplayer: u32,
    destroy_threshold: u32,
    construct_threshold: u32,
    compress: bool,
    thread_count: usize,
) -> *mut c_void {
    #[cfg(not(feature = "multi_threaded"))]
    return core::ptr::null_mut();
    #[cfg(feature = "multi_threaded")]
    {
        let voxelizer: Box<dyn Voxelizer> = Box::new(
            match crate::MultiThreaded::new(
                resolution,
                hit_multiplayer,
                target_multiplayer,
                destroy_threshold,
                construct_threshold,
                compress,
                thread_count,
            ) {
                Ok(v) => v,
                Err(_) => return core::ptr::null_mut(),
            },
        );
        Box::into_raw(Box::new(voxelizer)) as *mut c_void
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

#[unsafe(no_mangle)]
pub extern "C" fn drop_update(update: VoxelsUpdate) {
    if !update.data.is_null() && update.len != 0 {
        drop(unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(update.data, update.len)) });
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn drop_voxelizer(voxelizer: *mut c_void) {
    if !voxelizer.is_null() {
        drop(unsafe { Box::from_raw(voxelizer as *mut Box<dyn Voxelizer>) });
    }
}
