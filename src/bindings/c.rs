use core::ffi::c_void;

use crate::{SingleThreaded, Voxelizer, geometry::PointF, multi_threaded::MultiThreaded};

#[unsafe(no_mangle)]
pub extern "C" fn create_single_threaded_voxelizer(
    resolution: f32,
    hit_multiplayer: u32,
    target_multiplayer: u32,
    destroy_threshold: u32,
    construct_threshold: u32,
    compress: bool,
) -> *mut c_void {
    let voxelizer: Box<dyn Voxelizer> = Box::new(SingleThreaded::new(
        resolution,
        hit_multiplayer,
        target_multiplayer,
        destroy_threshold,
        construct_threshold,
        compress,
    ));
    Box::into_raw(Box::new(voxelizer)) as *mut c_void
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
    let voxelizer: Box<dyn Voxelizer> = Box::new(
        match MultiThreaded::new(
            resolution,
            hit_multiplayer,
            target_multiplayer,
            destroy_threshold,
            construct_threshold,
            compress,
            thread_count,
        ) {
            Ok(v) => v,
            Err(_) => return core::ptr::null_mut() as *mut c_void,
        },
    );
    Box::into_raw(Box::new(voxelizer)) as *mut c_void
}

#[unsafe(no_mangle)]
pub extern "C" fn add_points(
    voxelizer: *mut c_void,
    sensor_pos: PointF,
    points: *const PointF,
    count: usize,
) {
    let mut voxelizer = unsafe { Box::from_raw(voxelizer as *mut Box<dyn Voxelizer>) };
    let points = unsafe { std::slice::from_raw_parts(points, count) };
    voxelizer.add_points(sensor_pos, points);
    std::mem::forget(voxelizer);
}

#[unsafe(no_mangle)]
pub extern "C" fn drop_voxelizer(voxelizer: *mut c_void) {
    drop(unsafe { Box::from_raw(voxelizer as *mut Box<dyn Voxelizer>) });
}
