use core::ffi::c_void;

use crate::{
    CompressedVoxelGroup, Multipliers, POINTS_IN_VOXEL_GROUP, Voxelizer,
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

impl VoxelsUpdate {
    const EMPTY: Self = Self {
        data: core::ptr::null_mut(),
        len: 0,
    };

    /// Hands `points` over to C, freed by `drop_update`.
    fn from_points(points: Vec<PointI>) -> Self {
        let mut points = points.into_boxed_slice();
        let len = points.len();
        let data = points.as_mut_ptr();
        std::mem::forget(points);
        Self { data, len }
    }
}

impl VoxelGroups {
    const EMPTY: Self = Self {
        data: core::ptr::null_mut(),
        len: 0,
    };

    /// Hands `groups` over to C, freed by `drop_voxel_groups`.
    fn from_groups(groups: Vec<CompressedVoxelGroup>) -> Self {
        let mut groups: Box<[VoxelGroup]> = groups
            .into_iter()
            .map(|group| VoxelGroup {
                point: group.point,
                values: *group.values,
            })
            .collect();
        let len = groups.len();
        let data = groups.as_mut_ptr();
        std::mem::forget(groups);
        Self { data, len }
    }
}

/// The voxelizer behind a handle of `create_*_voxelizer`, `None` for NULL.
///
/// # Safety
///
/// `voxelizer` is NULL or a live handle, not used elsewhere meanwhile.
unsafe fn voxelizer_mut<'a>(voxelizer: *mut c_void) -> Option<&'a mut dyn Voxelizer> {
    // SAFETY: see above
    unsafe { (voxelizer as *mut Box<dyn Voxelizer>).as_mut() }.map(|voxelizer| &mut **voxelizer)
}

/// [`voxelizer_mut`] for reading.
///
/// # Safety
///
/// `voxelizer` is NULL or a live handle, not changed meanwhile.
unsafe fn voxelizer_ref<'a>(voxelizer: *const c_void) -> Option<&'a dyn Voxelizer> {
    // SAFETY: see above
    unsafe { (voxelizer as *const Box<dyn Voxelizer>).as_ref() }.map(|voxelizer| &**voxelizer)
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
    match unsafe { voxelizer_ref(voxelizer) } {
        Some(voxelizer) => VoxelsUpdate::from_points(voxelizer.get_all_points()),
        None => VoxelsUpdate::EMPTY,
    }
}

/// Every constructed voxel of `voxelizer`, as groups holding at least one.
/// Without `compress` a voxel counts as constructed once it reaches
/// `construct_threshold`. Free it with `drop_voxel_groups`. Empty when
/// `voxelizer` is NULL.
#[unsafe(no_mangle)]
pub extern "C" fn get_all_points_compressed(voxelizer: *const c_void) -> VoxelGroups {
    match unsafe { voxelizer_ref(voxelizer) } {
        Some(voxelizer) => VoxelGroups::from_groups(voxelizer.get_all_points_compressed()),
        None => VoxelGroups::EMPTY,
    }
}

/// Turns downsampling on: one downsampled voxel stands for a cube of
/// `downsampled_voxel_size`^3 voxels and is present while at least
/// `downsampled_threshold` of them are constructed. Drops the downsampled
/// state, the next downsampling call builds it again.
///
/// Returns false and changes nothing when `voxelizer` is NULL or was created
/// without `compress`, `downsampled_voxel_size` is not a power of two in
/// 2..=128 or `downsampled_threshold` is not in 1..=size^3.
#[unsafe(no_mangle)]
pub extern "C" fn set_downsampling_params(
    voxelizer: *mut c_void,
    downsampled_voxel_size: u8,
    downsampled_threshold: u32,
) -> bool {
    unsafe { voxelizer_mut(voxelizer) }.is_some_and(|voxelizer| {
        voxelizer
            .set_downsampling_params(downsampled_voxel_size, downsampled_threshold)
            .is_ok()
    })
}

/// Turns downsampling off. Does nothing when `voxelizer` is NULL.
#[unsafe(no_mangle)]
pub extern "C" fn disable_downsampling(voxelizer: *mut c_void) {
    if let Some(voxelizer) = unsafe { voxelizer_mut(voxelizer) } {
        voxelizer.disable_downsampling();
    }
}

/// Downsampled voxels that changed since the previous call, as 4x4x2 blocks
/// of downsampled voxels in the format of compress mode `add_points`. Every
/// present block after the downsampled state got dropped. Free it with
/// `drop_update`. Empty when `voxelizer` is NULL or downsampling is off.
#[unsafe(no_mangle)]
pub extern "C" fn get_downsampled(voxelizer: *mut c_void) -> VoxelsUpdate {
    match unsafe { voxelizer_mut(voxelizer) } {
        Some(voxelizer) => VoxelsUpdate::from_points(voxelizer.get_downsampled()),
        None => VoxelsUpdate::EMPTY,
    }
}

/// Every present downsampled voxel, grouped like `get_all_points_compressed`.
/// Free it with `drop_voxel_groups`. Empty when `voxelizer` is NULL or
/// downsampling is off.
#[unsafe(no_mangle)]
pub extern "C" fn get_all_downsampled(voxelizer: *const c_void) -> VoxelGroups {
    match unsafe { voxelizer_ref(voxelizer) } {
        Some(voxelizer) => VoxelGroups::from_groups(voxelizer.get_all_downsampled()),
        None => VoxelGroups::EMPTY,
    }
}

/// Removes every voxel of `voxelizer`. Downsampling keeps its params, its
/// state is dropped. Does nothing when `voxelizer` is NULL.
#[unsafe(no_mangle)]
pub extern "C" fn clear_voxelizer(voxelizer: *mut c_void) {
    if let Some(voxelizer) = unsafe { voxelizer_mut(voxelizer) } {
        voxelizer.clear();
    }
}

/// Drops the downsampled state of `voxelizer`, keeping the params: the next
/// `get_downsampled` returns every present block. Does nothing when
/// `voxelizer` is NULL.
#[unsafe(no_mangle)]
pub extern "C" fn clear_downsampled(voxelizer: *mut c_void) {
    if let Some(voxelizer) = unsafe { voxelizer_mut(voxelizer) } {
        voxelizer.clear_downsampled();
    }
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
