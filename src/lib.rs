use crate::geometry::{PointF, PointI};

mod bindings;
pub mod geometry;
#[cfg(feature = "multi_threaded")]
mod multi_threaded;
#[cfg(feature = "single_threaded")]
mod single_threaded;
#[cfg(test)]
mod test;
#[cfg(feature = "wgpu")]
mod wgpu;

#[cfg(feature = "multi_threaded")]
pub use multi_threaded::MultiThreaded;
#[cfg(feature = "single_threaded")]
pub use single_threaded::SingleThreaded;
#[cfg(feature = "wgpu")]
pub use wgpu::WgpuAccelerated;

pub trait Voxelizer {
    fn add_points(&mut self, sensor_pos: PointF, points: &[PointF]) -> Vec<PointI>;
}

/// Parameters shared by every [`Voxelizer`] implementation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VoxelizerParams {
    /// Size of a voxel, point coordinates are divided by it.
    pub resolution: f32,
    /// Scales the intensity of every point before it is truncated to an
    /// integer and multiplied by `hit_multiplier` or `target_multiplier`.
    /// `1.0` leaves intensities as they are.
    pub intensity_multiplier: f32,
    /// Scales the intensity subtracted from voxels a ray passes through.
    pub hit_multiplier: u32,
    /// Scales the intensity added to the voxel a ray ends in.
    pub target_multiplier: u32,
    /// With `compress`, a voxel dropping to it or below gets destroyed.
    pub destroy_threshold: u32,
    /// With `compress`, a voxel reaching it gets constructed.
    pub construct_threshold: u32,
    /// Voxel values saturate at it, so a voxel seen many times can still be
    /// cleared by later hits. `u32::MAX` for no cap.
    pub max_value: u32,
    /// Reports constructed and destroyed voxels as bitmasks of 4x4x2 blocks
    /// instead of voxel values.
    pub compress: bool,
}
