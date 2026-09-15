use crate::geometry::{PointF, PointI};

mod bindings;
pub mod geometry;
#[cfg(feature = "multi_threaded")]
mod multi_threaded;
#[cfg(feature = "single_threaded")]
mod single_threaded;
#[cfg(feature = "wgpu")]
mod wgpu;

#[cfg(feature = "single_threaded")]
pub use single_threaded::SingleThreaded;

pub trait Voxelizer {
    fn add_points(&mut self, sensor_pos: PointF, points: &[PointF]) -> Vec<PointI>;
}
