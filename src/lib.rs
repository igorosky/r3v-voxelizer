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
pub use multi_threaded::MultithreadedThreaded;
#[cfg(feature = "single_threaded")]
pub use single_threaded::SingleThreaded;

pub trait Voxelizer {
    fn add_points(&mut self, sensor_pos: PointF, points: &[PointF]) -> Vec<PointI>;
}
