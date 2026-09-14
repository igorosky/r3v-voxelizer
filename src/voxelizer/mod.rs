mod single_threaded;

use crate::geometry::{PointF, PointI};

pub trait Voxelizer {
    fn add_points(&mut self, sensor_pos: PointF, points: &[PointF]) -> Vec<PointI>;
}
