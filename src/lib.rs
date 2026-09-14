use crate::{
    geometry::{PointF, PointI},
    storage::Storage,
    voxelizer::Voxelizer,
};

mod bindings;
mod geometry;
mod storage;
mod voxelizer;

pub struct Context {
    voxelizer: Box<dyn Voxelizer>,
    voxels: Box<dyn Storage>,
    last_update: Option<Vec<PointI>>,
}

impl Context {
    pub fn add_points(&mut self, sensor_pos: PointF, points: &[PointF]) {}
    #[inline]
    pub fn last_update(&self) -> Option<&[PointI]> {
        self.last_update.as_ref().map(|v| v.as_slice())
    }
    pub fn last_update_compressed(&self) {}
    #[inline]
    pub fn get_voxels(&self) -> Vec<PointI> {
        self.voxels.iter().cloned().collect()
    }
    pub fn get_voxels_compressed(&self) {}
}
