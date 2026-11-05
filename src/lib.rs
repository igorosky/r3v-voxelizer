use crate::geometry::{PointF, PointI};

mod bindings;
#[cfg(any(
    feature = "single_threaded",
    feature = "multi_threaded",
    feature = "wgpu"
))]
mod downsampling;
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

/// Count of bits that encode one coordinate in all points group
pub const ALL_POINTS_GROUP_EDGE_BITS: u32 = 4;
/// Count of words in a compressed voxel group, one per fine page
pub const POINTS_IN_VOXEL_GROUP: usize = 1 << (3 * ALL_POINTS_GROUP_EDGE_BITS);

pub trait Voxelizer {
    fn add_points(&mut self, sensor_pos: PointF, points: &[PointF]) -> Vec<PointI>;
    fn get_all_points(&self) -> Vec<PointI>;
    fn get_all_points_compressed(&self) -> Vec<CompressedVoxelGroup>;
    /// Turns downsampling on: one downsampled voxel stands for a cube of
    /// `downsampled_voxel_size^3` voxels (`voxel >> log2(size)`) and is
    /// present while at least `downsampled_threshold` voxels of the cube are
    /// constructed.
    ///
    /// Needs `compress`, a size that is a power of two in `2..=128` and a
    /// threshold in `1..=size^3`. On error nothing changes. The previous
    /// downsampled state is dropped, the next downsampling call builds it
    /// again from the current voxels.
    fn set_downsampling_params(
        &mut self,
        downsampled_voxel_size: u8,
        downsampled_threshold: u32,
    ) -> Result<(), DownsamplingError>;
    /// Turns downsampling off and drops its state. Until it is turned on
    /// again `add_points` does no downsampling work.
    fn disable_downsampling(&mut self);
    /// Downsampled voxels that changed since the previous call, as 4x4x2
    /// blocks of downsampled voxels in the format of compress mode
    /// [`Voxelizer::add_points`]. After the state got dropped (params set,
    /// [`Voxelizer::clear_downsampled`], [`Voxelizer::clear`]) it returns every
    /// present block. Empty while downsampling is off.
    fn get_downsampled(&mut self) -> Vec<PointI>;
    /// Every present downsampled voxel, grouped like
    /// [`Voxelizer::get_all_points_compressed`] (in downsampled voxels).
    /// Empty while downsampling is off.
    fn get_all_downsampled(&self) -> Vec<CompressedVoxelGroup>;
    /// Removes every voxel. Downsampling keeps its params, its state is
    /// dropped.
    fn clear(&mut self);
    /// Drops the downsampled state, keeping the params. The next
    /// [`Voxelizer::get_downsampled`] returns every present block.
    fn clear_downsampled(&mut self);
    /// Multipliers used by the following [`Voxelizer::add_points`] calls.
    /// Voxels keep the values they got so far.
    fn set_multipliers(&mut self, multipliers: Multipliers);
    /// Multipliers used by [`Voxelizer::add_points`].
    fn multipliers(&self) -> Multipliers;
}

/// Scale the intensities points add to and subtract from voxels. A point of
/// intensity `i` adds `trunc(i * intensity) * target` to the voxel it ends in
/// and subtracts `trunc(i * intensity) * hit` from every voxel its ray passes
/// through.
#[cfg_attr(feature = "js", wasm_bindgen::prelude::wasm_bindgen)]
#[cfg_attr(
    feature = "python",
    pyo3::pyclass(from_py_object, module = "r3v_voxelizer")
)]
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Multipliers {
    /// Scales the intensity of every point before it is truncated to an
    /// integer and multiplied by `hit` or `target`. `1.0` leaves intensities
    /// as they are.
    pub intensity: f32,
    /// Scales the intensity subtracted from voxels a ray passes through.
    pub hit: u32,
    /// Scales the intensity added to the voxel a ray ends in.
    pub target: u32,
}

/// Why [`Voxelizer::set_downsampling_params`] refused the params.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DownsamplingError {
    /// The voxelizer was created without `compress`
    NotCompressed,
    /// Not a power of two in `2..=128`
    InvalidSize(u8),
    /// Not in `1..=size^3`
    InvalidThreshold(u32),
}

impl std::fmt::Display for DownsamplingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotCompressed => write!(f, "downsampling needs compress"),
            Self::InvalidSize(size) => write!(
                f,
                "downsampled voxel size {size} is not a power of two in 2..=128"
            ),
            Self::InvalidThreshold(threshold) => {
                write!(f, "downsampled threshold {threshold} is not in 1..=size^3")
            }
        }
    }
}

impl std::error::Error for DownsamplingError {}

/// Parameters shared by every [`Voxelizer`] implementation.
#[derive(Debug, Clone, PartialEq)]
pub struct VoxelizerParams {
    /// Size of a voxel, point coordinates are divided by it.
    pub resolution: f32,
    /// Multipliers the voxelizer starts with, see
    /// [`Voxelizer::set_multipliers`].
    pub multipliers: Multipliers,
    /// With `compress`, a constructed voxel dropping to it or below gets
    /// destroyed.
    pub destroy_threshold: u32,
    /// With `compress`, a voxel that is not constructed gets constructed when
    /// it reaches it. A voxel between the thresholds keeps its state.
    pub construct_threshold: u32,
    /// Voxel values saturate at it, so a voxel seen many times can still be
    /// cleared by later hits. `u32::MAX` for no cap.
    pub max_value: u32,
    /// Reports constructed and destroyed voxels as bitmasks of 4x4x2 blocks
    /// instead of voxel values.
    pub compress: bool,
}

#[cfg_attr(feature = "js", wasm_bindgen::prelude::wasm_bindgen)]
#[cfg_attr(
    feature = "python",
    pyo3::pyclass(from_py_object, module = "r3v_voxelizer")
)]
#[repr(C)]
#[derive(Debug, Clone, PartialEq)]
pub struct CompressedVoxelGroup {
    /// i elem is unused but this would be wasted anyway because of alignment
    #[cfg_attr(feature = "js", wasm_bindgen(getter_with_clone, readonly))]
    pub point: PointI,
    #[cfg_attr(feature = "js", wasm_bindgen(skip))]
    pub values: Box<[u64; POINTS_IN_VOXEL_GROUP]>,
}
