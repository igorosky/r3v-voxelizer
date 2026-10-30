#[pyo3::pymodule]
mod r3v_voxelizer {
    #[cfg(any(
        feature = "single_threaded",
        feature = "multi_threaded",
        feature = "wgpu"
    ))]
    use crate::Voxelizer;

    #[pyo3::pymethods]
    impl PointF {
        #[new]
        pub fn new(x: f32, y: f32, z: f32, i: f32) -> Self {
            Self { x, y, z, i }
        }
        pub fn x(&self) -> f32 {
            self.x
        }
        pub fn set_x(&mut self, x: f32) {
            self.x = x;
        }
        pub fn y(&self) -> f32 {
            self.y
        }
        pub fn set_y(&mut self, y: f32) {
            self.y = y;
        }
        pub fn z(&self) -> f32 {
            self.z
        }
        pub fn set_z(&mut self, z: f32) {
            self.z = z;
        }
        pub fn i(&self) -> f32 {
            self.i
        }
        pub fn set_i(&mut self, i: f32) {
            self.i = i;
        }
    }

    #[pyo3::pymethods]
    impl PointI {
        #[new]
        pub fn new(x: i32, y: i32, z: i32, i: u32) -> Self {
            Self { x, y, z, i }
        }
        pub fn x(&self) -> i32 {
            self.x
        }
        pub fn set_x(&mut self, x: i32) {
            self.x = x;
        }
        pub fn y(&self) -> i32 {
            self.y
        }
        pub fn set_y(&mut self, y: i32) {
            self.y = y;
        }
        pub fn z(&self) -> i32 {
            self.z
        }
        pub fn set_z(&mut self, z: i32) {
            self.z = z;
        }
        pub fn i(&self) -> u32 {
            self.i
        }
        pub fn set_i(&mut self, i: u32) {
            self.i = i;
        }
    }

    #[pyo3::pymethods]
    impl Multipliers {
        #[new]
        pub fn new(intensity: f32, hit: u32, target: u32) -> Self {
            Self {
                intensity,
                hit,
                target,
            }
        }
        pub fn intensity(&self) -> f32 {
            self.intensity
        }
        pub fn set_intensity(&mut self, intensity: f32) {
            self.intensity = intensity;
        }
        pub fn hit(&self) -> u32 {
            self.hit
        }
        pub fn set_hit(&mut self, hit: u32) {
            self.hit = hit;
        }
        pub fn target(&self) -> u32 {
            self.target
        }
        pub fn set_target(&mut self, target: u32) {
            self.target = target;
        }
    }

    #[pyo3::pymethods]
    impl CompressedVoxelGroup {
        /// Raises ValueError unless `values` holds a word for every fine page
        /// of the group (4096).
        #[new]
        pub fn new(point: PointI, values: Vec<u64>) -> pyo3::PyResult<Self> {
            match values.into_boxed_slice().try_into() {
                Ok(values) => Ok(Self { point, values }),
                Err(values) => Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "values holds {} words, a group holds {}",
                    values.len(),
                    crate::POINTS_IN_VOXEL_GROUP
                ))),
            }
        }
        /// Key of the group: voxel coordinates shifted right by 6, i is 0.
        pub fn point(&self) -> PointI {
            self.point.clone()
        }
        /// Occupancy of every fine page of the group, in z-order. Bit
        /// `x + 4 * y + 16 * z` of a word is set when voxel (x, y, z) of the
        /// page is constructed.
        pub fn values(&self) -> Vec<u64> {
            self.values.to_vec()
        }
    }

    #[pymodule_export]
    use crate::{CompressedVoxelGroup, Multipliers, PointF, PointI};

    #[cfg(feature = "single_threaded")]
    #[pyo3::pyclass]
    #[derive(Debug)]
    struct SingleThreaded(crate::SingleThreaded);

    #[cfg(feature = "single_threaded")]
    #[pyo3::pymethods]
    impl SingleThreaded {
        #[new]
        #[expect(clippy::too_many_arguments)]
        fn new(
            resolution: f32,
            intensity_multiplier: f32,
            hit_multiplier: u32,
            target_multiplier: u32,
            destroy_threshold: u32,
            construct_threshold: u32,
            max_value: u32,
            compress: bool,
        ) -> Self {
            Self(crate::SingleThreaded::new(crate::VoxelizerParams {
                resolution,
                multipliers: crate::Multipliers {
                    intensity: intensity_multiplier,
                    hit: hit_multiplier,
                    target: target_multiplier,
                },
                destroy_threshold,
                construct_threshold,
                max_value,
                compress,
            }))
        }
        fn add_points(&mut self, sensor_pos: PointF, points: Vec<PointF>) -> Vec<PointI> {
            self.0.add_points(sensor_pos, points.as_slice())
        }
        /// Multipliers used by the following `add_points` calls, voxels keep
        /// the values they got so far.
        fn set_multipliers(&mut self, multipliers: Multipliers) {
            self.0.set_multipliers(multipliers);
        }
        /// Copy of the multipliers used by `add_points`.
        fn get_multipliers(&self) -> Multipliers {
            self.0.multipliers()
        }
        /// Every voxel that is not 0, with its value.
        fn get_all_points(&self) -> Vec<PointI> {
            self.0.get_all_points()
        }
        /// Every constructed voxel, as groups holding at least one. Without
        /// `compress` a voxel counts as constructed once it reaches
        /// `construct_threshold`.
        fn get_all_points_compressed(&self) -> Vec<CompressedVoxelGroup> {
            self.0.get_all_points_compressed()
        }
    }

    #[cfg(feature = "multi_threaded")]
    #[pyo3::pyclass]
    #[derive(Debug)]
    struct MultiThreaded(crate::MultiThreaded);

    #[cfg(feature = "multi_threaded")]
    #[pyo3::pymethods]
    impl MultiThreaded {
        #[new]
        #[expect(clippy::too_many_arguments)]
        fn new(
            resolution: f32,
            intensity_multiplier: f32,
            hit_multiplier: u32,
            target_multiplier: u32,
            destroy_threshold: u32,
            construct_threshold: u32,
            max_value: u32,
            compress: bool,
            thread_count: usize,
        ) -> pyo3::PyResult<Self> {
            match crate::MultiThreaded::new(
                crate::VoxelizerParams {
                    resolution,
                    multipliers: crate::Multipliers {
                        intensity: intensity_multiplier,
                        hit: hit_multiplier,
                        target: target_multiplier,
                    },
                    destroy_threshold,
                    construct_threshold,
                    max_value,
                    compress,
                },
                thread_count,
            ) {
                Ok(v) => Ok(Self(v)),
                Err(err) => Err(pyo3::exceptions::PyException::new_err(err.to_string())),
            }
        }
        fn add_points(&mut self, sensor_pos: PointF, points: Vec<PointF>) -> Vec<PointI> {
            self.0.add_points(sensor_pos, points.as_slice())
        }
        /// Multipliers used by the following `add_points` calls, voxels keep
        /// the values they got so far.
        fn set_multipliers(&mut self, multipliers: Multipliers) {
            self.0.set_multipliers(multipliers);
        }
        /// Copy of the multipliers used by `add_points`.
        fn get_multipliers(&self) -> Multipliers {
            self.0.multipliers()
        }
        /// Every voxel that is not 0, with its value.
        fn get_all_points(&self) -> Vec<PointI> {
            self.0.get_all_points()
        }
        /// Every constructed voxel, as groups holding at least one. Without
        /// `compress` a voxel counts as constructed once it reaches
        /// `construct_threshold`.
        fn get_all_points_compressed(&self) -> Vec<CompressedVoxelGroup> {
            self.0.get_all_points_compressed()
        }
    }

    #[cfg(feature = "wgpu")]
    #[pyo3::pyclass]
    #[derive(Debug)]
    struct WgpuAccelerated(crate::WgpuAccelerated);

    #[cfg(feature = "wgpu")]
    #[pyo3::pymethods]
    impl WgpuAccelerated {
        #[new]
        #[expect(clippy::too_many_arguments)]
        fn new(
            resolution: f32,
            intensity_multiplier: f32,
            hit_multiplier: u32,
            target_multiplier: u32,
            destroy_threshold: u32,
            construct_threshold: u32,
            max_value: u32,
            compress: bool,
            bucket_count: u32,
        ) -> pyo3::PyResult<Self> {
            match pollster::block_on(crate::WgpuAccelerated::new(
                crate::VoxelizerParams {
                    resolution,
                    multipliers: crate::Multipliers {
                        intensity: intensity_multiplier,
                        hit: hit_multiplier,
                        target: target_multiplier,
                    },
                    destroy_threshold,
                    construct_threshold,
                    max_value,
                    compress,
                },
                bucket_count,
            )) {
                Some(v) => Ok(Self(v)),
                None => Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "no GPU adapter capable of running compute shaders, bucket_count is 0 \
                     or the hashmap does not fit the adapter's limits",
                )),
            }
        }
        fn add_points(&mut self, sensor_pos: PointF, points: Vec<PointF>) -> Vec<PointI> {
            self.0.add_points(sensor_pos, points.as_slice())
        }
        /// Multipliers used by the following `add_points` calls, voxels keep
        /// the values they got so far.
        fn set_multipliers(&mut self, multipliers: Multipliers) {
            self.0.set_multipliers(multipliers);
        }
        /// Copy of the multipliers used by `add_points`.
        fn get_multipliers(&self) -> Multipliers {
            self.0.multipliers()
        }
        /// Every voxel that is not 0, with its value.
        fn get_all_points(&self) -> Vec<PointI> {
            self.0.get_all_points()
        }
        /// Every constructed voxel, as groups holding at least one. Without
        /// `compress` a voxel counts as constructed once it reaches
        /// `construct_threshold`.
        fn get_all_points_compressed(&self) -> Vec<CompressedVoxelGroup> {
            self.0.get_all_points_compressed()
        }
        /// GPU memory taken by the hashmap of `bucket_count` buckets, in bytes.
        #[staticmethod]
        fn memory_size(bucket_count: u32) -> u64 {
            crate::WgpuAccelerated::memory_size(bucket_count)
        }
    }
}
