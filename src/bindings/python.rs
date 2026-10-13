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

    #[pymodule_export]
    use crate::{PointF, PointI};

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
                intensity_multiplier,
                hit_multiplier,
                target_multiplier,
                destroy_threshold,
                construct_threshold,
                max_value,
                compress,
            }))
        }
        fn add_points(&mut self, sensor_pos: PointF, points: Vec<PointF>) -> Vec<PointI> {
            self.0.add_points(sensor_pos, points.as_slice())
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
                    intensity_multiplier,
                    hit_multiplier,
                    target_multiplier,
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
                    intensity_multiplier,
                    hit_multiplier,
                    target_multiplier,
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
        /// GPU memory taken by the hashmap of `bucket_count` buckets, in bytes.
        #[staticmethod]
        fn memory_size(bucket_count: u32) -> u64 {
            crate::WgpuAccelerated::memory_size(bucket_count)
        }
    }
}
