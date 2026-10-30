use wasm_bindgen::prelude::wasm_bindgen;

use crate::{CompressedVoxelGroup, Multipliers, PointF, PointI};

#[wasm_bindgen]
impl Multipliers {
    #[wasm_bindgen(constructor)]
    pub fn new(intensity: f32, hit: u32, target: u32) -> Self {
        Self {
            intensity,
            hit,
            target,
        }
    }
}

#[wasm_bindgen]
impl PointF {
    #[wasm_bindgen(constructor)]
    pub fn new(x: f32, y: f32, z: f32, i: f32) -> Self {
        Self { x, y, z, i }
    }
}

#[wasm_bindgen]
impl PointI {
    #[wasm_bindgen(constructor)]
    pub fn new(x: i32, y: i32, z: i32, i: u32) -> Self {
        Self { x, y, z, i }
    }
}

#[wasm_bindgen]
impl CompressedVoxelGroup {
    /// Words of the group, as a BigUint64Array.
    #[wasm_bindgen(getter)]
    pub fn values(&self) -> Vec<u64> {
        self.values.to_vec()
    }
}

#[cfg(any(feature = "single_threaded", feature = "wgpu"))]
#[wasm_bindgen(typescript_custom_section)]
const TYPES: &str = r#"
/** A point `add_points` takes: a `PointF` or a plain object. */
export interface PointFLike { x: number; y: number; z: number; i: number }
"#;

#[cfg(any(feature = "single_threaded", feature = "wgpu"))]
fn points_from_js(value: &wasm_bindgen::JsValue) -> Result<Vec<PointF>, wasm_bindgen::JsError> {
    use wasm_bindgen::{JsCast, JsError, JsValue};
    if !value.is_array() {
        if value.is_null_or_undefined() {
            return Err(JsError::new("`points` can't be null or undefined"));
        }
        return Err(JsError::new("`points` has to be an array"));
    }
    let field = |point: &JsValue, i: u32, key: &str| -> Result<f32, JsError> {
        js_sys::Reflect::get(point, &JsValue::from_str(key))
            .ok()
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .ok_or_else(|| JsError::new(&format!("points[{i}].{key} is not a number")))
    };
    value
        .unchecked_ref::<js_sys::Array>()
        .iter()
        .zip(0..)
        .map(|(v, i)| {
            Ok(PointF {
                x: field(&v, i, "x")?,
                y: field(&v, i, "y")?,
                z: field(&v, i, "z")?,
                i: field(&v, i, "i")?,
            })
        })
        .collect()
}

#[cfg(feature = "single_threaded")]
mod st {
    use super::points_from_js;
    use crate::Voxelizer;
    use crate::{CompressedVoxelGroup, Multipliers, PointF, PointI};
    use wasm_bindgen::prelude::wasm_bindgen;
    use wasm_bindgen::{JsError, JsValue};

    #[wasm_bindgen]
    pub struct SingleThreaded(crate::SingleThreaded);

    #[wasm_bindgen]
    impl SingleThreaded {
        #[wasm_bindgen(constructor)]
        #[expect(clippy::too_many_arguments)]
        pub fn new(
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
                multipliers: Multipliers {
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
        pub fn add_points(
            &mut self,
            sensor_pos: &PointF,
            #[wasm_bindgen(unchecked_param_type = "PointFLike[]")] points: JsValue,
        ) -> Result<Vec<PointI>, JsError> {
            Ok(self
                .0
                .add_points(sensor_pos.clone(), points_from_js(&points)?.as_slice()))
        }
        /// Multipliers used by the following `add_points` calls, voxels keep
        /// the values they got so far.
        pub fn set_multipliers(&mut self, multipliers: &Multipliers) {
            self.0.set_multipliers(*multipliers);
        }
        /// Copy of the multipliers used by `add_points`.
        pub fn get_multipliers(&self) -> Multipliers {
            self.0.multipliers()
        }
        /// Every voxel that is not 0, with its value.
        pub fn get_all_points(&self) -> Vec<PointI> {
            self.0.get_all_points()
        }
        /// Every constructed voxel, as groups holding at least one. Without
        /// `compress` a voxel counts as constructed once it reaches
        /// `construct_threshold`.
        pub fn get_all_points_compressed(&self) -> Vec<CompressedVoxelGroup> {
            self.0.get_all_points_compressed()
        }
    }
}

#[cfg(feature = "wgpu")]
mod gpu {
    use std::cell::Cell;

    use super::points_from_js;
    use crate::Voxelizer;
    use crate::{CompressedVoxelGroup, Multipliers, PointF, PointI};
    use wasm_bindgen::prelude::wasm_bindgen;
    use wasm_bindgen::{JsError, JsValue};

    /// Runs on WebGPU. Created with the async `create`, `add_points` and the
    /// dumps return a Promise and a call has to finish before the next one
    /// starts.
    ///
    /// The voxelizer is taken out while a call runs, so an overlapping call
    /// finds none and rejects. Borrowing `&mut self` across the await instead
    /// would make wasm-bindgen throw from inside the future, which surfaces as
    /// an uncaught error rather than a rejection.
    #[wasm_bindgen]
    pub struct WgpuAccelerated(Cell<Option<crate::WgpuAccelerated>>);

    impl WgpuAccelerated {
        /// Takes the voxelizer out, it has to be put back once `what` is done.
        fn take(&self, what: &str) -> Result<crate::WgpuAccelerated, JsError> {
            self.0.take().ok_or_else(|| {
                JsError::new(&format!(
                    "another call is still running, it has to finish before {what}"
                ))
            })
        }
    }

    #[wasm_bindgen]
    impl WgpuAccelerated {
        /// Rejects when there is no WebGPU adapter capable of running compute
        /// shaders, `bucket_count` is 0 or the hashmap does not fit the
        /// adapter's limits.
        #[expect(clippy::too_many_arguments)]
        pub async fn create(
            resolution: f32,
            intensity_multiplier: f32,
            hit_multiplier: u32,
            target_multiplier: u32,
            destroy_threshold: u32,
            construct_threshold: u32,
            max_value: u32,
            compress: bool,
            bucket_count: u32,
        ) -> Result<WgpuAccelerated, JsError> {
            crate::WgpuAccelerated::new(
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
                },
                bucket_count,
            )
            .await
            .map(|v| Self(Cell::new(Some(v))))
            .ok_or_else(|| {
                JsError::new(
                    "no WebGPU adapter capable of running compute shaders, bucket_count is 0 \
                     or the hashmap does not fit the adapter's limits",
                )
            })
        }
        pub async fn add_points(
            &self,
            sensor_pos: &PointF,
            #[wasm_bindgen(unchecked_param_type = "PointFLike[]")] points: JsValue,
        ) -> Result<Vec<PointI>, JsError> {
            let points = points_from_js(&points)?;
            let mut voxelizer = self.take("add_points")?;
            let update = voxelizer
                .add_points_async(sensor_pos.clone(), points.as_slice())
                .await;
            self.0.set(Some(voxelizer));
            Ok(update)
        }
        /// Multipliers used by the following `add_points` calls, voxels keep
        /// the values they got so far. Throws while another call is running.
        pub fn set_multipliers(&self, multipliers: &Multipliers) -> Result<(), JsError> {
            let mut voxelizer = self.take("set_multipliers")?;
            voxelizer.set_multipliers(*multipliers);
            self.0.set(Some(voxelizer));
            Ok(())
        }
        /// Copy of the multipliers used by `add_points`. Throws while another
        /// call is running.
        pub fn get_multipliers(&self) -> Result<Multipliers, JsError> {
            let voxelizer = self.take("get_multipliers")?;
            let multipliers = voxelizer.multipliers();
            self.0.set(Some(voxelizer));
            Ok(multipliers)
        }
        /// Every voxel that is not 0, with its value. Rejects while another
        /// call is running.
        pub async fn get_all_points(&self) -> Result<Vec<PointI>, JsError> {
            let voxelizer = self.take("get_all_points")?;
            let points = voxelizer.get_all_points_async().await;
            self.0.set(Some(voxelizer));
            Ok(points)
        }
        /// Every constructed voxel, as groups holding at least one. Without
        /// `compress` a voxel counts as constructed once it reaches
        /// `construct_threshold`. Rejects while another call is running.
        pub async fn get_all_points_compressed(
            &self,
        ) -> Result<Vec<CompressedVoxelGroup>, JsError> {
            let voxelizer = self.take("get_all_points_compressed")?;
            let groups = voxelizer.get_all_points_compressed_async().await;
            self.0.set(Some(voxelizer));
            Ok(groups)
        }
        /// GPU memory taken by the hashmap of `bucket_count` buckets, in
        /// bytes. Exact as a number, it stays below 2 ** 53.
        pub fn memory_size(bucket_count: u32) -> f64 {
            crate::WgpuAccelerated::memory_size(bucket_count) as f64
        }
    }
}
