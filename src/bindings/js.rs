use wasm_bindgen::prelude::wasm_bindgen;

use crate::{PointF, PointI};

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
    use crate::{PointF, PointI};
    use wasm_bindgen::prelude::wasm_bindgen;
    use wasm_bindgen::{JsError, JsValue};

    #[wasm_bindgen]
    pub struct SingleThreaded(crate::SingleThreaded);

    #[wasm_bindgen]
    impl SingleThreaded {
        #[wasm_bindgen(constructor)]
        pub fn new(
            resolution: f32,
            hit_multiplier: u32,
            target_multiplier: u32,
            destroy_threshold: u32,
            construct_threshold: u32,
            max_value: u32,
            compress: bool,
        ) -> Self {
            Self(crate::SingleThreaded::new(crate::VoxelizerParams {
                resolution,
                hit_multiplier,
                target_multiplier,
                destroy_threshold,
                construct_threshold,
                max_value,
                compress,
            }))
        }
        pub fn add_points(
            &mut self,
            sensor_pos: &PointF,
            points: JsValue,
        ) -> Result<Vec<PointI>, JsError> {
            Ok(self
                .0
                .add_points(sensor_pos.clone(), points_from_js(&points)?.as_slice()))
        }
    }
}

#[cfg(feature = "wgpu")]
mod gpu {
    use std::cell::Cell;

    use super::points_from_js;
    use crate::{PointF, PointI};
    use wasm_bindgen::prelude::wasm_bindgen;
    use wasm_bindgen::{JsError, JsValue};

    /// Runs on WebGPU. Created with the async `create`, `add_points` returns
    /// a Promise and a call has to finish before the next one starts.
    ///
    /// The voxelizer is taken out while a call runs, so an overlapping call
    /// finds none and rejects. Borrowing `&mut self` across the await instead
    /// would make wasm-bindgen throw from inside the future, which surfaces as
    /// an uncaught error rather than a rejection.
    #[wasm_bindgen]
    pub struct WgpuAccelerated(Cell<Option<crate::WgpuAccelerated>>);

    #[wasm_bindgen]
    impl WgpuAccelerated {
        /// Rejects when there is no WebGPU adapter capable of running compute
        /// shaders, `bucket_count` is 0 or the hashmap does not fit the
        /// adapter's limits.
        #[expect(clippy::too_many_arguments)]
        pub async fn create(
            resolution: f32,
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
                    hit_multiplier,
                    target_multiplier,
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
            points: JsValue,
        ) -> Result<Vec<PointI>, JsError> {
            let points = points_from_js(&points)?;
            let mut voxelizer = self.0.take().ok_or_else(|| {
                JsError::new(
                    "add_points is still running, a call has to finish before the next one",
                )
            })?;
            let update = voxelizer
                .add_points_async(sensor_pos.clone(), points.as_slice())
                .await;
            self.0.set(Some(voxelizer));
            Ok(update)
        }
        /// GPU memory taken by the hashmap of `bucket_count` buckets, in
        /// bytes. Exact as a number, it stays below 2 ** 53.
        pub fn memory_size(bucket_count: u32) -> f64 {
            crate::WgpuAccelerated::memory_size(bucket_count) as f64
        }
    }
}
