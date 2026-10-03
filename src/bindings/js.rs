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

#[cfg(feature = "single_threaded")]
mod st {
    use crate::Voxelizer;
    use crate::{PointF, PointI};
    use wasm_bindgen::prelude::wasm_bindgen;
    use wasm_bindgen::{JsError, JsValue};

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
            compress: bool,
        ) -> Self {
            Self(crate::SingleThreaded::new(
                resolution,
                hit_multiplier,
                target_multiplier,
                destroy_threshold,
                construct_threshold,
                compress,
            ))
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
