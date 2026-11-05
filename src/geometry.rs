use std::ops::{Div, DivAssign, SubAssign};

#[cfg_attr(feature = "js", wasm_bindgen::prelude::wasm_bindgen)]
#[cfg_attr(
    feature = "python",
    pyo3::pyclass(from_py_object, module = "r3v_voxelizer")
)]
#[repr(C)]
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct PointF {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub i: f32,
}

impl PointF {
    #[inline]
    pub fn len(&self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }
}

impl SubAssign<&Self> for PointF {
    fn sub_assign(&mut self, rhs: &Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
        self.z -= rhs.z;
    }
}

impl Div<f32> for &PointF {
    type Output = PointF;

    fn div(self, rhs: f32) -> Self::Output {
        PointF {
            x: self.x / rhs,
            y: self.y / rhs,
            z: self.z / rhs,
            i: self.i,
        }
    }
}

impl DivAssign<f32> for PointF {
    #[inline]
    fn div_assign(&mut self, rhs: f32) {
        self.x /= rhs;
        self.y /= rhs;
        self.z /= rhs;
    }
}

#[cfg_attr(feature = "js", wasm_bindgen::prelude::wasm_bindgen)]
#[cfg_attr(
    feature = "python",
    pyo3::pyclass(from_py_object, module = "r3v_voxelizer")
)]
#[repr(C)]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PointI {
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub i: u32,
}

#[cfg(any(
    feature = "single_threaded",
    feature = "multi_threaded",
    feature = "wgpu"
))]
mod zorder_mod {
    /// Bits of one coordinate in a z-order index: bit k of x goes to bit 3k,
    /// y and z sit 1 and 2 above (`AXIS_MASK << 1`, `AXIS_MASK << 2`).
    const AXIS_MASK: usize = 0x249;

    /// Spreads the 4 low bits of `v` 3 apart (bit k to bit 3k).
    const fn spread(v: u16) -> u16 {
        let mut result = 0;
        let mut bit = 0;
        while bit < 4 {
            result |= ((v >> bit) & 1) << (3 * bit);
            bit += 1;
        }
        result
    }

    const fn spread_table(shift: u32) -> [u16; 16] {
        let mut table = [0; 16];
        let mut v = 0;
        while v < 16 {
            table[v] = spread(v as u16) << shift;
            v += 1;
        }
        table
    }

    /// Inverse of [`spread`], indexed by the bits of one axis
    /// (`index & AXIS_MASK`). Only 16 entries are used.
    const fn compact_table() -> [u8; AXIS_MASK + 1] {
        let mut table = [0; AXIS_MASK + 1];
        let mut v = 0;
        while v < 16 {
            table[spread(v) as usize] = v as u8;
            v += 1;
        }
        table
    }

    static SPREAD_X: [u16; 16] = spread_table(0);
    static SPREAD_Y: [u16; 16] = spread_table(1);
    static SPREAD_Z: [u16; 16] = spread_table(2);
    static COMPACT: [u8; AXIS_MASK + 1] = compact_table();

    /// Z-order (Morton) index of 3 coordinates of up to 4 bits each: x in bit
    /// 0, y in bit 1, z in bit 2, then the next bit of each. The same layout
    /// as `get_z_order` in the shader and the words of
    /// [`crate::CompressedVoxelGroup`].
    ///
    /// Lookup tables instead of BMI2 `pdep`/`pext`: those are behind
    /// `#[target_feature]`, which is never inlined, and a call costs more
    /// than a few loads.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    pub struct ZOrder {
        _private: (),
    }

    impl ZOrder {
        /// Coordinates of a z-order index below 4096.
        #[inline]
        pub fn coord_of(&self, index: u64) -> [u16; 3] {
            debug_assert!(index < 4096, "z-order index {index} has more than 12 bits");
            let index = index as usize;
            [
                COMPACT[index & AXIS_MASK] as u16,
                COMPACT[(index >> 1) & AXIS_MASK] as u16,
                COMPACT[(index >> 2) & AXIS_MASK] as u16,
            ]
        }
        /// Z-order index of coordinates below 16.
        #[inline]
        pub fn index_of(&self, [x, y, z]: [u16; 3]) -> u64 {
            debug_assert!(
                x < 16 && y < 16 && z < 16,
                "z-order coordinates {x}, {y}, {z} have more than 4 bits"
            );
            (SPREAD_X[(x & 15) as usize]
                | SPREAD_Y[(y & 15) as usize]
                | SPREAD_Z[(z & 15) as usize]) as u64
        }
    }

    #[cfg(test)]
    mod test {
        use super::ZOrder;

        /// Bit by bit reference of the layout.
        fn reference([x, y, z]: [u16; 3]) -> u64 {
            (0..4).fold(0, |index, bit| {
                index
                    | (((x >> bit) & 1) as u64) << (3 * bit)
                    | (((y >> bit) & 1) as u64) << (3 * bit + 1)
                    | (((z >> bit) & 1) as u64) << (3 * bit + 2)
            })
        }

        #[test]
        fn zorder_matches_reference_on_every_input() {
            let zorder = ZOrder::default();
            for x in 0..16 {
                for y in 0..16 {
                    for z in 0..16 {
                        let index = zorder.index_of([x, y, z]);
                        assert_eq!(index, reference([x, y, z]), "index_of {x} {y} {z}");
                        assert_eq!(zorder.coord_of(index), [x, y, z], "coord_of {index}");
                    }
                }
            }
        }

        #[test]
        fn zorder_axis_order() {
            let zorder = ZOrder::default();
            assert_eq!(zorder.index_of([1, 0, 0]), 1);
            assert_eq!(zorder.index_of([0, 1, 0]), 2);
            assert_eq!(zorder.index_of([0, 0, 1]), 4);
            assert_eq!(zorder.index_of([2, 0, 0]), 8);
            assert_eq!(zorder.index_of([15, 15, 15]), 4095);
            assert_eq!(zorder.coord_of(63), [3, 3, 3]);
        }
    }
}
#[cfg(any(
    feature = "single_threaded",
    feature = "multi_threaded",
    feature = "wgpu"
))]
pub use zorder_mod::ZOrder;
