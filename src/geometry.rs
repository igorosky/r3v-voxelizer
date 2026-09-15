use std::ops::{DivAssign, SubAssign};

use zorder::{
    Deinterleave, Interleave,
    bmi2::{self, DeinterleaveBMI2, HardwareSupportToken, InterleaveBMI2},
    coord_of, index_of,
};

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

impl DivAssign<f32> for PointF {
    #[inline]
    fn div_assign(&mut self, rhs: f32) {
        self.x /= rhs;
        self.y /= rhs;
        self.z /= rhs;
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PointI {
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub i: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ZOrder {
    token: Option<HardwareSupportToken>,
}

impl Default for ZOrder {
    #[inline]
    fn default() -> Self {
        Self {
            token: HardwareSupportToken::new(),
        }
    }
}

impl ZOrder {
    #[inline]
    pub fn coord_of<I, const N: usize>(&self, index: I) -> [<I as Deinterleave<N>>::Output; N]
    where
        I: DeinterleaveBMI2<N> + Copy,
    {
        if let Some(token) = self.token {
            bmi2::coord_of(index, token)
        } else {
            coord_of(index)
        }
    }
    #[inline]
    pub fn index_of<I, const N: usize>(&self, array: [I; N]) -> <I as Interleave<N>>::Output
    where
        I: InterleaveBMI2<N>,
    {
        if let Some(token) = self.token {
            bmi2::index_of(array, token)
        } else {
            index_of(array)
        }
    }
}
