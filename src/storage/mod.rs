use crate::geometry::PointI;

pub trait Storage {
    fn add_point(&self, point: PointI);
    fn remove_point(&self, point: PointI);
    fn iter(&self) -> Box<dyn Iterator<Item = &PointI>>;
    fn iter_pack(&self) -> Box<dyn Iterator<Item = u64>>;
    fn get_point_pack(&self, point: PointI) -> u64;
    #[inline]
    fn get_point_pack_u32(&self, point: PointI) -> u32 {
        let z_lsb = ((point.z & 3) > 1) as u64 * 32;
        (self.get_point_pack(point) >> z_lsb) as u32
    }
    #[inline]
    fn iter_pack32(&self) -> Box<dyn Iterator<Item = u32>> {
        Box::new(
            self.iter_pack()
                .flat_map(|v| [v as u32, (v >> 32) as u32].into_iter()),
        )
    }
}
