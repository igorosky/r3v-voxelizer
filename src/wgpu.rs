use crate::{
    Voxelizer,
    geometry::{PointF, PointI},
};

#[derive(Debug, Clone)]
struct WgpuAccelerated {}

impl Voxelizer for WgpuAccelerated {
    fn add_points(&mut self, sensor_pos: PointF, points: &[PointF]) -> Vec<PointI> {
        todo!()
    }
}

impl WgpuAccelerated {
    pub fn new() -> Option<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .ok()?;
        if !adapter
            .get_downlevel_capabilities()
            .flags
            .contains(wgpu::DownlevelFlags::COMPUTE_SHADERS)
        {
            return None;
        }
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .ok()?;
        todo!()
    }
}
