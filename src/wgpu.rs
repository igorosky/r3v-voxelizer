use crate::{
    Voxelizer,
    geometry::{PointF, PointI},
};

#[derive(Debug, Clone)]
pub struct WgpuAccelerated {
    has_uma: bool,
}

impl Voxelizer for WgpuAccelerated {
    fn add_points(&mut self, sensor_pos: PointF, points: &[PointF]) -> Vec<PointI> {
        todo!()
    }
}

impl WgpuAccelerated {
    pub async fn new() -> Option<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .ok()?;
        if !adapter
            .get_downlevel_capabilities()
            .flags
            .contains(wgpu::DownlevelFlags::COMPUTE_SHADERS)
        {
            return None;
        }
        let has_uma = {
            use wgpu::DeviceType;
            match adapter.get_info().device_type {
                DeviceType::Cpu | DeviceType::IntegratedGpu => true,
                DeviceType::DiscreteGpu | DeviceType::VirtualGpu | DeviceType::Other => false,
            }
        };
        let mut required_features = wgpu::Features::empty();
        if has_uma {
            required_features |= wgpu::Features::MAPPABLE_PRIMARY_BUFFERS;
        }
        // WARN: currently uma is not utilized (only request feature, but does nothing more with it)
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features,
                required_limits: wgpu::Limits::downlevel_defaults(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .ok()?;
        let shader = device.create_shader_module(wgpu::include_wgsl!("wgpu_shader.wgsl"));
        let cpu_input_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 1024,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::MAP_WRITE,
            mapped_at_creation: false,
        });
        let input_points_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 1024, // TODO
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        todo!()
    }
}
