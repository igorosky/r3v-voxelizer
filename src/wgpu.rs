use crate::{
    Voxelizer,
    geometry::{PointF, PointI},
};

// Must match the shader
const WORKGROUP_SIZE: u32 = 256;
const BITS_FOR_ENTRY_VOXEL_COUNT: u32 = 2;
const ENTRY_VOXEL_COUNT: u64 = 1 << (3 * BITS_FOR_ENTRY_VOXEL_COUNT);
const BUCKET_SIZE: u64 = 2;
/// `Entry`: x, y, z, offset, flags, occupancy[2]
const ENTRY_BYTES: u64 = 7 * 4;
/// `Bucket`: lock + entries
const BUCKET_BYTES: u64 = 4 + BUCKET_SIZE * ENTRY_BYTES;
/// `Voxel`: value + prev
const VOXEL_BYTES: u64 = 2 * 4;
/// `Altered` header: len
const ALTERED_HEADER_BYTES: u64 = 4;
/// `Output` header: len + error, padded to the alignment of `vec4i`
const OUTPUT_HEADER_BYTES: u64 = 16;
/// `Params`
const PARAMS_BYTES: u64 = 12 * 4;
/// Input (`vec4u`) and output (`vec4i`) element
const POINT_BYTES: u64 = 16;
/// Capacity (in points) the input and output buffers start with
const INITIAL_CAPACITY: u64 = 1024;

#[derive(Debug, Clone, Copy)]
enum Kernel {
    RayMarch,
    AddPoint,
    Collect,
    SetVoxels,
}

impl Kernel {
    const ALL: [Kernel; 4] = [
        Kernel::RayMarch,
        Kernel::AddPoint,
        Kernel::Collect,
        Kernel::SetVoxels,
    ];
    fn entry_point(self) -> &'static str {
        match self {
            Kernel::RayMarch => "ray_march",
            Kernel::AddPoint => "add_point",
            Kernel::Collect => "collect",
            Kernel::SetVoxels => "set_voxels",
        }
    }
}

/// Voxelizer running on the GPU.
///
/// Voxels live in a fixed size hashmap of `bucket_count` buckets, each one
/// holding 2 pages of 4x4x4 voxels. A bucket takes about 1 KiB of GPU memory
/// (see [`WgpuAccelerated::memory_size`]). Values of pages that do not fit
/// anymore are dropped.
#[derive(Debug)]
pub struct WgpuAccelerated {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: [wgpu::ComputePipeline; 4],
    io_layout: wgpu::BindGroupLayout,
    io_bind_group: wgpu::BindGroup,
    storage_bind_group: wgpu::BindGroup,
    params: wgpu::Buffer,
    input: wgpu::Buffer,
    altered: wgpu::Buffer,
    output: wgpu::Buffer,
    // Used through `storage_bind_group`, the fields are read back by the tests
    #[cfg_attr(not(test), expect(dead_code))]
    buckets: wgpu::Buffer,
    #[cfg_attr(not(test), expect(dead_code))]
    voxels: wgpu::Buffer,
    max_binding_size: u64,
    max_workgroups: u32,
    resolution: f32,
    hit_multiplier: u32,
    target_multiplier: u32,
    destroy_threshold: u32,
    construct_threshold: u32,
    compress: bool,
}

impl Voxelizer for WgpuAccelerated {
    fn add_points(&mut self, sensor_pos: PointF, points: &[PointF]) -> Vec<PointI> {
        let camera = [sensor_pos.x, sensor_pos.y, sensor_pos.z];
        let input: Vec<[u32; 4]> = points
            .iter()
            .map(|p| [p.x.to_bits(), p.y.to_bits(), p.z.to_bits(), p.i.to_bits()])
            .collect();
        // Every ray has to be marched before any point is added. When the
        // input fits one dispatch it is uploaded once for both kernels.
        let chunks: Vec<_> = input.chunks(self.max_elements(0)).collect();
        let passes: &[&[Kernel]] = if chunks.len() <= 1 {
            &[&[Kernel::RayMarch, Kernel::AddPoint]]
        } else {
            &[&[Kernel::RayMarch], &[Kernel::AddPoint]]
        };
        self.queue
            .write_buffer(&self.altered, 0, &0u32.to_ne_bytes());
        for kernels in passes {
            for chunk in &chunks {
                self.upload_input(chunk);
                self.write_params(camera, chunk.len() as u32, 0);
                let mut encoder = self.encoder();
                for kernel in *kernels {
                    self.dispatch(&mut encoder, *kernel, chunk.len() as u32);
                }
                self.queue.submit([encoder.finish()]);
            }
        }

        let altered = self.read_words(self.encoder(), &self.altered, 0, 1)[0];
        let per_entry = if self.compress {
            2
        } else {
            ENTRY_VOXEL_COUNT as u32
        };
        let entries_per_chunk = (self.max_elements(OUTPUT_HEADER_BYTES) / per_entry as usize)
            .max(1)
            .min(u32::MAX as usize) as u32;
        let mut result = Vec::new();
        for first in (0..altered).step_by(entries_per_chunk as usize) {
            let count = entries_per_chunk.min(altered - first);
            self.reserve_output(u64::from(count * per_entry));
            self.queue.write_buffer(&self.output, 0, &[0; 8]);
            self.write_params([0.0; 3], count, first);
            let mut encoder = self.encoder();
            self.dispatch(&mut encoder, Kernel::Collect, count * 2);
            let [len, error] = self.read_words(encoder, &self.output, 0, 2)[..] else {
                unreachable!("2 words are read");
            };
            if error != 0 {
                panic!("It must have existed if it is destroyed");
            }
            if len != 0 {
                let data = self.read_back(
                    self.encoder(),
                    &self.output,
                    OUTPUT_HEADER_BYTES,
                    u64::from(len) * POINT_BYTES,
                );
                result.extend(
                    bytemuck::pod_collect_to_vec::<u8, [u32; 4]>(&data)
                        .into_iter()
                        .map(|[x, y, z, i]| PointI {
                            x: x as i32,
                            y: y as i32,
                            z: z as i32,
                            i,
                        }),
                );
            }
        }
        result
    }
}

impl WgpuAccelerated {
    /// GPU memory taken by the hashmap of `bucket_count` buckets, in bytes.
    pub fn memory_size(bucket_count: u32) -> u64 {
        let entry_count = u64::from(bucket_count) * BUCKET_SIZE;
        u64::from(bucket_count) * BUCKET_BYTES
            + entry_count * ENTRY_VOXEL_COUNT * VOXEL_BYTES
            + ALTERED_HEADER_BYTES
            + entry_count * 4
    }

    /// Returns `None` when there is no adapter capable of running compute
    /// shaders, `bucket_count` is 0 or the hashmap does not fit the adapter's
    /// limits.
    pub async fn new(
        resolution: f32,
        hit_multiplier: u32,
        target_multiplier: u32,
        destroy_threshold: u32,
        construct_threshold: u32,
        compress: bool,
        bucket_count: u32,
    ) -> Option<Self> {
        let entry_count = u64::from(bucket_count) * BUCKET_SIZE;
        let voxel_count = entry_count * ENTRY_VOXEL_COUNT;
        // Voxels are indexed with u32 in the shader
        if bucket_count == 0 || voxel_count > 1 << 32 {
            return None;
        }
        let buckets_size = u64::from(bucket_count) * BUCKET_BYTES;
        let voxels_size = voxel_count * VOXEL_BYTES;
        let altered_size = ALTERED_HEADER_BYTES + entry_count * 4;

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
        let adapter_limits = adapter.limits();
        let max_binding_size = adapter_limits
            .max_storage_buffer_binding_size
            .min(adapter_limits.max_buffer_size);
        if buckets_size.max(voxels_size).max(altered_size) > max_binding_size {
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
                required_limits: wgpu::Limits {
                    max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size,
                    max_buffer_size: adapter_limits.max_buffer_size,
                    ..wgpu::Limits::default()
                },
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .ok()?;
        let shader = device.create_shader_module(wgpu::include_wgsl!("wgpu_shader.wgsl"));

        let layout_entry = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let storage = |read_only| wgpu::BufferBindingType::Storage { read_only };
        let io_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("io"),
            entries: &[
                layout_entry(0, wgpu::BufferBindingType::Uniform),
                layout_entry(1, storage(true)),
                layout_entry(2, storage(false)),
                layout_entry(3, storage(false)),
            ],
        });
        let storage_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hashmap"),
            entries: &[
                layout_entry(0, storage(false)),
                layout_entry(1, storage(false)),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&io_layout), Some(&storage_layout)],
            immediate_size: 0,
        });
        let constants = [("BUCKET_COUNT", f64::from(bucket_count))];
        let pipelines = Kernel::ALL.map(|kernel| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(kernel.entry_point()),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(kernel.entry_point()),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &constants,
                    ..Default::default()
                },
                cache: None,
            })
        });

        use wgpu::BufferUsages as Usage;
        let buffer = |label, size, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let params = buffer("params", PARAMS_BYTES, Usage::UNIFORM | Usage::COPY_DST);
        let input = Self::input_buffer(&device, INITIAL_CAPACITY);
        let output = Self::output_buffer(&device, INITIAL_CAPACITY);
        let readable = Usage::STORAGE | Usage::COPY_SRC | Usage::COPY_DST;
        let altered = buffer("altered", altered_size, readable);
        // Zero initialized, which is an empty hashmap
        let buckets = buffer("buckets", buckets_size, readable);
        let voxels = buffer("voxels", voxels_size, readable);
        let io_bind_group =
            Self::io_bind_group(&device, &io_layout, &params, &input, &altered, &output);
        let storage_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hashmap"),
            layout: &storage_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buckets.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: voxels.as_entire_binding(),
                },
            ],
        });
        let device_limits = device.limits();
        Some(Self {
            max_binding_size: device_limits
                .max_storage_buffer_binding_size
                .min(device_limits.max_buffer_size),
            max_workgroups: device_limits.max_compute_workgroups_per_dimension,
            device,
            queue,
            pipelines,
            io_layout,
            io_bind_group,
            storage_bind_group,
            params,
            input,
            altered,
            output,
            buckets,
            voxels,
            resolution,
            hit_multiplier,
            target_multiplier,
            destroy_threshold,
            construct_threshold,
            compress,
        })
    }

    fn input_buffer(device: &wgpu::Device, capacity: u64) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("input"),
            size: capacity * POINT_BYTES,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn output_buffer(device: &wgpu::Device, capacity: u64) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output"),
            size: OUTPUT_HEADER_BYTES + capacity * POINT_BYTES,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn io_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        params: &wgpu::Buffer,
        input: &wgpu::Buffer,
        altered: &wgpu::Buffer,
        output: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("io"),
            layout,
            entries: &[params, input, altered, output]
                .into_iter()
                .enumerate()
                .map(|(binding, buffer)| wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource: buffer.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        })
    }

    fn rebind_io(&mut self) {
        self.io_bind_group = Self::io_bind_group(
            &self.device,
            &self.io_layout,
            &self.params,
            &self.input,
            &self.altered,
            &self.output,
        );
    }

    /// How many points fit one binding after a header of `header` bytes.
    fn max_elements(&self, header: u64) -> usize {
        ((self.max_binding_size - header) / POINT_BYTES).min(u32::MAX.into()) as usize
    }

    fn upload_input(&mut self, input: &[[u32; 4]]) {
        let needed = input.len() as u64 * POINT_BYTES;
        if needed > self.input.size() {
            let capacity = (input.len() as u64)
                .next_power_of_two()
                .min(self.max_elements(0) as u64);
            self.input = Self::input_buffer(&self.device, capacity);
            self.rebind_io();
        }
        self.queue
            .write_buffer(&self.input, 0, bytemuck::cast_slice(input));
    }

    fn reserve_output(&mut self, capacity: u64) {
        if OUTPUT_HEADER_BYTES + capacity * POINT_BYTES > self.output.size() {
            let capacity = capacity
                .next_power_of_two()
                .min(self.max_elements(OUTPUT_HEADER_BYTES) as u64);
            self.output = Self::output_buffer(&self.device, capacity);
            self.rebind_io();
        }
    }

    /// `count` and `first` are the shader's `Params::count` and `Params::first`.
    fn write_params(&self, camera: [f32; 3], count: u32, first: u32) {
        let params = [
            camera[0].to_bits(),
            camera[1].to_bits(),
            camera[2].to_bits(),
            self.resolution.to_bits(),
            self.hit_multiplier,
            self.target_multiplier,
            self.destroy_threshold,
            self.construct_threshold,
            count,
            first,
            u32::from(self.compress),
            0,
        ];
        self.queue
            .write_buffer(&self.params, 0, bytemuck::cast_slice(&params));
    }

    fn encoder(&self) -> wgpu::CommandEncoder {
        self.device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default())
    }

    /// Runs `kernel` on `threads` threads, split into two dimensions when
    /// there are more workgroups than one dimension allows.
    fn dispatch(&self, encoder: &mut wgpu::CommandEncoder, kernel: Kernel, threads: u32) {
        let workgroups = threads.div_ceil(WORKGROUP_SIZE);
        if workgroups == 0 {
            return;
        }
        let x = workgroups.min(self.max_workgroups);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(kernel.entry_point()),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipelines[kernel as usize]);
        pass.set_bind_group(0, &self.io_bind_group, &[]);
        pass.set_bind_group(1, &self.storage_bind_group, &[]);
        pass.dispatch_workgroups(x, workgroups.div_ceil(x), 1);
    }

    /// Submits `encoder` and returns `size` bytes of `source` starting at
    /// `offset` once everything submitted so far has finished.
    fn read_back(
        &self,
        mut encoder: wgpu::CommandEncoder,
        source: &wgpu::Buffer,
        offset: u64,
        size: u64,
    ) -> Vec<u8> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(source, offset, &staging, 0, size);
        self.queue.submit([encoder.finish()]);
        // A failed mapping is reported by `get_mapped_range`
        staging.map_async(wgpu::MapMode::Read, .., |_| ());
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("waiting for the GPU failed");
        let data = staging
            .get_mapped_range(..)
            .expect("mapping the GPU buffer failed")
            .to_vec();
        staging.unmap();
        data
    }

    fn read_words(
        &self,
        encoder: wgpu::CommandEncoder,
        source: &wgpu::Buffer,
        offset: u64,
        count: u64,
    ) -> Vec<u32> {
        bytemuck::pod_collect_to_vec(&self.read_back(encoder, source, offset, count * 4))
    }
}

#[cfg(test)]
mod test {
    use std::collections::BTreeMap;

    use super::{
        BITS_FOR_ENTRY_VOXEL_COUNT, BUCKET_BYTES, BUCKET_SIZE, ENTRY_BYTES, ENTRY_VOXEL_COUNT,
        Kernel, VOXEL_BYTES, WgpuAccelerated,
    };
    use crate::{
        Voxelizer,
        geometry::{PointI, ZOrder},
        test::{
            COMPRESSED_PAGE_LEN, Coord, PAGE_LEN, PagedIndex, TestVoxelizer, cube_faces,
            from_above, get_voxel, nonzero_voxels, pf, pi, seed, sorted, with_i,
        },
    };

    /// Hashmap size used by the shared suite: 8192 pages.
    const TEST_BUCKETS: u32 = 1 << 12;
    const BITS_FOR_COMPRESSED_ZORDER: u32 = 4;
    const BITS_FOR_COMPRESSED_ZORDER_MASK: i32 = 2_i32.pow(BITS_FOR_COMPRESSED_ZORDER) - 1;
    // Must match the shader
    const ENTRY_OCCUPIED_MASK: u32 = 1 << 31;
    const ENTRY_TOUCHED: u32 = 1;
    const ENTRY_CONSTRUCTED: u32 = 2;

    fn new(
        resolution: f32,
        hit_multiplier: u32,
        target_multiplier: u32,
        destroy_threshold: u32,
        construct_threshold: u32,
        compress: bool,
        bucket_count: u32,
    ) -> WgpuAccelerated {
        pollster::block_on(WgpuAccelerated::new(
            resolution,
            hit_multiplier,
            target_multiplier,
            destroy_threshold,
            construct_threshold,
            compress,
            bucket_count,
        ))
        .expect("a GPU adapter with compute shaders is required")
    }

    /// Occupied hashmap entry (page) read back from the GPU.
    struct Entry {
        page: Coord,
        flags: u32,
        occupancy: u64,
        voxels: [u32; PAGE_LEN],
    }

    impl WgpuAccelerated {
        fn set_voxels(&mut self, voxels: &[(i32, i32, i32, u32)]) {
            let input: Vec<[u32; 4]> = voxels
                .iter()
                .map(|&(x, y, z, value)| [x as u32, y as u32, z as u32, value])
                .collect();
            self.upload_input(&input);
            self.write_params([0.0; 3], input.len() as u32, 0);
            let mut encoder = self.encoder();
            self.dispatch(&mut encoder, Kernel::SetVoxels, input.len() as u32);
            self.queue.submit([encoder.finish()]);
        }

        fn entries(&self) -> Vec<Entry> {
            let buckets: Vec<u32> = bytemuck::pod_collect_to_vec(&self.read_back(
                self.encoder(),
                &self.buckets,
                0,
                self.buckets.size(),
            ));
            let voxels: Vec<u32> = bytemuck::pod_collect_to_vec(&self.read_back(
                self.encoder(),
                &self.voxels,
                0,
                self.voxels.size(),
            ));
            let bucket_words = (BUCKET_BYTES / 4) as usize;
            let entry_words = (ENTRY_BYTES / 4) as usize;
            let voxel_words = (VOXEL_BYTES / 4) as usize;
            let mut out = Vec::new();
            for bucket in 0..(self.buckets.size() / BUCKET_BYTES) as usize {
                for i in 0..BUCKET_SIZE as usize {
                    let start = bucket * bucket_words + 1 + i * entry_words;
                    let [x, y, z, offset, flags, low, high] = buckets[start..start + entry_words]
                    else {
                        unreachable!("an entry is 7 words");
                    };
                    if offset & ENTRY_OCCUPIED_MASK == 0 {
                        continue;
                    }
                    let entry = bucket * BUCKET_SIZE as usize + i;
                    // Shifting left restores the sign of the page coordinates
                    let page = |c: u32| ((c << BITS_FOR_ENTRY_VOXEL_COUNT) as i32) >> 2;
                    out.push(Entry {
                        page: (page(x), page(y), page(z)),
                        flags,
                        occupancy: u64::from(low) | (u64::from(high) << 32),
                        voxels: std::array::from_fn(|inner| {
                            voxels[(entry * ENTRY_VOXEL_COUNT as usize + inner) * voxel_words]
                        }),
                    });
                }
            }
            out
        }
    }

    impl TestVoxelizer for WgpuAccelerated {
        const BITS_FOR_ZORDER: u32 = BITS_FOR_ENTRY_VOXEL_COUNT;
        const BITS_FOR_COMPRESSED_ZORDER: u32 = BITS_FOR_COMPRESSED_ZORDER;
        const BITS_FOR_COMPRESSED_ZORDER_MASK: i32 = BITS_FOR_COMPRESSED_ZORDER_MASK;
        const PAGE_SIZE: u32 = COMPRESSED_PAGE_LEN as u32;
        fn make(
            resolution: f32,
            hit_multiplier: u32,
            target_multiplier: u32,
            destroy_threshold: u32,
            construct_threshold: u32,
            compress: bool,
        ) -> Self {
            new(
                resolution,
                hit_multiplier,
                target_multiplier,
                destroy_threshold,
                construct_threshold,
                compress,
                TEST_BUCKETS,
            )
        }
        // Same z-order as the shader's `get_z_order`
        fn get_index(&self, point: &PointI) -> PagedIndex {
            (
                (point.x >> 2, point.y >> 2, point.z >> 2),
                ZOrder::default().index_of([
                    (point.x & 3) as u16,
                    (point.y & 3) as u16,
                    (point.z & 3) as u16,
                ]) as usize,
            )
        }
        fn point_from_index(&self, idx: PagedIndex, i: u32) -> PointI {
            let [x, y, z] = ZOrder::default().coord_of(idx.1 as u64);
            PointI {
                x: (idx.0.0 << 2) | x as i32,
                y: (idx.0.1 << 2) | y as i32,
                z: (idx.0.2 << 2) | z as i32,
                i,
            }
        }
        // Mirror of the shader's `closest_boundary`
        fn closest_boundary(upper: bool, position: f32, len: f32) -> f32 {
            (position.floor() + upper as i32 as f32 - position) / len
        }
        fn set_voxel(&mut self, x: i32, y: i32, z: i32, value: u32) {
            self.set_voxels(&[(x, y, z, value)]);
        }
        fn pages(&self) -> Vec<(Coord, [u32; PAGE_LEN])> {
            let mut out: Vec<_> = self
                .entries()
                .into_iter()
                .map(|e| (e.page, e.voxels))
                .collect();
            out.sort_by_key(|(k, _)| *k);
            out
        }
        // A compressed page exists once any of its voxels got constructed
        fn compressed_pages(&self) -> Option<Vec<(Coord, Vec<u64>)>> {
            if !self.compress {
                return None;
            }
            let zorder = ZOrder::default();
            let mut out = BTreeMap::new();
            for e in self.entries() {
                if e.flags & ENTRY_CONSTRUCTED == 0 {
                    continue;
                }
                let key = (
                    e.page.0 >> BITS_FOR_COMPRESSED_ZORDER,
                    e.page.1 >> BITS_FOR_COMPRESSED_ZORDER,
                    e.page.2 >> BITS_FOR_COMPRESSED_ZORDER,
                );
                let inner = zorder.index_of([
                    (e.page.0 & BITS_FOR_COMPRESSED_ZORDER_MASK) as u16,
                    (e.page.1 & BITS_FOR_COMPRESSED_ZORDER_MASK) as u16,
                    (e.page.2 & BITS_FOR_COMPRESSED_ZORDER_MASK) as u16,
                ]) as usize;
                out.entry(key)
                    .or_insert_with(|| vec![0; COMPRESSED_PAGE_LEN])[inner] = e.occupancy;
            }
            Some(out.into_iter().collect())
        }
        fn pending_results(&self) -> usize {
            self.entries()
                .iter()
                .filter(|e| e.flags & ENTRY_TOUCHED != 0)
                .count()
        }
    }

    crate::test::shared_tests!(WgpuAccelerated);

    /// Mirror of the shader's `hash` for pages with non negative coordinates.
    fn bucket_of(page: Coord, bucket_count: u32) -> u32 {
        const P1: u32 = 73856093;
        const P2: u32 = 19349669;
        const P3: u32 = 83492791;
        ((page.0 as u32).wrapping_mul(P1)
            ^ (page.1 as u32).wrapping_mul(P2)
            ^ (page.2 as u32).wrapping_mul(P3))
            % bucket_count
    }

    // ------------------------------------------------------------ constructor

    #[test]
    fn new_stores_parameters() {
        let v = new(0.5, 2, 3, 4, 5, false, 16);
        assert_eq!(v.resolution, 0.5);
        assert_eq!(v.hit_multiplier, 2);
        assert_eq!(v.target_multiplier, 3);
        assert_eq!(v.destroy_threshold, 4);
        assert_eq!(v.construct_threshold, 5);
        assert!(!v.compress);
        assert_eq!(v.buckets.size(), 16 * BUCKET_BYTES);
        assert_eq!(
            v.voxels.size(),
            16 * BUCKET_SIZE * ENTRY_VOXEL_COUNT * VOXEL_BYTES
        );
        assert!(v.pages().is_empty());
        assert_eq!(v.pending_results(), 0);
    }

    #[test]
    fn new_rejects_invalid_bucket_count() {
        assert!(pollster::block_on(WgpuAccelerated::new(1.0, 1, 1, 0, 1, false, 0)).is_none());
        // Voxel indices would not fit u32
        assert!(
            pollster::block_on(WgpuAccelerated::new(1.0, 1, 1, 0, 1, false, u32::MAX)).is_none()
        );
    }

    #[test]
    fn memory_size_matches_buffers() {
        let v = new(1.0, 1, 1, 0, 1, false, 100);
        assert_eq!(
            WgpuAccelerated::memory_size(100),
            v.buckets.size() + v.voxels.size() + v.altered.size()
        );
    }

    // ------------------------------------------------------------ concurrency

    #[test]
    fn concurrent_same_target() {
        // Every thread adds to the same target and subtracts from the same
        // three voxels on the way: no update may get lost and the result
        // must hold the final values once.
        const N: u32 = 10_000;
        let mut v = new(1.0, 1, 1, 0, 1, false, TEST_BUCKETS);
        let hits = [(0, 0, 0), (1, 0, 0), (2, 0, 0)];
        seed(&mut v, &hits, 2 * N);
        let points = vec![pf(3.5, 0.5, 0.5, 1.0); N as usize];
        let result = v.add_points(pf(0.5, 0.5, 0.5, 1.0), &points);
        let mut expected = with_i(&hits, N);
        expected.push(pi(3, 0, 0, N));
        let expected = sorted(expected);
        assert_eq!(sorted(result), expected);
        assert_eq!(nonzero_voxels(&v), expected);
        assert_eq!(v.pending_results(), 0);
    }

    #[test]
    fn concurrent_distinct_pages() {
        // Targets 8 apart on the faces of a cube: each in its own page, none
        // on the path of another ray from the origin, all inserted at once.
        let mut v = new(1.0, 1, 1, 0, 1, false, TEST_BUCKETS);
        let targets = cube_faces(40, 32, 8);
        let intensity = |n: usize| (1 + n % 7) as u32;
        let points: Vec<_> = targets
            .iter()
            .enumerate()
            .map(|(n, &(x, y, z))| {
                pf(
                    x as f32 + 0.5,
                    y as f32 + 0.5,
                    z as f32 + 0.5,
                    intensity(n) as f32,
                )
            })
            .collect();
        let result = v.add_points(pf(0.5, 0.5, 0.5, 1.0), &points);
        let expected = sorted(
            targets
                .iter()
                .enumerate()
                .map(|(n, &(x, y, z))| pi(x, y, z, intensity(n)))
                .collect(),
        );
        assert_eq!(sorted(result), expected);
        assert_eq!(nonzero_voxels(&v), expected);
        assert_eq!(v.pages().len(), targets.len());
        assert_eq!(v.pending_results(), 0);
    }

    // ------------------------------------------------------------- hashmap

    #[test]
    fn hash_collisions_use_lists() {
        // 6 pages hashed into one of 4 buckets: 2 fit the bucket, the next 3
        // extend its list into the first entry of each following bucket and
        // the last one does not fit anymore.
        const BUCKETS: u32 = 4;
        let pages: Vec<Coord> = (0..)
            .map(|k| (k, 0, 0))
            .filter(|&page| bucket_of(page, BUCKETS) == 0)
            .take(6)
            .collect();
        let voxel = |(x, y, z): Coord| (4 * x + 1, 4 * y + 1, 4 * z + 1);
        let mut v = new(1.0, 1, 1, 0, 1, false, BUCKETS);
        for &page in &pages[..5] {
            let (x, y, z) = voxel(page);
            assert_eq!(from_above(&mut v, voxel(page), 1.0), vec![pi(x, y, z, 1)]);
        }
        assert!(from_above(&mut v, voxel(pages[5]), 1.0).is_empty());
        assert_eq!(v.pages().len(), 5);
        // Every stored page is found again, also through the list
        for &page in &pages[..5] {
            let (x, y, z) = voxel(page);
            assert_eq!(from_above(&mut v, voxel(page), 1.0), vec![pi(x, y, z, 2)]);
        }
        let (x, y, z) = voxel(pages[5]);
        assert_eq!(get_voxel(&v, x, y, z), None);
        assert_eq!(v.pages().len(), 5);
    }

    #[test]
    fn full_table_drops_without_hanging() {
        // Far more pages than entries, inserted concurrently: whatever got
        // stored must be consistent with what got reported.
        let mut v = new(1.0, 1, 1, 0, 1, false, 8);
        let targets = cube_faces(40, 32, 8);
        let points: Vec<_> = targets
            .iter()
            .map(|&(x, y, z)| pf(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5, 1.0))
            .collect();
        let result = sorted(v.add_points(pf(0.5, 0.5, 0.5, 1.0), &points));
        assert!(!result.is_empty());
        assert!(result.len() <= 16);
        assert!(
            result
                .iter()
                .all(|p| targets.contains(&(p.x, p.y, p.z)) && p.i == 1)
        );
        assert_eq!(nonzero_voxels(&v), result);
        assert_eq!(v.pages().len(), result.len());
        assert_eq!(v.pending_results(), 0);
    }
}
