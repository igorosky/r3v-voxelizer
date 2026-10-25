use crate::{
    Multipliers, Voxelizer, VoxelizerParams,
    geometry::{PointF, PointI},
};

// Must match the shader
const WORKGROUP_SIZE: u32 = 256;
const BITS_FOR_ENTRY_VOXEL_COUNT: u32 = 2;
const ENTRY_VOXEL_COUNT: u64 = 1 << (3 * BITS_FOR_ENTRY_VOXEL_COUNT);
const BUCKET_SIZE: u64 = 2;
/// `Entry`: key[3], flags, occupancy[2]
const ENTRY_BYTES: u64 = 6 * 4;
/// Entries first probed for the keys hashed into a bucket
const BUCKET_BYTES: u64 = BUCKET_SIZE * ENTRY_BYTES;
/// `Voxel`: value + prev
const VOXEL_BYTES: u64 = 2 * 4;
/// `EntryList` header: len
const ENTRY_LIST_HEADER_BYTES: u64 = 4;
/// `Output` header: len, padded to the alignment of `vec4i`
const OUTPUT_HEADER_BYTES: u64 = 16;
/// `Params`
const PARAMS_BYTES: u64 = 16 * 4;
/// Input (`vec4u`) and output (`vec4i`) element
const POINT_BYTES: u64 = 16;
/// Capacity (in points) the input and output buffers start with
const INITIAL_CAPACITY: u64 = 1024;
/// Environment variable turning on the diagnostics printed to stderr
const DEBUG_ENV: &str = "R3V_VOXELIZER_DEBUG";
/// Misaligned outputs reported per voxelizer in debug mode
const MAX_DEBUG_REPORTS: u32 = 3;
/// Blocks and entries a debug report lists
const DEBUG_REPORT_ITEMS: usize = 8;

#[derive(Debug, Clone, Copy)]
enum Kernel {
    RayMarch,
    AddPoint,
    Collect,
    RemoveEmpty,
    ReclaimRemoved,
    SetVoxels,
}

impl Kernel {
    const ALL: [Kernel; 6] = [
        Kernel::RayMarch,
        Kernel::AddPoint,
        Kernel::Collect,
        Kernel::RemoveEmpty,
        Kernel::ReclaimRemoved,
        Kernel::SetVoxels,
    ];
    fn entry_point(self) -> &'static str {
        match self {
            Kernel::RayMarch => "ray_march",
            Kernel::AddPoint => "add_point",
            Kernel::Collect => "collect",
            Kernel::RemoveEmpty => "remove_empty",
            Kernel::ReclaimRemoved => "reclaim_removed",
            Kernel::SetVoxels => "set_voxels",
        }
    }
}

/// Voxelizer running on the GPU.
///
/// Voxels live in a fixed size hashmap of `bucket_count` buckets, each one
/// holding 2 pages of 4x4x4 voxels. A bucket takes about 1 KiB of GPU memory
/// (see [`WgpuAccelerated::memory_size`]). Values of pages that do not fit
/// anymore are dropped. Pages whose voxels all drop to 0 during a call are
/// removed at its end, freeing their space.
///
/// With `R3V_VOXELIZER_DEBUG` set in the environment it prints the adapter
/// it runs on and, in compress mode, a report of the GPU state whenever
/// blocks come back off the 4x4x2 block grid, to stderr.
#[derive(Debug)]
pub struct WgpuAccelerated {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: [wgpu::ComputePipeline; Kernel::ALL.len()],
    io_layout: wgpu::BindGroupLayout,
    io_bind_group: wgpu::BindGroup,
    storage_bind_group: wgpu::BindGroup,
    params: wgpu::Buffer,
    input: wgpu::Buffer,
    altered: wgpu::Buffer,
    output: wgpu::Buffer,
    emptied: wgpu::Buffer,
    // Used through `storage_bind_group`, read back by the tests and debug
    // reports
    buckets: wgpu::Buffer,
    #[cfg_attr(not(test), expect(dead_code))]
    voxels: wgpu::Buffer,
    max_binding_size: u64,
    max_workgroups: u32,
    resolution: f32,
    multipliers: Multipliers,
    destroy_threshold: u32,
    construct_threshold: u32,
    max_value: u32,
    compress: bool,
    /// `R3V_VOXELIZER_DEBUG` was set when it was created
    debug: bool,
    adapter_info: wgpu::AdapterInfo,
    debug_reports: u32,
}

impl Voxelizer for WgpuAccelerated {
    /// Blocks until the GPU is done, see [`WgpuAccelerated::add_points_async`]
    /// for targets that cannot block (the web).
    fn add_points(&mut self, sensor_pos: PointF, points: &[PointF]) -> Vec<PointI> {
        pollster::block_on(self.add_points_async(sensor_pos, points))
    }
    /// Takes effect with the next call, the shader gets the multipliers with
    /// every dispatch.
    #[inline]
    fn set_multipliers(&mut self, multipliers: Multipliers) {
        self.multipliers = multipliers;
    }
    #[inline]
    fn multipliers(&self) -> Multipliers {
        self.multipliers
    }
}

impl WgpuAccelerated {
    /// Same as [`Voxelizer::add_points`], without blocking while the GPU
    /// works. The only way to add points on the web, where waiting for the
    /// GPU has to give control back to the browser.
    pub async fn add_points_async(&mut self, sensor_pos: PointF, points: &[PointF]) -> Vec<PointI> {
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
        self.queue
            .write_buffer(&self.emptied, 0, &0u32.to_ne_bytes());
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

        let altered = self.read_words(self.encoder(), &self.altered, 0, 1).await[0];
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
            self.queue.write_buffer(&self.output, 0, &[0; 4]);
            self.write_params([0.0; 3], count, first);
            let mut encoder = self.encoder();
            self.dispatch(&mut encoder, Kernel::Collect, count * 2);
            if first + count == altered {
                // Emptied entries are altered as well, so this many threads cover them
                self.dispatch(&mut encoder, Kernel::RemoveEmpty, altered);
                self.dispatch(&mut encoder, Kernel::ReclaimRemoved, altered);
            }
            let len = self.read_words(encoder, &self.output, 0, 1).await[0];
            if len != 0 {
                let data = self
                    .read_back(
                        self.encoder(),
                        &self.output,
                        OUTPUT_HEADER_BYTES,
                        u64::from(len) * POINT_BYTES,
                    )
                    .await;
                let blocks = bytemuck::pod_collect_to_vec::<u8, [u32; 4]>(&data);
                if self.compress
                    && self.debug
                    && self.debug_reports < MAX_DEBUG_REPORTS
                    && blocks.iter().any(off_block_grid)
                {
                    self.debug_reports += 1;
                    self.report_misaligned(camera, points.len(), &blocks, altered, first, count)
                        .await;
                }
                result.extend(blocks.into_iter().map(|[x, y, z, i]| PointI {
                    x: x as i32,
                    y: y as i32,
                    z: z as i32,
                    i,
                }));
            }
        }
        result
    }

    /// GPU memory taken by the hashmap of `bucket_count` buckets, in bytes.
    pub fn memory_size(bucket_count: u32) -> u64 {
        let entry_count = u64::from(bucket_count) * BUCKET_SIZE;
        u64::from(bucket_count) * BUCKET_BYTES
            + entry_count * ENTRY_VOXEL_COUNT * VOXEL_BYTES
            + 2 * Self::entry_list_size(entry_count)
    }

    /// Size of a list (`altered` or `emptied`) able to hold every entry.
    fn entry_list_size(entry_count: u64) -> u64 {
        ENTRY_LIST_HEADER_BYTES + entry_count * 4
    }

    /// Returns `None` when there is no adapter capable of running compute
    /// shaders, `bucket_count` is 0 or the hashmap does not fit the adapter's
    /// limits.
    pub async fn new(params: VoxelizerParams, bucket_count: u32) -> Option<Self> {
        Self::with_debug(params, bucket_count, std::env::var_os(DEBUG_ENV).is_some()).await
    }

    /// [`WgpuAccelerated::new`] with the diagnostics on or off whatever the
    /// environment says.
    async fn with_debug(params: VoxelizerParams, bucket_count: u32, debug: bool) -> Option<Self> {
        let VoxelizerParams {
            resolution,
            multipliers,
            destroy_threshold,
            construct_threshold,
            max_value,
            compress,
        } = params;
        let entry_count = u64::from(bucket_count) * BUCKET_SIZE;
        let voxel_count = entry_count * ENTRY_VOXEL_COUNT;
        // Voxels are indexed with u32 in the shader
        if bucket_count == 0 || voxel_count > 1 << 32 {
            return None;
        }
        let buckets_size = u64::from(bucket_count) * BUCKET_BYTES;
        let voxels_size = voxel_count * VOXEL_BYTES;
        let list_size = Self::entry_list_size(entry_count);

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
        let adapter_info = adapter.get_info();
        let adapter_limits = adapter.limits();
        let max_binding_size = adapter_limits
            .max_storage_buffer_binding_size
            .min(adapter_limits.max_buffer_size);
        if buckets_size.max(voxels_size).max(list_size) > max_binding_size {
            return None;
        }
        let has_uma = {
            use wgpu::DeviceType;
            match adapter_info.device_type {
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
                layout_entry(4, storage(false)),
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
        // Debug reports read the params back
        let params_usage = if debug {
            Usage::UNIFORM | Usage::COPY_DST | Usage::COPY_SRC
        } else {
            Usage::UNIFORM | Usage::COPY_DST
        };
        let params = buffer("params", PARAMS_BYTES, params_usage);
        let input = Self::input_buffer(&device, INITIAL_CAPACITY);
        let output = Self::output_buffer(&device, INITIAL_CAPACITY);
        let readable = Usage::STORAGE | Usage::COPY_SRC | Usage::COPY_DST;
        let altered = buffer("altered", list_size, readable);
        let emptied = buffer("emptied", list_size, Usage::STORAGE | Usage::COPY_DST);
        // Zero initialized, which is an empty hashmap
        let buckets = buffer("buckets", buckets_size, readable);
        let voxels = buffer("voxels", voxels_size, readable);
        let io_bind_group = Self::io_bind_group(
            &device, &io_layout, &params, &input, &altered, &output, &emptied,
        );
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
        if debug {
            eprintln!(
                "r3v-voxelizer wgpu: adapter {:?} ({:?}, {:?}), driver {:?} {:?}, uma {has_uma}, \
                 max_storage_buffer_binding_size {}, max_buffer_size {}, \
                 buffers: buckets {buckets_size} B, voxels {voxels_size} B, lists {list_size} B",
                adapter_info.name,
                adapter_info.backend,
                adapter_info.device_type,
                adapter_info.driver,
                adapter_info.driver_info,
                device_limits.max_storage_buffer_binding_size,
                device_limits.max_buffer_size,
            );
        }
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
            emptied,
            buckets,
            voxels,
            resolution,
            multipliers,
            destroy_threshold,
            construct_threshold,
            max_value,
            compress,
            debug,
            adapter_info,
            debug_reports: 0,
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
        emptied: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("io"),
            layout,
            entries: &[params, input, altered, output, emptied]
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
            &self.emptied,
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
            self.multipliers.hit,
            self.multipliers.target,
            self.destroy_threshold,
            self.construct_threshold,
            count,
            first,
            u32::from(self.compress),
            self.max_value,
            self.multipliers.intensity.to_bits(),
            0,
            0,
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
    async fn read_back(
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
        let (mapped, on_mapped) = futures_channel::oneshot::channel();
        staging.map_async(wgpu::MapMode::Read, .., move |_| {
            let _ = mapped.send(());
        });
        // Native backends map (and call back) while polling, the web maps on
        // its own and polling does nothing there
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("waiting for the GPU failed");
        on_mapped
            .await
            .expect("the GPU dropped the mapping callback");
        let data = staging
            .get_mapped_range(..)
            .expect("mapping the GPU buffer failed")
            .to_vec();
        staging.unmap();
        data
    }

    async fn read_words(
        &self,
        encoder: wgpu::CommandEncoder,
        source: &wgpu::Buffer,
        offset: u64,
        count: u64,
    ) -> Vec<u32> {
        bytemuck::pod_collect_to_vec(&self.read_back(encoder, source, offset, count * 4).await)
    }

    /// Prints the state behind `blocks`, the output of one `collect` chunk
    /// some of which is off the block grid, which `collect` never writes in
    /// compress mode: what was read back and whether reading it again gives
    /// the same, the params the GPU got and the first altered entries of the
    /// chunk.
    async fn report_misaligned(
        &self,
        camera: [f32; 3],
        point_count: usize,
        blocks: &[[u32; 4]],
        altered: u32,
        first: u32,
        count: u32,
    ) {
        use std::fmt::Write;

        let block =
            |[x, y, z, i]: [u32; 4]| format!("({}, {}, {}, i {i})", x as i32, y as i32, z as i32);
        let misaligned: Vec<_> = blocks
            .iter()
            .enumerate()
            .filter(|(_, block)| off_block_grid(block))
            .collect();
        let len = blocks.len() as u64;
        let sensor = camera.map(|c| (c / self.resolution).floor() as i32);
        // Writing to a String does not fail
        let mut out = String::new();
        let _ = writeln!(
            out,
            "r3v-voxelizer wgpu: {} of {len} blocks off the 4x4x2 grid in compress mode",
            misaligned.len()
        );
        let _ = writeln!(
            out,
            "  adapter {:?} ({:?}, {:?}), driver {:?} {:?}",
            self.adapter_info.name,
            self.adapter_info.backend,
            self.adapter_info.device_type,
            self.adapter_info.driver,
            self.adapter_info.driver_info,
        );
        let _ = writeln!(
            out,
            "  {point_count} points, sensor at voxel {sensor:?}; altered {altered}, chunk first {first} count {count}, \
             at most {} blocks, output capacity {} blocks",
            u64::from(count) * 2,
            (self.output.size() - OUTPUT_HEADER_BYTES) / POINT_BYTES,
        );
        let _ = writeln!(
            out,
            "  i: max {}, {} of {len} at most max_value {}",
            blocks.iter().map(|block| block[3]).max().unwrap_or(0),
            blocks
                .iter()
                .filter(|block| block[3] <= self.max_value)
                .count(),
            self.max_value,
        );
        let _ = writeln!(
            out,
            "  first blocks: {}",
            blocks
                .iter()
                .take(DEBUG_REPORT_ITEMS)
                .map(|&b| block(b))
                .collect::<Vec<_>>()
                .join(", ")
        );
        let _ = writeln!(
            out,
            "  first misaligned: {}",
            misaligned
                .iter()
                .take(DEBUG_REPORT_ITEMS)
                .map(|&(k, &b)| format!("[{k}] {}", block(b)))
                .collect::<Vec<_>>()
                .join(", ")
        );

        let len_again = self.read_words(self.encoder(), &self.output, 0, 1).await[0];
        let again = bytemuck::pod_collect_to_vec::<u8, [u32; 4]>(
            &self
                .read_back(
                    self.encoder(),
                    &self.output,
                    OUTPUT_HEADER_BYTES,
                    len * POINT_BYTES,
                )
                .await,
        );
        let _ = writeln!(
            out,
            "  read again: len {len_again}, {} of {len} blocks differ",
            blocks.iter().zip(&again).filter(|(a, b)| a != b).count()
        );

        let p = self
            .read_words(self.encoder(), &self.params, 0, PARAMS_BYTES / 4)
            .await;
        let f = f32::from_bits;
        let _ = writeln!(
            out,
            "  params read back: camera ({}, {}, {}), resolution {}, hit {}, target {}, destroy {}, \
             construct {}, count {}, first {}, compress {}, max_value {}, intensity {}, padding {:?}",
            f(p[0]),
            f(p[1]),
            f(p[2]),
            f(p[3]),
            p[4],
            p[5],
            p[6],
            p[7],
            p[8],
            p[9],
            p[10],
            p[11],
            f(p[12]),
            &p[13..],
        );

        let shown = count.min(DEBUG_REPORT_ITEMS as u32);
        let ids = self
            .read_words(
                self.encoder(),
                &self.altered,
                ENTRY_LIST_HEADER_BYTES + u64::from(first) * 4,
                u64::from(shown),
            )
            .await;
        let entry_count = self.buckets.size() / ENTRY_BYTES;
        // Shifting left restores the voxel coordinates and drops KEY_SET
        let origin = |key: u32| (key << BITS_FOR_ENTRY_VOXEL_COUNT) as i32;
        for id in ids {
            if u64::from(id) >= entry_count {
                let _ = writeln!(
                    out,
                    "  altered entry {id}: out of range ({entry_count} entries)"
                );
                continue;
            }
            let w = self
                .read_words(
                    self.encoder(),
                    &self.buckets,
                    u64::from(id) * ENTRY_BYTES,
                    ENTRY_BYTES / 4,
                )
                .await;
            let _ = writeln!(
                out,
                "  altered entry {id}: key {:#010x} {:#010x} {:#010x} -> origin ({}, {}, {}), flags {}, \
                 occupancy {:#010x} {:#010x}",
                w[0],
                w[1],
                w[2],
                origin(w[0]),
                origin(w[1]),
                origin(w[2]),
                w[3],
                w[4],
                w[5],
            );
        }
        eprint!("{out}");
    }
}

/// Whether a compress mode output `[x, y, z, mask]` is off the 4x4x2 block
/// grid the renderer's tesselator takes. `collect` writes only block origins.
fn off_block_grid(&[x, y, z, _]: &[u32; 4]) -> bool {
    x % 4 != 0 || y % 4 != 0 || z % 2 != 0
}

#[cfg(test)]
mod test {
    use std::collections::BTreeMap;

    use super::{
        BITS_FOR_ENTRY_VOXEL_COUNT, BUCKET_BYTES, BUCKET_SIZE, ENTRY_BYTES, ENTRY_VOXEL_COUNT,
        Kernel, VOXEL_BYTES, WgpuAccelerated, off_block_grid,
    };
    use crate::{
        Multipliers, Voxelizer, VoxelizerParams,
        geometry::{PointI, ZOrder},
        test::{
            COMPRESSED_PAGE_LEN, Coord, PAGE_LEN, PLAIN, PagedIndex, TestVoxelizer, along_x,
            cube_faces, from_above, get_voxel, nonzero_voxels, pf, pi, seed, sorted, with_i,
        },
    };

    /// Hashmap size used by the shared suite: 8192 pages.
    const TEST_BUCKETS: u32 = 1 << 12;
    const BITS_FOR_COMPRESSED_ZORDER: u32 = 4;
    const BITS_FOR_COMPRESSED_ZORDER_MASK: i32 = 2_i32.pow(BITS_FOR_COMPRESSED_ZORDER) - 1;
    // Must match the shader
    const KEY_SET: u32 = 1 << 31;
    const KEY_REMOVED: u32 = 1;
    const ENTRY_TOUCHED: u32 = 1;
    const ENTRY_MAYBE_EMPTY: u32 = 2;

    fn with_buckets(params: VoxelizerParams, bucket_count: u32) -> WgpuAccelerated {
        pollster::block_on(WgpuAccelerated::new(params, bucket_count))
            .expect("a GPU adapter with compute shaders is required")
    }

    /// Without a cap on voxel values or scaling of intensities.
    fn new(
        resolution: f32,
        hit_multiplier: u32,
        target_multiplier: u32,
        destroy_threshold: u32,
        construct_threshold: u32,
        compress: bool,
        bucket_count: u32,
    ) -> WgpuAccelerated {
        with_buckets(
            VoxelizerParams {
                resolution,
                multipliers: Multipliers {
                    intensity: 1.0,
                    hit: hit_multiplier,
                    target: target_multiplier,
                },
                destroy_threshold,
                construct_threshold,
                max_value: u32::MAX,
                compress,
            },
            bucket_count,
        )
    }

    /// Occupied hashmap entry (page) read back from the GPU.
    struct Entry {
        page: Coord,
        flags: u32,
        occupancy: u64,
        voxels: [u32; PAGE_LEN],
    }

    impl WgpuAccelerated {
        /// Whole `buffer`, read back as words.
        fn words_of(&self, buffer: &wgpu::Buffer) -> Vec<u32> {
            bytemuck::pod_collect_to_vec(&pollster::block_on(self.read_back(
                self.encoder(),
                buffer,
                0,
                buffer.size(),
            )))
        }

        fn entries(&self) -> Vec<Entry> {
            let entries = self.words_of(&self.buckets);
            let voxels = self.words_of(&self.voxels);
            let entry_words = (ENTRY_BYTES / 4) as usize;
            let voxel_words = (VOXEL_BYTES / 4) as usize;
            let mut out = Vec::new();
            for (entry, words) in entries.chunks_exact(entry_words).enumerate() {
                let [x, y, z, flags, low, high] = *words else {
                    unreachable!("an entry is 6 words");
                };
                if x & KEY_SET == 0 {
                    continue;
                }
                // Shifting left restores the sign of the page coordinates and
                // drops KEY_SET
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
            out
        }

        /// First key word of every hashmap entry, set ones included.
        fn key_heads(&self) -> Vec<u32> {
            let entries = self.words_of(&self.buckets);
            let entry_words = (ENTRY_BYTES / 4) as usize;
            entries
                .chunks_exact(entry_words)
                .map(|words| words[0])
                .collect()
        }
    }

    impl TestVoxelizer for WgpuAccelerated {
        const BITS_FOR_ZORDER: u32 = BITS_FOR_ENTRY_VOXEL_COUNT;
        const BITS_FOR_COMPRESSED_ZORDER: u32 = BITS_FOR_COMPRESSED_ZORDER;
        const BITS_FOR_COMPRESSED_ZORDER_MASK: i32 = BITS_FOR_COMPRESSED_ZORDER_MASK;
        const PAGE_SIZE: u32 = COMPRESSED_PAGE_LEN as u32;
        fn with_params(params: VoxelizerParams) -> Self {
            with_buckets(params, TEST_BUCKETS)
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
        // One dispatch for all of them
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
        fn pages(&self) -> Vec<(Coord, [u32; PAGE_LEN])> {
            let mut out: Vec<_> = self
                .entries()
                .into_iter()
                .map(|e| (e.page, e.voxels))
                .collect();
            out.sort_by_key(|(k, _)| *k);
            out
        }
        // A compressed page exists while any of its voxels is constructed
        fn compressed_pages(&self) -> Option<Vec<(Coord, Vec<u64>)>> {
            if !self.compress {
                return None;
            }
            let zorder = ZOrder::default();
            let mut out = BTreeMap::new();
            for e in self.entries() {
                if e.occupancy == 0 {
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
                .filter(|e| e.flags & (ENTRY_TOUCHED | ENTRY_MAYBE_EMPTY) != 0)
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
        let v = with_buckets(
            VoxelizerParams {
                resolution: 0.5,
                multipliers: Multipliers {
                    intensity: 0.25,
                    hit: 2,
                    target: 3,
                },
                destroy_threshold: 4,
                construct_threshold: 5,
                max_value: 6,
                compress: false,
            },
            16,
        );
        assert_eq!(v.resolution, 0.5);
        assert_eq!(
            v.multipliers,
            Multipliers {
                intensity: 0.25,
                hit: 2,
                target: 3,
            }
        );
        assert_eq!(v.destroy_threshold, 4);
        assert_eq!(v.construct_threshold, 5);
        assert_eq!(v.max_value, 6);
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
        assert!(pollster::block_on(WgpuAccelerated::new(PLAIN, 0)).is_none());
        // Voxel indices would not fit u32
        assert!(pollster::block_on(WgpuAccelerated::new(PLAIN, u32::MAX)).is_none());
    }

    #[test]
    fn memory_size_matches_buffers() {
        let v = new(1.0, 1, 1, 0, 1, false, 100);
        assert_eq!(
            WgpuAccelerated::memory_size(100),
            v.buckets.size() + v.voxels.size() + v.altered.size() + v.emptied.size()
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

    /// First `count` pages (k, 0, 0) hashed into the first bucket.
    fn colliding_pages(count: usize, bucket_count: u32) -> Vec<Coord> {
        (0..)
            .map(|k| (k, 0, 0))
            .filter(|&page| bucket_of(page, bucket_count) == 0)
            .take(count)
            .collect()
    }

    /// Voxel inside `page`.
    fn voxel_of((x, y, z): Coord) -> Coord {
        (4 * x + 1, 4 * y + 1, 4 * z + 1)
    }

    #[test]
    fn hash_collisions_probe_following_entries() {
        // 9 pages hashed into the first of 4 buckets: the first 8 take all 8
        // entries, probing past the bucket, and the last one does not fit.
        const BUCKETS: u32 = 4;
        let entries = (u64::from(BUCKETS) * BUCKET_SIZE) as usize;
        let pages = colliding_pages(entries + 1, BUCKETS);
        let mut v = new(1.0, 1, 1, 0, 1, false, BUCKETS);
        for &page in &pages[..entries] {
            let (x, y, z) = voxel_of(page);
            assert_eq!(from_above(&mut v, (x, y, z), 1.0), vec![pi(x, y, z, 1)]);
        }
        assert!(from_above(&mut v, voxel_of(pages[entries]), 1.0).is_empty());
        assert_eq!(v.pages().len(), entries);
        // Every stored page is found again, also past its bucket
        for &page in &pages[..entries] {
            let (x, y, z) = voxel_of(page);
            assert_eq!(from_above(&mut v, (x, y, z), 1.0), vec![pi(x, y, z, 2)]);
        }
        let (x, y, z) = voxel_of(pages[entries]);
        assert_eq!(get_voxel(&v, x, y, z), None);
        assert_eq!(v.pages().len(), entries);
    }

    #[test]
    fn concurrent_inserts_of_colliding_pages() {
        // Many threads insert the same new pages at once, all of them hashed
        // into the same bucket: every page must end up in exactly one entry.
        const BUCKETS: u32 = 8;
        const PAGES: usize = 8;
        const PER_PAGE: u32 = 1000;
        let targets: Vec<Coord> = colliding_pages(PAGES, BUCKETS)
            .into_iter()
            .map(voxel_of)
            .collect();
        // Interleaved, so neighbouring threads insert different pages
        let points: Vec<_> = (0..PER_PAGE)
            .flat_map(|_| {
                targets
                    .iter()
                    .map(|&(x, y, z)| pf(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5, 1.0))
            })
            .collect();
        let mut v = new(1.0, 1, 1, 0, 1, false, BUCKETS);
        let result = v.add_points(pf(0.5, 0.5, 0.5, 1.0), &points);
        let expected = sorted(
            targets
                .iter()
                .map(|&(x, y, z)| pi(x, y, z, PER_PAGE))
                .collect(),
        );
        assert_eq!(sorted(result), expected);
        assert_eq!(nonzero_voxels(&v), expected);
        assert_eq!(v.pages().len(), PAGES);
        assert_eq!(v.pending_results(), 0);
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

    #[test]
    fn tombstone_keeps_probe_chain() {
        // Pages A, B and C hashed into the first bucket take entries 0, 1, 2.
        const BUCKETS: u32 = 4;
        let pages = colliding_pages(3, BUCKETS);
        let [a, b, c] = [0, 1, 2].map(|n| voxel_of(pages[n]));
        let mut v = new(1.0, 1, 1, 0, 1, false, BUCKETS);
        for (x, y, z) in [a, b, c] {
            v.set_voxel(x, y, z, 1);
        }
        assert!(v.key_heads()[..3].iter().all(|key| key & KEY_SET != 0));

        // Empties B with a ray ending in A: B leaves a tombstone, since C is
        // stored right after it.
        let result = along_x(&mut v, b.0, a.0, a.1, a.2, 1.0);
        assert_eq!(
            sorted(result),
            vec![pi(a.0, a.1, a.2, 2), pi(b.0, b.1, b.2, 0)]
        );
        assert_eq!(v.key_heads()[1], KEY_REMOVED);
        assert_eq!(v.pages().len(), 2);
        assert_eq!(v.pending_results(), 0);

        // C is still found past the tombstone, nothing is inserted twice
        assert_eq!(from_above(&mut v, c, 1.0), vec![pi(c.0, c.1, c.2, 2)]);
        assert_eq!(v.pages().len(), 2);

        // Emptying C frees its entry and the tombstone before it
        let result = along_x(&mut v, c.0, a.0, a.1, a.2, 2.0);
        assert_eq!(
            sorted(result),
            vec![pi(a.0, a.1, a.2, 4), pi(c.0, c.1, c.2, 0)]
        );
        let heads = v.key_heads();
        assert_ne!(heads[0] & KEY_SET, 0);
        assert_eq!(heads[1..3], [0, 0]);
        assert_eq!(v.pages().len(), 1);
        assert_eq!(v.pending_results(), 0);
    }

    #[test]
    fn full_table_reused_after_removal() {
        // Every entry taken, then emptied one by one by rays whose targets do
        // not fit: no entry is freed before the last one, which frees them all.
        const BUCKETS: u32 = 4;
        let entries = (u64::from(BUCKETS) * BUCKET_SIZE) as usize;
        let pages = colliding_pages(2 * entries, BUCKETS);
        let mut v = new(1.0, 1, 1, 0, 1, false, BUCKETS);
        for &page in &pages[..entries] {
            let (x, y, z) = voxel_of(page);
            v.set_voxel(x, y, z, 1);
        }
        for (n, &page) in pages[..entries].iter().enumerate() {
            let (x, y, z) = voxel_of(page);
            // The target lies in the next page, which is not stored
            assert_eq!(along_x(&mut v, x, x + 4, y, z, 1.0), vec![pi(x, y, z, 0)]);
            assert_eq!(v.pages().len(), entries - n - 1);
            if n + 1 < entries {
                assert_eq!(v.key_heads()[n], KEY_REMOVED);
            }
        }
        assert!(v.key_heads().iter().all(|&key| key == 0));
        assert_eq!(v.pending_results(), 0);

        for &page in &pages[entries..] {
            let (x, y, z) = voxel_of(page);
            assert_eq!(from_above(&mut v, (x, y, z), 1.0), vec![pi(x, y, z, 1)]);
        }
        assert_eq!(v.pages().len(), entries);
    }

    // ------------------------------------------------------------ debug

    #[test]
    fn debug_report_reads_valid_ranges() {
        // Compress output is on the block grid, so the report is fed made up
        // blocks after a real call: every range it reads back must be valid,
        // or wgpu panics just when the report is needed.
        let params = VoxelizerParams {
            resolution: 1.0,
            multipliers: Multipliers {
                intensity: 1.0,
                hit: 1,
                target: 1,
            },
            destroy_threshold: 0,
            construct_threshold: 1,
            max_value: u32::MAX,
            compress: true,
        };
        let mut v = pollster::block_on(WgpuAccelerated::with_debug(params, TEST_BUCKETS, true))
            .expect("a GPU adapter with compute shaders is required");
        let points: Vec<_> = cube_faces(40, 32, 8)
            .iter()
            .map(|&(x, y, z)| pf(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5, 1.0))
            .collect();
        let result = v.add_points(pf(0.5, 0.5, 0.5, 1.0), &points);
        assert!(!result.is_empty());
        assert!(
            !result
                .iter()
                .any(|b| off_block_grid(&[b.x as u32, b.y as u32, b.z as u32, b.i]))
        );
        let altered = pollster::block_on(v.read_words(v.encoder(), &v.altered, 0, 1))[0];
        assert!(altered > 0);
        let blocks = [[1, 0, 0, 5], [4, 4, 2, 1]];
        pollster::block_on(v.report_misaligned(
            [0.5; 3],
            points.len(),
            &blocks,
            altered,
            0,
            altered,
        ));
    }
}
