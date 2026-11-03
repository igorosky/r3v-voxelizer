//! Workloads, implementations and helpers shared by the benchmarks:
//! `voxelizer.rs` (the full sweep) and `check.rs` (the quick comparison with
//! earlier runs). A directory module, so that cargo does not take it for a
//! benchmark of its own.

// Each benchmark uses only part of this
#![allow(dead_code)]
// Without any implementation there is nothing to run, most of this is unused
#![cfg_attr(
    not(any(
        feature = "single_threaded",
        feature = "multi_threaded",
        feature = "wgpu"
    )),
    allow(unused)
)]

use std::{
    f32::consts::TAU,
    hash::{Hash, Hasher},
    num::NonZeroUsize,
    process,
    time::{Duration, Instant},
};

#[cfg(feature = "multi_threaded")]
use r3v_voxelizer::MultiThreaded;
#[cfg(feature = "single_threaded")]
use r3v_voxelizer::SingleThreaded;
#[cfg(feature = "wgpu")]
use r3v_voxelizer::WgpuAccelerated;
use r3v_voxelizer::{Multipliers, Voxelizer, VoxelizerParams, geometry::PointF};

/// Size of a voxel. Workloads are generated in voxels and scaled by it.
pub const RESOLUTION: f32 = 0.1;
/// `compress` is set by every run.
pub const PARAMS: VoxelizerParams = VoxelizerParams {
    resolution: RESOLUTION,
    multipliers: Multipliers {
        intensity: 1.0,
        hit: 1,
        target: 4,
    },
    destroy_threshold: 2,
    construct_threshold: 8,
    max_value: 64,
    compress: false,
};
/// Points get intensities `1..=MAX_INTENSITY`.
pub const MAX_INTENSITY: u32 = 10;
/// Points land up to this many voxels in front of the wall their ray hits,
/// so later rays pass through voxels earlier ones ended in.
const NOISE: f32 = 2.0;
/// Share of the rays of `room` ending on an obstacle inside the room.
const OBSTACLE_RATIO: f32 = 0.05;
/// Half size, in voxels, of the wall patch every ray of `cone` ends on.
const CONE_PATCH: f32 = 4.0;
pub const DEFAULT_BUCKETS: u32 = 1 << 17;

/// Tiny LCG (the one of the unit tests), so that workloads need no extra
/// dependency and are the same on every run.
struct Rng(u64);

impl Rng {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    /// Uniform in `0.0..1.0`.
    fn unit(&mut self) -> f32 {
        // 24 bits fit an f32 exactly
        (self.next_u32() >> 7) as f32 / (1 << 24) as f32
    }
}

/// FNV-1a. Digests of the results are compared with earlier runs, so unlike
/// `DefaultHasher` this one must not change between Rust versions.
pub struct StableHasher(u64);

impl StableHasher {
    pub fn new() -> Self {
        Self(0xcbf29ce484222325)
    }
}

impl Hasher for StableHasher {
    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 = (self.0 ^ byte as u64).wrapping_mul(0x100000001b3);
        }
    }
    fn finish(&self) -> u64 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Scenario {
    Room,
    Cone,
}

impl Scenario {
    pub fn name(self) -> &'static str {
        match self {
            Scenario::Room => "room",
            Scenario::Cone => "cone",
        }
    }
    pub fn parse(name: &str) -> Option<Self> {
        [Scenario::Room, Scenario::Cone]
            .into_iter()
            .find(|scenario| scenario.name() == name)
    }
}

/// One `add_points` call.
pub struct Frame {
    pub sensor: PointF,
    pub points: Vec<PointF>,
    /// Voxels the rays march through.
    pub steps: u64,
}

/// Uniformly distributed on the unit sphere.
fn random_direction(rng: &mut Rng) -> [f32; 3] {
    let z = 2.0 * rng.unit() - 1.0;
    let angle = TAU * rng.unit();
    let r = (1.0 - z * z).sqrt();
    [r * angle.cos(), r * angle.sin(), z]
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = v.iter().map(|c| c * c).sum::<f32>().sqrt();
    v.map(|c| c / len)
}

/// Distance from `origin` along `dir` to the walls of the box
/// `-half..half` that holds `origin`.
fn wall_distance(origin: [f32; 3], dir: [f32; 3], half: f32) -> f32 {
    origin
        .iter()
        .zip(dir)
        .filter(|(_, d)| *d != 0.0)
        .map(|(o, d)| (half.copysign(d) - o) / d)
        .fold(f32::INFINITY, f32::min)
}

/// Voxel holding `p`, computed the way the voxelizers do.
fn voxel(p: &PointF) -> [i64; 3] {
    [p.x, p.y, p.z].map(|c| (c / RESOLUTION).floor() as i64)
}

pub fn workload(scenario: Scenario, points: usize, range: u32, frames: usize) -> Vec<Frame> {
    let half = range as f32;
    let mut rng = Rng(0x5EED);
    (0..frames)
        .map(|frame| {
            let [x, y] = match scenario {
                // Circles a quarter of the way to the walls
                Scenario::Room => {
                    let angle = frame as f32 * 0.05;
                    [half / 4.0 * angle.cos(), half / 4.0 * angle.sin()]
                }
                Scenario::Cone => [0.0, 0.0],
            };
            // Off the voxel grid
            let origin = [x + 0.37, y + 0.61, 0.13];
            let points: Vec<_> = (0..points)
                .map(|_| {
                    let dir = match scenario {
                        Scenario::Room => random_direction(&mut rng),
                        Scenario::Cone => {
                            let spread = CONE_PATCH / half;
                            normalize([
                                1.0,
                                spread * (2.0 * rng.unit() - 1.0),
                                spread * (2.0 * rng.unit() - 1.0),
                            ])
                        }
                    };
                    let mut dist = wall_distance(origin, dir, half) - NOISE * rng.unit();
                    if scenario == Scenario::Room && rng.unit() < OBSTACLE_RATIO {
                        dist *= rng.unit();
                    }
                    let [x, y, z] = [0, 1, 2].map(|a| (origin[a] + dir[a] * dist) * RESOLUTION);
                    let i = (1 + rng.next_u32() % MAX_INTENSITY) as f32;
                    PointF { x, y, z, i }
                })
                .collect();
            let [x, y, z] = origin.map(|c| c * RESOLUTION);
            let sensor = PointF { x, y, z, i: 1.0 };
            let from = voxel(&sensor);
            let steps = points
                .iter()
                .map(|p| {
                    voxel(p)
                        .into_iter()
                        .zip(from)
                        .map(|(a, b)| a.abs_diff(b))
                        .sum::<u64>()
                })
                .sum();
            Frame {
                sensor,
                points,
                steps,
            }
        })
        .collect()
}

/// One implementation to benchmark.
pub struct Impl {
    /// `st`, `mt` or `wgpu`
    pub name: &'static str,
    /// Worker threads of `mt`
    pub threads: Option<usize>,
    /// Hashmap buckets of `wgpu`
    pub buckets: Option<u32>,
    /// `None` when the voxelizer cannot be created
    pub create: Box<dyn Fn(VoxelizerParams) -> Option<Box<dyn Voxelizer>>>,
}

impl Impl {
    pub fn label(&self) -> String {
        match self.threads {
            Some(threads) => format!("{}/{threads}", self.name),
            None => self.name.to_owned(),
        }
    }
}

/// The implementations built and named by `impls` (`None` for every one),
/// `mt` once per thread count.
pub fn implementations(
    impls: Option<&[String]>,
    #[cfg_attr(not(feature = "multi_threaded"), allow(unused_variables))] threads: &[usize],
    #[cfg_attr(not(feature = "wgpu"), allow(unused_variables))] buckets: u32,
) -> Vec<Impl> {
    let wanted = |name: &str| impls.is_none_or(|impls| impls.iter().any(|n| n == name));
    let mut list = Vec::new();
    #[cfg(feature = "single_threaded")]
    if wanted("st") {
        list.push(Impl {
            name: "st",
            threads: None,
            buckets: None,
            create: Box::new(|params| -> Option<Box<dyn Voxelizer>> {
                Some(Box::new(SingleThreaded::new(params)))
            }),
        });
    }
    #[cfg(feature = "multi_threaded")]
    if wanted("mt") {
        for &threads in threads {
            list.push(Impl {
                name: "mt",
                threads: Some(threads),
                buckets: None,
                create: Box::new(move |params| -> Option<Box<dyn Voxelizer>> {
                    let voxelizer = MultiThreaded::new(params, threads).ok()?;
                    Some(Box::new(voxelizer))
                }),
            });
        }
    }
    #[cfg(feature = "wgpu")]
    if wanted("wgpu") {
        list.push(Impl {
            name: "wgpu",
            threads: None,
            buckets: Some(buckets),
            create: Box::new(move |params| -> Option<Box<dyn Voxelizer>> {
                let voxelizer = pollster::block_on(WgpuAccelerated::new(params, buckets))?;
                Some(Box::new(voxelizer))
            }),
        });
    }
    for (name, feature, built) in [
        ("st", "single_threaded", cfg!(feature = "single_threaded")),
        ("mt", "multi_threaded", cfg!(feature = "multi_threaded")),
        ("wgpu", "wgpu", cfg!(feature = "wgpu")),
    ] {
        if !built && impls.is_some() && wanted(name) {
            println!("{name} skipped: built without the {feature} feature");
        }
    }
    list
}

/// Implementation features this benchmark was built with.
pub fn features() -> String {
    [
        ("single_threaded", cfg!(feature = "single_threaded")),
        ("multi_threaded", cfg!(feature = "multi_threaded")),
        ("wgpu", cfg!(feature = "wgpu")),
    ]
    .into_iter()
    .filter_map(|(name, built)| built.then_some(name))
    .collect::<Vec<_>>()
    .join(",")
}

/// The GPU adapter `WgpuAccelerated` picks, it asks for the default one too.
#[cfg(feature = "wgpu")]
pub fn adapter() -> String {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())) {
        Ok(adapter) => {
            let info = adapter.get_info();
            format!("{} ({:?}, {:?})", info.name, info.backend, info.device_type)
        }
        Err(err) => format!("none ({err})"),
    }
}

pub fn cpus() -> usize {
    std::thread::available_parallelism().map_or(1, NonZeroUsize::get)
}

pub struct Measurement {
    pub setup: Duration,
    /// Duration of every timed frame
    pub times: Vec<Duration>,
    /// Voxels returned by every timed frame
    pub changed: Vec<usize>,
    /// Hash of the sorted results of every frame
    pub digest: u64,
}

/// Runs every frame on a new voxelizer, timing the ones after `warmup`.
pub fn measure(
    imp: &Impl,
    params: VoxelizerParams,
    frames: &[Frame],
    warmup: usize,
) -> Option<Measurement> {
    let start = Instant::now();
    let mut voxelizer = (imp.create)(params)?;
    let setup = start.elapsed();
    let mut hasher = StableHasher::new();
    let mut times = Vec::new();
    let mut changed = Vec::new();
    for (n, frame) in frames.iter().enumerate() {
        let start = Instant::now();
        let mut result = voxelizer.add_points(frame.sensor.clone(), &frame.points);
        let time = start.elapsed();
        if n >= warmup {
            times.push(time);
            changed.push(result.len());
        }
        // Implementations return the voxels in different orders
        result.sort_unstable();
        result.hash(&mut hasher);
    }
    Some(Measurement {
        setup,
        times,
        changed,
        digest: hasher.finish(),
    })
}

pub fn fail(message: &str) -> ! {
    eprintln!("error: {message}");
    process::exit(2)
}

pub fn list<T>(flag: &str, value: &str, parse: impl Fn(&str) -> Option<T>) -> Vec<T> {
    value
        .split(',')
        .map(|item| {
            parse(item.trim()).unwrap_or_else(|| fail(&format!("invalid {flag} value: {item}")))
        })
        .collect()
}

pub fn number<T: std::str::FromStr>(flag: &str, value: &str) -> T {
    value
        .parse()
        .unwrap_or_else(|_| fail(&format!("invalid {flag} value: {value}")))
}

/// Parses `--impls`.
pub fn impl_names(flag: &str, value: &str) -> Vec<String> {
    list(flag, value, |name| {
        ["st", "mt", "wgpu"]
            .contains(&name)
            .then(|| name.to_owned())
    })
}
