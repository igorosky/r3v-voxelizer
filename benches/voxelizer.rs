//! Runs every voxelizer implementation the features enable on the same
//! generated workloads and reports how long `add_points` takes. `bench.sh`
//! runs it with every implementation and plots the results, or run it
//! directly:
//!
//!   cargo bench --features multi_threaded,wgpu --bench voxelizer -- --quick
//!
//! `--help` lists the options. A run feeds one implementation a sequence of
//! frames, each one an `add_points` call from a moving sensor. Every
//! implementation gets the same frames, so their results are compared too:
//! the `match` column tells whether a run returned the same voxels as the
//! first run of the same workload (`ref`).

// Without any implementation there is nothing to run, most of this is unused
#![cfg_attr(
    not(any(
        feature = "single_threaded",
        feature = "multi_threaded",
        feature = "wgpu"
    )),
    allow(unused)
)]

mod common;

use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    panic::{self, AssertUnwindSafe},
    path::PathBuf,
    process,
    time::{Duration, Instant},
};

#[cfg(feature = "wgpu")]
use common::adapter;
use common::{
    DEFAULT_BUCKETS, Impl, MAX_INTENSITY, Measurement, PARAMS, Scenario, cpus, fail, impl_names,
    implementations, list, measure, number, workload,
};
#[cfg(feature = "wgpu")]
use r3v_voxelizer::WgpuAccelerated;
use r3v_voxelizer::{Multipliers, VoxelizerParams};

const USAGE: &str = "\
Usage: cargo bench --features <implementations> --bench voxelizer -- [options]

Lists are comma separated, defaults in brackets.
  --impls LIST      implementations: st, mt, wgpu [every one built]
  --threads LIST    thread counts of mt [1, 2, 4, ... up to the CPU threads]
  --scenarios LIST  room: sensor circling in a box room, rays in every
                    direction, some ending on obstacles inside the room
                    cone: static sensor, every ray ends on a small patch of
                    a wall (contention on the same voxels) [room,cone]
  --points LIST     points per frame [1000,10000,100000]
  --ranges LIST     distance from the room's center to its walls, in voxels,
                    at least 8 [16,64,256]
  --compress LIST   off, on [off,on]
  --frames N        timed frames per run [10]
  --warmup N        frames run before the timed ones [2]
  --buckets N       hashmap buckets of wgpu, about 1 KiB of GPU memory each
                    [131072]
  --csv PATH        also write the results to PATH
  --quick           smaller sweep: --points 1000,10000 --ranges 16,64
                    --frames 5 (other options override it)
  --help            print this
";

/// Frame durations in milliseconds, percentiles are nearest rank.
struct Stats {
    mean: f64,
    median: f64,
    p95: f64,
    min: f64,
    max: f64,
}

impl Stats {
    fn new(times: &[Duration]) -> Self {
        let mut ms: Vec<f64> = times.iter().map(|t| t.as_secs_f64() * 1e3).collect();
        ms.sort_by(f64::total_cmp);
        let rank = |q: f64| ms[((q * ms.len() as f64).ceil() as usize).clamp(1, ms.len()) - 1];
        Self {
            mean: ms.iter().sum::<f64>() / ms.len() as f64,
            median: rank(0.5),
            p95: rank(0.95),
            min: ms[0],
            max: ms[ms.len() - 1],
        }
    }
}

struct Options {
    /// `None` runs every implementation built
    impls: Option<Vec<String>>,
    threads: Vec<usize>,
    scenarios: Vec<Scenario>,
    points: Vec<usize>,
    ranges: Vec<u32>,
    compress: Vec<bool>,
    frames: usize,
    warmup: usize,
    buckets: u32,
    csv: Option<PathBuf>,
}

impl Options {
    fn parse() -> Self {
        let args: Vec<String> = std::env::args().skip(1).collect();
        let quick = args.iter().any(|arg| arg == "--quick");
        let cpus = cpus();
        let mut threads: Vec<_> = (0..)
            .map(|shift| 1 << shift)
            .take_while(|&threads| threads < cpus)
            .collect();
        threads.push(cpus);
        let mut options = Options {
            impls: None,
            threads,
            scenarios: vec![Scenario::Room, Scenario::Cone],
            points: if quick {
                vec![1_000, 10_000]
            } else {
                vec![1_000, 10_000, 100_000]
            },
            ranges: if quick {
                vec![16, 64]
            } else {
                vec![16, 64, 256]
            },
            compress: vec![false, true],
            frames: if quick { 5 } else { 10 },
            warmup: 2,
            buckets: DEFAULT_BUCKETS,
            csv: None,
        };
        let mut args = args.into_iter();
        while let Some(arg) = args.next() {
            let mut value = || {
                args.next()
                    .unwrap_or_else(|| fail(&format!("{arg} needs a value")))
            };
            match arg.as_str() {
                // cargo bench passes --bench
                "--bench" | "--quick" => {}
                "--help" | "-h" => {
                    print!("{USAGE}");
                    process::exit(0)
                }
                "--impls" => options.impls = Some(impl_names(&arg, &value())),
                "--threads" => {
                    options.threads = list(&arg, &value(), |n| n.parse().ok().filter(|&n| n > 0))
                }
                "--scenarios" => options.scenarios = list(&arg, &value(), Scenario::parse),
                "--points" => {
                    options.points = list(&arg, &value(), |n| n.parse().ok().filter(|&n| n > 0))
                }
                "--ranges" => {
                    options.ranges = list(&arg, &value(), |n| n.parse().ok().filter(|&n| n >= 8))
                }
                "--compress" => {
                    options.compress = list(&arg, &value(), |v| match v {
                        "off" => Some(false),
                        "on" => Some(true),
                        _ => None,
                    })
                }
                "--frames" => options.frames = number(&arg, &value()),
                "--warmup" => options.warmup = number(&arg, &value()),
                "--buckets" => options.buckets = number(&arg, &value()),
                "--csv" => options.csv = Some(value().into()),
                _ => fail(&format!("unknown option {arg}, see --help")),
            }
        }
        if options.frames == 0 {
            fail("--frames must be at least 1");
        }
        if options.buckets == 0 {
            fail("--buckets must be at least 1");
        }
        options
    }
}

/// What a run that did not panic reports.
struct Summary {
    setup_ms: f64,
    stats: Stats,
    frames: usize,
    mpoints_per_s: f64,
    msteps_per_s: f64,
    changed_per_frame: f64,
}

impl Summary {
    /// `steps` is the sum of the steps of the timed frames.
    fn new(m: &Measurement, points: usize, steps: u64) -> Self {
        let total = m.times.iter().sum::<Duration>().as_secs_f64();
        Self {
            setup_ms: m.setup.as_secs_f64() * 1e3,
            stats: Stats::new(&m.times),
            frames: m.times.len(),
            mpoints_per_s: (points * m.times.len()) as f64 / total / 1e6,
            msteps_per_s: steps as f64 / total / 1e6,
            changed_per_frame: m.changed.iter().sum::<usize>() as f64 / m.changed.len() as f64,
        }
    }
}

/// One line of the results.
struct Row<'a> {
    imp: &'a Impl,
    scenario: Scenario,
    points: usize,
    range: u32,
    compress: bool,
    /// `None` when the run panicked
    summary: Option<Summary>,
    /// `ref`, `yes`, `no` or `panicked`
    matches: &'static str,
}

const TABLE_HEADER: &str = "impl     scenario  points range compress  setup ms  median ms     \
                            p95 ms   Mpts/s  Msteps/s  changed  match";

const CSV_HEADER: &str = "impl,threads,buckets,scenario,points,range,compress,frames,\
    setup_ms,mean_ms,median_ms,p95_ms,min_ms,max_ms,mpoints_per_s,msteps_per_s,\
    changed_per_frame,matches";

impl Row<'_> {
    fn compress(&self) -> &'static str {
        if self.compress { "on" } else { "off" }
    }

    fn table(&self) -> String {
        let start = format!(
            "{:<8} {:<8} {:>7} {:>5} {:<8}",
            self.imp.label(),
            self.scenario.name(),
            self.points,
            self.range,
            self.compress()
        );
        let Some(s) = &self.summary else {
            return format!("{start}  panicked, see the message above");
        };
        format!(
            "{start} {:>9.1} {:>10.3} {:>10.3} {:>8.2} {:>9.1} {:>8.0}  {}",
            s.setup_ms,
            s.stats.median,
            s.stats.p95,
            s.mpoints_per_s,
            s.msteps_per_s,
            s.changed_per_frame,
            self.matches
        )
    }

    /// Numbers of a run that panicked are left empty.
    fn csv(&self) -> String {
        let numbers = self.summary.as_ref().map_or(",".repeat(9), |s| {
            format!(
                "{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.4},{:.4},{:.1}",
                s.frames,
                s.setup_ms,
                s.stats.mean,
                s.stats.median,
                s.stats.p95,
                s.stats.min,
                s.stats.max,
                s.mpoints_per_s,
                s.msteps_per_s,
                s.changed_per_frame
            )
        });
        format!(
            "{},{},{},{},{},{},{},{numbers},{}",
            self.imp.name,
            self.imp.threads.map_or(String::new(), |t| t.to_string()),
            self.imp.buckets.map_or(String::new(), |b| b.to_string()),
            self.scenario.name(),
            self.points,
            self.range,
            self.compress(),
            self.matches
        )
    }
}

fn main() {
    let options = Options::parse();
    let impls = implementations(options.impls.as_deref(), &options.threads, options.buckets);
    if impls.is_empty() {
        println!(
            "no implementation to run, enable the single_threaded, multi_threaded or wgpu \
             feature"
        );
        return;
    }

    let cpus = cpus();
    let labels: Vec<_> = impls.iter().map(Impl::label).collect();
    println!("cpu threads:     {cpus}");
    println!("implementations: {}", labels.join(", "));
    #[cfg(feature = "wgpu")]
    if impls.iter().any(|imp| imp.name == "wgpu") {
        let memory = WgpuAccelerated::memory_size(options.buckets) as f64 / (1 << 20) as f64;
        println!(
            "wgpu:            {} buckets ({memory:.0} MiB), adapter {}",
            options.buckets,
            adapter()
        );
    }
    println!(
        "frames:          {} warmup + {} timed per run",
        options.warmup, options.frames
    );
    let VoxelizerParams {
        resolution,
        multipliers: Multipliers { hit, target, .. },
        destroy_threshold,
        construct_threshold,
        max_value,
        ..
    } = PARAMS;
    println!(
        "params:          resolution {resolution}, hit {hit}, target {target}, \
         destroy {destroy_threshold}, construct {construct_threshold}, \
         max value {max_value}, intensities 1..={MAX_INTENSITY}"
    );
    println!("match:           whether a run returned the same voxels as the ref run");

    let mut csv = options.csv.as_ref().map(|path| {
        if let Some(dir) = path.parent() {
            fs::create_dir_all(dir)
                .unwrap_or_else(|err| fail(&format!("cannot create {}: {err}", dir.display())));
        }
        let mut file = BufWriter::new(
            File::create(path)
                .unwrap_or_else(|err| fail(&format!("cannot create {}: {err}", path.display()))),
        );
        writeln!(file, "{CSV_HEADER}").expect("writing the CSV failed");
        file
    });

    let started = Instant::now();
    let mut mismatches = 0;
    let mut panics = 0;
    for &scenario in &options.scenarios {
        for &range in &options.ranges {
            for &points in &options.points {
                let frames = workload(scenario, points, range, options.warmup + options.frames);
                let steps = frames[options.warmup..].iter().map(|f| f.steps).sum();
                for &compress in &options.compress {
                    println!();
                    println!("{TABLE_HEADER}");
                    let params = VoxelizerParams { compress, ..PARAMS };
                    let mut reference = None;
                    for imp in &impls {
                        // A panic (a bug of the implementation) ends only its run
                        let run = panic::catch_unwind(AssertUnwindSafe(|| {
                            measure(imp, params.clone(), &frames, options.warmup, None)
                        }));
                        let (summary, matches) = match run {
                            Ok(None) => {
                                println!("{:<8} could not be created, skipped", imp.label());
                                continue;
                            }
                            Ok(Some(m)) => {
                                let matches = match reference {
                                    None => {
                                        reference = Some(m.digest);
                                        "ref"
                                    }
                                    Some(digest) if digest == m.digest => "yes",
                                    Some(_) => {
                                        mismatches += 1;
                                        "no"
                                    }
                                };
                                (Some(Summary::new(&m, points, steps)), matches)
                            }
                            Err(_) => {
                                panics += 1;
                                (None, "panicked")
                            }
                        };
                        let row = Row {
                            imp,
                            scenario,
                            points,
                            range,
                            compress,
                            summary,
                            matches,
                        };
                        println!("{}", row.table());
                        if let Some(file) = csv.as_mut() {
                            writeln!(file, "{}", row.csv())
                                .and_then(|()| file.flush())
                                .expect("writing the CSV failed");
                        }
                    }
                }
            }
        }
    }
    println!();
    println!("done in {:.1} s", started.elapsed().as_secs_f64());
    if mismatches != 0 {
        println!(
            "{mismatches} runs returned other voxels than their ref run (a wgpu hashmap too \
             small for the workload drops voxels, see --buckets)"
        );
    }
    if panics != 0 {
        println!("{panics} runs panicked");
    }
}
