//! Quick performance check for optimization work. Runs a small fixed set of
//! workloads (`compress` on) on every implementation built for about a
//! minute, then compares the run with a saved baseline and with the previous
//! run: whether each case got faster, slower or stayed within the noise, and
//! whether the voxels returned changed. `bench.sh --check` runs it, or:
//!
//!   cargo bench --features multi_threaded,wgpu --bench check -- --save-baseline
//!
//! `--help` lists the options. Runs are written to
//! `target/bench/check/runs/<unix seconds>.csv`, one line per rep, and
//! `--save-baseline` also copies the run to `target/bench/check/baseline.csv`.
//!
//! Every case runs once per round on a new voxelizer (a rep), rounds go on
//! until the time budget is used. Interleaving the cases spreads slow periods
//! (thermal throttling, background load) over all of them. Short cases run a
//! few times per round and record the median. On a CPU with performance and
//! efficiency cores the single threaded implementations (st and the host side
//! of wgpu) are pinned to one performance core: moved by the scheduler they
//! change speed about twice.

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
    fmt::Write as _,
    fs,
    panic::{self, AssertUnwindSafe},
    path::{Path, PathBuf},
    process::{self, Command},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use common::{
    DEFAULT_BUCKETS, Frame, Impl, PARAMS, Scenario, cpus, fail, features, impl_names,
    implementations, measure, number, workload,
};
use r3v_voxelizer::VoxelizerParams;

const USAGE: &str = "\
Usage: cargo bench --features <implementations> --bench check -- [options]

Runs a fixed set of cases for about a minute and compares the run with the
baseline and with the previous run. Defaults in brackets.
  --impls LIST       implementations, comma separated: st, mt, wgpu
                     [every one built]
  --threads N        thread count of mt [CPU threads]
  --budget SECONDS   time to spend on rounds, warm-up included, at least
                     --min-reps rounds run anyway [60]
  --warmup SECONDS   rounds run before the measured ones, at least one round
                     unless 0 [5]
  --min-reps N       measured rounds run whatever the budget [3]
  --threshold PCT    smallest change in percent reported as faster or slower,
                     noisier cases need more [5]
  --buckets N        hashmap buckets of wgpu [131072]
  --pin CPU|off      CPU st and wgpu run on, mt runs on every CPU [the last
                     performance core of a CPU with two kinds of cores, off
                     on other CPUs]
  --save-baseline    after the comparison, save this run as the baseline
  --baseline FILE    compare with FILE (a run of target/bench/check/runs)
                     instead of the saved baseline
  --no-save          do not save the run, the next run compares with the
                     same previous run (bench.sh --flamegraph passes it)
  --help             print this

Columns: ms/frame is the median over reps of the mean add_points duration
(add_points and get_downsampled in the -ds cases),
± the median absolute deviation of the reps in percent of it (keep it below
the changes you look for: close other programs, use a performance power
profile, run one implementation with --impls), blocks the
4x4x2 blocks returned per frame. A change is reported as faster or slower
only when it is larger than --threshold and twice the ± of both runs, ~
otherwise. output tells whether the voxels returned are the same as in the
baseline (the previous run without a baseline).
";

const DIR: &str = "target/bench/check";
const WARMUP: usize = 2;
const FRAMES: usize = 6;
const MAX_REPS: usize = 30;
/// A job shorter than this runs several times per round, its rep is the
/// median of them
const INNER_TARGET: Duration = Duration::from_millis(400);
const MAX_INNER: usize = 16;
/// Digest of a rep whose runs returned different voxels
const UNSTABLE: u64 = 0;

struct Case {
    name: &'static str,
    scenario: Scenario,
    points: usize,
    range: u32,
    /// Edge and threshold of the downsampling, `get_downsampled` after
    /// every `add_points`
    downsampling: Option<(u8, u32)>,
}

const CASES: [Case; 6] = [
    // A lidar frame in a small room
    Case {
        name: "room-50k",
        scenario: Scenario::Room,
        points: 50_000,
        range: 32,
        downsampling: None,
    },
    // Long rays
    Case {
        name: "room-10k-far",
        scenario: Scenario::Room,
        points: 10_000,
        range: 256,
        downsampling: None,
    },
    // Many rays ending on the same voxels
    Case {
        name: "cone-50k",
        scenario: Scenario::Cone,
        points: 50_000,
        range: 64,
        downsampling: None,
    },
    // Cost of a call itself
    Case {
        name: "room-1k",
        scenario: Scenario::Room,
        points: 1_000,
        range: 16,
        downsampling: None,
    },
    // Downsampling: edge 2 (a special case on the GPU) and edge 8
    Case {
        name: "room-50k-ds2",
        scenario: Scenario::Room,
        points: 50_000,
        range: 32,
        downsampling: Some((2, 2)),
    },
    Case {
        name: "room-50k-ds8",
        scenario: Scenario::Room,
        points: 50_000,
        range: 32,
        downsampling: Some((8, 8)),
    },
];

struct Options {
    impls: Option<Vec<String>>,
    threads: usize,
    budget: Duration,
    warmup: Duration,
    min_reps: usize,
    /// Fraction, not percent
    threshold: f64,
    buckets: u32,
    pin: Pin,
    save_baseline: bool,
    baseline: Option<PathBuf>,
    /// Not written to `runs`, it does not become the previous run
    no_save: bool,
}

#[derive(Clone, Copy)]
enum Pin {
    Auto,
    Off,
    Cpu(usize),
}

fn seconds(flag: &str, value: &str) -> Duration {
    let seconds: f64 = number(flag, value);
    if !seconds.is_finite() || seconds < 0.0 {
        fail(&format!("{flag} must be a number of seconds"));
    }
    Duration::from_secs_f64(seconds)
}

impl Options {
    fn parse() -> Self {
        let mut options = Options {
            impls: None,
            threads: cpus(),
            budget: Duration::from_secs(60),
            warmup: Duration::from_secs(5),
            min_reps: 3,
            threshold: 0.05,
            buckets: DEFAULT_BUCKETS,
            pin: Pin::Auto,
            save_baseline: false,
            baseline: None,
            no_save: false,
        };
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            let mut value = || {
                args.next()
                    .unwrap_or_else(|| fail(&format!("{arg} needs a value")))
            };
            match arg.as_str() {
                // cargo bench passes --bench
                "--bench" => {}
                "--help" | "-h" => {
                    print!("{USAGE}");
                    process::exit(0)
                }
                "--impls" => options.impls = Some(impl_names(&arg, &value())),
                "--threads" => options.threads = number(&arg, &value()),
                "--budget" => options.budget = seconds(&arg, &value()),
                "--warmup" => options.warmup = seconds(&arg, &value()),
                "--min-reps" => options.min_reps = number(&arg, &value()),
                "--threshold" => {
                    let percent: f64 = number(&arg, &value());
                    if !percent.is_finite() || percent < 0.0 {
                        fail("--threshold must be a percentage");
                    }
                    options.threshold = percent / 100.0;
                }
                "--buckets" => options.buckets = number(&arg, &value()),
                "--pin" => {
                    options.pin = match value().as_str() {
                        "off" => Pin::Off,
                        cpu => Pin::Cpu(number(&arg, cpu)),
                    }
                }
                "--save-baseline" => options.save_baseline = true,
                "--baseline" => options.baseline = Some(value().into()),
                "--no-save" => options.no_save = true,
                _ => fail(&format!("unknown option {arg}, see --help")),
            }
        }
        if options.no_save && options.save_baseline {
            fail("--no-save and --save-baseline exclude each other");
        }
        if options.threads == 0 {
            fail("--threads must be at least 1");
        }
        if options.min_reps == 0 || options.min_reps > MAX_REPS {
            fail(&format!("--min-reps must be between 1 and {MAX_REPS}"));
        }
        if options.buckets == 0 {
            fail("--buckets must be at least 1");
        }
        options
    }
}

/// Keeps a thread on chosen CPUs.
#[cfg(target_os = "linux")]
mod affinity {
    use std::{fs, mem};

    pub struct CpuSet(libc::cpu_set_t);

    impl CpuSet {
        /// CPUs the calling thread may run on.
        pub fn current() -> Option<Self> {
            // SAFETY: a zeroed cpu_set_t is an empty set
            let mut set: libc::cpu_set_t = unsafe { mem::zeroed() };
            // SAFETY: set is a cpu_set_t of the size passed
            let result = unsafe { libc::sched_getaffinity(0, mem::size_of_val(&set), &mut set) };
            (result == 0).then_some(CpuSet(set))
        }

        /// `None` when `cpu` does not fit a set.
        pub fn one(cpu: usize) -> Option<Self> {
            if cpu >= libc::CPU_SETSIZE as usize {
                return None;
            }
            // SAFETY: a zeroed cpu_set_t is an empty set, cpu is in range
            unsafe {
                let mut set: libc::cpu_set_t = mem::zeroed();
                libc::CPU_SET(cpu, &mut set);
                Some(CpuSet(set))
            }
        }

        pub fn contains(&self, cpu: usize) -> bool {
            // SAFETY: cpu is in range
            cpu < libc::CPU_SETSIZE as usize && unsafe { libc::CPU_ISSET(cpu, &self.0) }
        }

        /// Moves the calling thread to these CPUs. Threads it starts later
        /// inherit them, so it has to run before a thread pool is created.
        pub fn apply(&self) {
            // SAFETY: self.0 is a cpu_set_t of the size passed
            unsafe { libc::sched_setaffinity(0, mem::size_of_val(&self.0), &self.0) };
        }
    }

    /// The last performance core of a CPU with performance and efficiency
    /// cores the calling thread may run on.
    pub fn performance_core(allowed: &CpuSet) -> Option<usize> {
        let list = fs::read_to_string("/sys/devices/cpu_core/cpus").ok()?;
        let mut last = None;
        // A CPU list: 0-7,12
        for range in list.trim().split(',') {
            let (first, end) = range.split_once('-').unwrap_or((range, range));
            for cpu in first.parse::<usize>().ok()?..=end.parse().ok()? {
                if allowed.contains(cpu) {
                    last = Some(cpu);
                }
            }
        }
        last
    }
}

/// Pinning is Linux only.
#[cfg(not(target_os = "linux"))]
mod affinity {
    pub struct CpuSet;

    impl CpuSet {
        pub fn current() -> Option<Self> {
            None
        }
        pub fn one(_: usize) -> Option<Self> {
            None
        }
        pub fn contains(&self, _: usize) -> bool {
            false
        }
        pub fn apply(&self) {}
    }

    pub fn performance_core(_: &CpuSet) -> Option<usize> {
        None
    }
}

/// CPUs jobs run on: single threaded ones on `pinned`, the others on `all`.
struct Affinity {
    all: Option<affinity::CpuSet>,
    pinned: Option<affinity::CpuSet>,
}

impl Affinity {
    /// Also returns the pinned CPU.
    fn new(pin: Pin) -> (Self, Option<usize>) {
        let all = affinity::CpuSet::current();
        let cpu = match pin {
            Pin::Off => None,
            Pin::Auto => all.as_ref().and_then(affinity::performance_core),
            Pin::Cpu(cpu) => {
                if !all.as_ref().is_some_and(|all| all.contains(cpu)) {
                    fail(&format!(
                        "cannot pin to CPU {cpu}, not one this process may use"
                    ));
                }
                Some(cpu)
            }
        };
        let pinned = cpu.and_then(affinity::CpuSet::one);
        (Affinity { all, pinned }, cpu)
    }

    fn apply(&self, imp: &Impl) {
        let set = if imp.name == "mt" {
            &self.all
        } else {
            &self.pinned
        };
        if let Some(set) = set.as_ref().or(self.all.as_ref()) {
            set.apply();
        }
    }
}

/// One case run once on a new voxelizer.
struct Rep {
    case: String,
    /// `st`, `mt/16`, ...
    imp: String,
    ms_per_frame: f64,
    msteps_per_s: f64,
    changed_per_frame: f64,
    digest: u64,
}

/// A run as written to and read from its CSV.
struct Run {
    /// `# key: value` lines of the CSV: commit, dirty, cpu, features, adapter,
    /// time
    meta: Vec<(String, String)>,
    reps: Vec<Rep>,
}

const CSV_HEADER: &str = "case,impl,rep,ms_per_frame,msteps_per_s,changed_per_frame,digest";

impl Run {
    fn meta(&self, key: &str) -> Option<&str> {
        self.meta
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    fn to_csv(&self) -> String {
        let mut csv = String::new();
        for (key, value) in &self.meta {
            writeln!(csv, "# {key}: {value}").unwrap();
        }
        writeln!(csv, "{CSV_HEADER}").unwrap();
        let mut counts: Vec<(&str, &str, usize)> = Vec::new();
        for rep in &self.reps {
            let n = match counts
                .iter_mut()
                .find(|(c, i, _)| *c == rep.case && *i == rep.imp)
            {
                Some((_, _, n)) => {
                    *n += 1;
                    *n
                }
                None => {
                    counts.push((&rep.case, &rep.imp, 1));
                    1
                }
            };
            writeln!(
                csv,
                "{},{},{n},{:.4},{:.3},{:.1},{:016x}",
                rep.case,
                rep.imp,
                rep.ms_per_frame,
                rep.msteps_per_s,
                rep.changed_per_frame,
                rep.digest
            )
            .unwrap();
        }
        csv
    }

    fn read(path: &Path) -> Result<Self, String> {
        let text = fs::read_to_string(path).map_err(|err| err.to_string())?;
        let mut meta = Vec::new();
        let mut reps = Vec::new();
        for (n, line) in text.lines().enumerate() {
            if let Some(comment) = line.strip_prefix('#') {
                if let Some((key, value)) = comment.split_once(':') {
                    meta.push((key.trim().to_owned(), value.trim().to_owned()));
                }
                continue;
            }
            if line.is_empty() || line == CSV_HEADER {
                continue;
            }
            let bad = || format!("line {}: cannot parse {line:?}", n + 1);
            let fields: Vec<_> = line.split(',').collect();
            let [case, imp, _, ms, msteps, changed, digest] = fields[..] else {
                return Err(bad());
            };
            let float = |s: &str| s.parse::<f64>().map_err(|_| bad());
            reps.push(Rep {
                case: case.to_owned(),
                imp: imp.to_owned(),
                ms_per_frame: float(ms)?,
                msteps_per_s: float(msteps)?,
                changed_per_frame: float(changed)?,
                digest: u64::from_str_radix(digest, 16).map_err(|_| bad())?,
            });
        }
        Ok(Run { meta, reps })
    }
}

/// The reps of one (case, implementation) of a run.
struct Slice {
    case: String,
    imp: String,
    reps: usize,
    median: f64,
    /// Median absolute deviation of the reps, relative to `median`
    mad: f64,
    msteps_per_s: f64,
    changed_per_frame: f64,
    /// `None` when reps returned different voxels
    digest: Option<u64>,
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    }
}

/// How much slower the last third of the reps of every slice ran than its
/// first third, geometric mean over the slices. `None` with too few reps.
fn drift(reps: &[Rep], current: &[Slice]) -> Option<f64> {
    let ratios: Vec<f64> = current
        .iter()
        .filter_map(|slice| {
            let ms: Vec<_> = reps
                .iter()
                .filter(|r| r.case == slice.case && r.imp == slice.imp)
                .map(|r| r.ms_per_frame)
                .collect();
            let third = ms.len() / 3;
            (third >= 2).then(|| {
                median(&mut ms[ms.len() - third..].to_vec()) / median(&mut ms[..third].to_vec())
            })
        })
        .collect();
    (!ratios.is_empty())
        .then(|| (ratios.iter().map(|r| r.ln()).sum::<f64>() / ratios.len() as f64).exp())
}

/// Slices in the order their first rep ran.
fn slices(reps: &[Rep]) -> Vec<Slice> {
    let mut keys: Vec<(&str, &str)> = Vec::new();
    for rep in reps {
        if !keys.contains(&(&rep.case, &rep.imp)) {
            keys.push((&rep.case, &rep.imp));
        }
    }
    keys.into_iter()
        .map(|(case, imp)| {
            let reps: Vec<_> = reps
                .iter()
                .filter(|r| r.case == case && r.imp == imp)
                .collect();
            let mut ms: Vec<_> = reps.iter().map(|r| r.ms_per_frame).collect();
            let med = median(&mut ms);
            let mut deviations: Vec<_> = ms.iter().map(|v| (v - med).abs()).collect();
            let mad = if med > 0.0 {
                median(&mut deviations) / med
            } else {
                0.0
            };
            let digest = reps[0].digest;
            Slice {
                case: case.to_owned(),
                imp: imp.to_owned(),
                reps: reps.len(),
                median: med,
                mad,
                msteps_per_s: median(&mut reps.iter().map(|r| r.msteps_per_s).collect::<Vec<_>>()),
                changed_per_frame: reps[0].changed_per_frame,
                digest: (digest != UNSTABLE && reps.iter().all(|r| r.digest == digest))
                    .then_some(digest),
            }
        })
        .collect()
}

#[derive(Clone, Copy, PartialEq)]
enum Verdict {
    Faster,
    Slower,
    Noise,
}

/// `current / reference` of the median and whether that is more than noise.
/// `ratio` (current / reference) is noise unless it leaves `1 ± band`.
struct Comparison {
    ratio: f64,
    band: f64,
    verdict: Verdict,
}

impl Comparison {
    fn new(ratio: f64, band: f64) -> Self {
        let verdict = if ratio < 1.0 - band {
            Verdict::Faster
        } else if ratio > 1.0 + band {
            Verdict::Slower
        } else {
            Verdict::Noise
        };
        Comparison {
            ratio,
            band,
            verdict,
        }
    }
}

fn compare(current: &Slice, reference: &Slice, threshold: f64) -> Comparison {
    let band = threshold.max(2.0 * current.mad.max(reference.mad));
    Comparison::new(current.median / reference.median, band)
}

/// A run to compare with.
struct Reference {
    /// `baseline` or `previous`
    name: &'static str,
    path: PathBuf,
    run: Run,
    slices: Vec<Slice>,
}

impl Reference {
    fn load(name: &'static str, path: PathBuf) -> Option<Self> {
        match Run::read(&path) {
            Ok(run) => Some(Reference {
                name,
                slices: slices(&run.reps),
                path,
                run,
            }),
            Err(err) => {
                println!("cannot read the {name} {}: {err}", path.display());
                None
            }
        }
    }

    fn find(&self, slice: &Slice) -> Option<&Slice> {
        self.slices
            .iter()
            .find(|s| s.case == slice.case && s.imp == slice.imp)
    }

    /// One line on what this reference is.
    fn describe(&self, now: u64) -> String {
        let mut text = format!(
            "{:<9} {}, commit {}",
            format!("{}:", self.name),
            self.path.display(),
            self.run.meta("commit").unwrap_or("unknown")
        );
        if self.run.meta("dirty") == Some("yes") {
            text += " with uncommitted changes";
        }
        if let Some(time) = self.run.meta("time").and_then(|t| t.parse::<u64>().ok()) {
            text += &format!(", {} ago", age(now.saturating_sub(time)));
        }
        text
    }

    /// Lines on settings that make the comparison doubtful.
    fn mismatches(&self, run: &Run) -> Vec<String> {
        ["cpu", "impls", "pin", "profile", "adapter", "buckets"]
            .into_iter()
            .filter_map(|key| {
                let theirs = self.run.meta(key);
                let ours = run.meta(key);
                (theirs.is_some() && ours.is_some() && theirs != ours).then(|| {
                    let mut text = format!(
                        "warning: the {} ran with {key} {}, this run with {}",
                        self.name,
                        theirs.unwrap(),
                        ours.unwrap()
                    );
                    if key == "impls" {
                        // Heat of the others changes the clocks one runs at
                        text += ", an implementation runs at other clocks next to others, \
                                 compare runs of the same --impls";
                    }
                    text
                })
            })
            .collect()
    }
}

fn age(seconds: u64) -> String {
    match seconds {
        0..120 => format!("{seconds} s"),
        120..7200 => format!("{} min", seconds / 60),
        7200..172800 => format!("{} h", seconds / 3600),
        _ => format!("{} days", seconds / 86400),
    }
}

/// `-12.4% faster`, `+0.4% ~`, `new`
fn change(current: &Slice, reference: Option<&Reference>, threshold: f64) -> String {
    let Some(reference) = reference else {
        return String::new();
    };
    let Some(theirs) = reference.find(current) else {
        return "new".to_owned();
    };
    let Comparison { ratio, verdict, .. } = compare(current, theirs, threshold);
    let word = match verdict {
        Verdict::Faster => "faster",
        Verdict::Slower => "slower",
        Verdict::Noise => "~",
    };
    format!("{:+.1}% {word}", (ratio - 1.0) * 100.0)
}

fn output(current: &Slice, reference: Option<&Reference>) -> &'static str {
    let Some(digest) = current.digest else {
        return "unstable";
    };
    match reference.and_then(|r| r.find(current)) {
        None => "-",
        Some(theirs) if theirs.digest == Some(digest) => "same",
        Some(_) => "CHANGED",
    }
}

fn git(args: &[&str]) -> Option<process::Output> {
    Command::new("git").args(args).output().ok()
}

fn cpu_model() -> String {
    fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|info| {
            info.lines()
                .find_map(|line| line.strip_prefix("model name"))
                .and_then(|rest| rest.split_once(':'))
                .map(|(_, model)| model.trim().to_owned())
        })
        .unwrap_or_else(|| "unknown".to_owned())
}

/// ACPI platform profile (quiet, balanced, performance, ...), it sets how
/// long the CPU sustains load before it throttles.
fn power_profile() -> String {
    fs::read_to_string("/sys/firmware/acpi/platform_profile")
        .map_or_else(|_| "unknown".to_owned(), |p| p.trim().to_owned())
}

/// Adapter and buckets, when wgpu runs.
#[cfg_attr(not(feature = "wgpu"), allow(unused_variables))]
fn wgpu_meta(options: &Options, impls: &[Impl]) -> Vec<(String, String)> {
    #[cfg(feature = "wgpu")]
    if impls.iter().any(|imp| imp.name == "wgpu") {
        return vec![
            ("adapter".to_owned(), common::adapter()),
            ("buckets".to_owned(), options.buckets.to_string()),
        ];
    }
    Vec::new()
}

fn meta(options: &Options, impls: &[Impl], time: u64) -> Vec<(String, String)> {
    let commit = git(&["rev-parse", "--short", "HEAD"])
        .filter(|out| out.status.success())
        .map_or("unknown".to_owned(), |out| {
            String::from_utf8_lossy(&out.stdout).trim().to_owned()
        });
    let dirty = git(&["diff", "--quiet", "HEAD"]).is_some_and(|out| out.status.code() == Some(1));
    let mut meta = vec![
        ("commit".to_owned(), commit),
        (
            "dirty".to_owned(),
            if dirty { "yes" } else { "no" }.to_owned(),
        ),
        ("cpu".to_owned(), cpu_model()),
        ("features".to_owned(), features()),
        (
            "impls".to_owned(),
            impls.iter().map(Impl::label).collect::<Vec<_>>().join(" "),
        ),
        ("profile".to_owned(), power_profile()),
        ("time".to_owned(), time.to_string()),
    ];
    meta.extend(wgpu_meta(options, impls));
    meta
}

/// The newest run in `runs`.
fn previous_run(runs: &Path) -> Option<PathBuf> {
    fs::read_dir(runs)
        .ok()?
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            let stamp: u64 = path
                .file_name()?
                .to_str()?
                .strip_suffix(".csv")?
                .parse()
                .ok()?;
            Some((stamp, path))
        })
        .max_by_key(|(stamp, _)| *stamp)
        .map(|(_, path)| path)
}

/// One (case, implementation) pair to run every round.
struct Job<'a> {
    case: &'a Case,
    frames: &'a [Frame],
    steps: u64,
    imp: &'a Impl,
    /// Runs per round
    inner: usize,
    /// Wall time of its last run, setup included
    wall: Duration,
    /// Set when it panicked or could not be created, it runs no more
    done: bool,
}

/// Runs every job `inner` times, recording a rep each into `reps`.
fn round(
    jobs: &mut [Job],
    params: &VoxelizerParams,
    affinity: &Affinity,
    mut reps: Option<&mut Vec<Rep>>,
) {
    for job in jobs.iter_mut().filter(|job| !job.done) {
        affinity.apply(job.imp);
        let mut runs = Vec::new();
        for _ in 0..job.inner {
            let start = Instant::now();
            let measured = panic::catch_unwind(AssertUnwindSafe(|| {
                measure(
                    job.imp,
                    params.clone(),
                    job.frames,
                    WARMUP,
                    job.case.downsampling,
                )
            }));
            job.wall = start.elapsed();
            match measured {
                Ok(Some(m)) => runs.push(m),
                Ok(None) => {
                    println!("{} could not be created, skipped", job.imp.label());
                    job.done = true;
                    break;
                }
                Err(_) => {
                    println!(
                        "{} panicked on {}, skipped from now on",
                        job.imp.label(),
                        job.case.name
                    );
                    job.done = true;
                    break;
                }
            }
        }
        let Some(reps) = reps.as_deref_mut().filter(|_| !job.done) else {
            continue;
        };
        let mut totals: Vec<_> = runs
            .iter()
            .map(|m| m.times.iter().sum::<Duration>().as_secs_f64())
            .collect();
        let total = median(&mut totals);
        let first = &runs[0];
        reps.push(Rep {
            case: job.case.name.to_owned(),
            imp: job.imp.label(),
            ms_per_frame: total * 1e3 / first.times.len() as f64,
            msteps_per_s: job.steps as f64 / total / 1e6,
            changed_per_frame: first.changed.iter().sum::<usize>() as f64
                / first.changed.len() as f64,
            digest: if runs.iter().all(|m| m.digest == first.digest) {
                first.digest
            } else {
                UNSTABLE
            },
        });
    }
}

/// Runs short jobs several times per round, from their last wall time.
fn plan_inner(jobs: &mut [Job]) {
    for job in jobs {
        let fit = INNER_TARGET.as_secs_f64() / job.wall.as_secs_f64().max(1e-6);
        job.inner = (fit.ceil() as usize).clamp(1, MAX_INNER);
    }
}

fn main() {
    let options = Options::parse();
    let impls = implementations(
        options.impls.as_deref(),
        &[options.threads],
        options.buckets,
    );
    if impls.is_empty() {
        println!(
            "no implementation to run, enable the single_threaded, multi_threaded or wgpu \
             feature"
        );
        return;
    }
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let (affinity, pinned) = Affinity::new(options.pin);
    let mut run_meta = meta(&options, &impls, now);
    let pin = pinned.map_or("off".to_owned(), |cpu| format!("cpu {cpu}"));
    run_meta.push(("pin".to_owned(), pin));
    let labels: Vec<_> = impls.iter().map(Impl::label).collect();
    println!("implementations: {}", labels.join(", "));
    for (key, value) in &run_meta {
        if !matches!(key.as_str(), "time" | "features" | "impls") {
            println!("{:<16} {value}", format!("{key}:"));
        }
    }
    println!(
        "per rep:         new voxelizer, {WARMUP} warmup + {FRAMES} timed frames, compress on"
    );

    let workloads: Vec<_> = CASES
        .iter()
        .map(|case| {
            let frames = workload(case.scenario, case.points, case.range, WARMUP + FRAMES);
            let steps = frames[WARMUP..].iter().map(|f| f.steps).sum::<u64>();
            (frames, steps)
        })
        .collect();
    // The cases of an implementation one after the other, so that a GPU
    // keeps its clocks up between them
    let mut jobs: Vec<_> = impls
        .iter()
        .flat_map(|imp| {
            CASES
                .iter()
                .zip(&workloads)
                .map(move |(case, (frames, steps))| Job {
                    case,
                    frames,
                    steps: *steps,
                    imp,
                    inner: 1,
                    wall: Duration::ZERO,
                    done: false,
                })
        })
        .collect();

    let params = VoxelizerParams {
        compress: true,
        ..PARAMS
    };
    let started = Instant::now();
    // A CPU coming out of idle runs the first seconds of load several times
    // slower, those rounds are thrown away
    let mut warmup_rounds = 0;
    while started.elapsed() < options.warmup && jobs.iter().any(|job| !job.done) {
        round(&mut jobs, &params, &affinity, None);
        plan_inner(&mut jobs);
        warmup_rounds += 1;
    }
    if warmup_rounds > 0 {
        println!(
            "warm-up:         {warmup_rounds} rounds in {:.1} s, not measured",
            started.elapsed().as_secs_f64()
        );
    }
    let mut reps = Vec::new();
    let measuring = Instant::now();
    let mut rounds = 0;
    while jobs.iter().any(|job| !job.done) {
        round(&mut jobs, &params, &affinity, Some(&mut reps));
        if warmup_rounds == 0 && rounds == 0 {
            plan_inner(&mut jobs);
        }
        rounds += 1;
        let per_round = measuring.elapsed() / rounds as u32;
        let left = options.budget.saturating_sub(started.elapsed());
        if rounds == 1 {
            let fit = 1 + (left.as_secs_f64() / per_round.as_secs_f64()) as usize;
            println!(
                "round 1 took {:.1} s, about {} rounds fit",
                per_round.as_secs_f64(),
                fit.clamp(options.min_reps, MAX_REPS)
            );
        }
        if rounds >= options.min_reps && (rounds >= MAX_REPS || per_round > left) {
            break;
        }
    }
    println!(
        "{rounds} rounds in {:.1} s",
        measuring.elapsed().as_secs_f64()
    );
    if reps.is_empty() {
        println!("nothing ran");
        process::exit(1);
    }

    let run = Run {
        meta: run_meta,
        reps,
    };
    let current = slices(&run.reps);
    let runs = Path::new(DIR).join("runs");
    let previous = previous_run(&runs).and_then(|path| Reference::load("previous", path));
    let baseline_path = options
        .baseline
        .clone()
        .unwrap_or_else(|| Path::new(DIR).join("baseline.csv"));
    let baseline = if options.baseline.is_some() || baseline_path.exists() {
        Reference::load("baseline", baseline_path)
    } else {
        None
    };
    report(
        &run,
        &current,
        baseline.as_ref(),
        previous.as_ref(),
        &options,
        now,
    );

    if options.no_save {
        println!();
        println!("run not saved (--no-save)");
        return;
    }
    let path = runs.join(format!("{now}.csv"));
    let saved = fs::create_dir_all(&runs).and_then(|()| fs::write(&path, run.to_csv()));
    if let Err(err) = saved {
        fail(&format!("cannot write {}: {err}", path.display()));
    }
    println!();
    println!("run saved to {}", path.display());
    if options.save_baseline {
        let to = Path::new(DIR).join("baseline.csv");
        if let Err(err) = fs::copy(&path, &to) {
            fail(&format!("cannot write {}: {err}", to.display()));
        }
        println!("saved as the baseline {}", to.display());
    } else if baseline.is_none() {
        println!("no baseline yet, --save-baseline saves this run as one");
    }
}

fn report(
    run: &Run,
    current: &[Slice],
    baseline: Option<&Reference>,
    previous: Option<&Reference>,
    options: &Options,
    now: u64,
) {
    println!();
    for reference in [baseline, previous].into_iter().flatten() {
        println!("{}", reference.describe(now));
    }
    if baseline.is_some() || previous.is_some() {
        println!();
    }
    println!(
        "{:<13} {:<7} {:>9} {:>6} {:>9} {:>7}  {:<15} {:<15} output",
        "case", "impl", "ms/frame", "±", "Msteps/s", "blocks", "vs baseline", "vs previous"
    );
    // Output is compared with the baseline, or the previous run without one
    let output_reference = baseline.or(previous);
    for slice in current {
        println!(
            "{:<13} {:<7} {:>9.3} {:>5.1}% {:>9.1} {:>7.0}  {:<15} {:<15} {}",
            slice.case,
            slice.imp,
            slice.median,
            slice.mad * 100.0,
            slice.msteps_per_s,
            slice.changed_per_frame,
            change(slice, baseline, options.threshold),
            change(slice, previous, options.threshold),
            output(slice, output_reference)
        );
    }
    println!(
        "{} reps per case, threshold {:.1}%",
        current.iter().map(|s| s.reps).min().unwrap_or(0),
        options.threshold * 100.0
    );

    // Overall change of every implementation over the cases
    if let Some(reference) = baseline.or(previous) {
        println!();
        println!(
            "overall vs {} (geometric mean of the cases):",
            reference.name
        );
        let mut imps: Vec<&str> = Vec::new();
        for slice in current {
            if !imps.contains(&slice.imp.as_str()) {
                imps.push(&slice.imp);
            }
        }
        for imp in imps {
            let compared: Vec<_> = current
                .iter()
                .filter(|s| s.imp == imp)
                .filter_map(|s| Some(compare(s, reference.find(s)?, options.threshold)))
                .collect();
            if compared.is_empty() {
                println!("  {imp:<7} not in the {}", reference.name);
                continue;
            }
            let n = compared.len() as f64;
            let ratio = (compared.iter().map(|c| c.ratio.ln()).sum::<f64>() / n).exp();
            // Noise of the cases is mostly common to them (the CPU got hot),
            // so it does not average out
            let band = median(&mut compared.iter().map(|c| c.band).collect::<Vec<_>>());
            let overall = Comparison::new(ratio, band);
            let text = match overall.verdict {
                Verdict::Faster => format!("{:.1}% faster", (1.0 / ratio - 1.0) * 100.0),
                Verdict::Slower => format!("{:.1}% slower", (ratio - 1.0) * 100.0),
                Verdict::Noise => format!("within noise ({:+.1}%)", (ratio - 1.0) * 100.0),
            };
            let count = |v| compared.iter().filter(|c| c.verdict == v).count();
            println!(
                "  {imp:<7} {text:<22} band ±{:.1}%, {} faster, {} ~, {} slower",
                overall.band * 100.0,
                count(Verdict::Faster),
                count(Verdict::Noise),
                count(Verdict::Slower)
            );
        }
    }

    // What makes the numbers doubtful or the change wrong
    let mut notes = Vec::new();
    let limit = options.threshold.max(0.05);
    match drift(&run.reps, current) {
        Some(drift) if drift > 1.0 + limit => {
            let mut note = format!(
                "warning: the last third of the reps ran {:.0}% slower than the first third \
                 (thermal throttling or other load), the ± and so the bands grew",
                (drift - 1.0) * 100.0
            );
            if matches!(run.meta("profile"), Some("quiet" | "low-power")) {
                note += ", the balanced or performance power profile throttles less";
            }
            notes.push(note);
        }
        Some(drift) if drift < 1.0 - limit => notes.push(format!(
            "warning: the last third of the reps ran {:.0}% faster than the first third, the \
             CPU was still warming up, raise --warmup",
            (1.0 / drift - 1.0) * 100.0
        )),
        _ => {}
    }
    for reference in [baseline, previous].into_iter().flatten() {
        notes.extend(reference.mismatches(run));
    }
    let changed: Vec<_> = current
        .iter()
        .filter(|s| output(s, output_reference) == "CHANGED")
        .map(|s| format!("{} {}", s.imp, s.case))
        .collect();
    if !changed.is_empty() {
        notes.push(format!(
            "WARNING: other voxels than in the {} ({}), the change alters the results, run the \
             tests",
            output_reference.map_or("reference", |r| r.name),
            changed.join(", ")
        ));
    }
    for slice in current.iter().filter(|s| s.digest.is_none()) {
        notes.push(format!(
            "WARNING: {} returned different voxels between reps of {}",
            slice.imp, slice.case
        ));
    }
    for case in &CASES {
        // Implementations grouped by the voxels they returned
        let mut groups: Vec<(u64, Vec<&str>)> = Vec::new();
        for slice in current.iter().filter(|s| s.case == case.name) {
            let Some(digest) = slice.digest else { continue };
            match groups.iter_mut().find(|(d, _)| *d == digest) {
                Some((_, imps)) => imps.push(&slice.imp),
                None => groups.push((digest, vec![&slice.imp])),
            }
        }
        if groups.len() > 1 {
            let groups: Vec<_> = groups.iter().map(|(_, imps)| imps.join(", ")).collect();
            notes.push(format!(
                "note: implementations return different voxels on {}: {} (a wgpu hashmap too \
                 small drops voxels, see --buckets)",
                case.name,
                groups.join(" vs ")
            ));
        }
    }
    if !notes.is_empty() {
        println!();
        for note in notes {
            println!("{note}");
        }
    }
}
