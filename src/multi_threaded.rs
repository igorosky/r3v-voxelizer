//! Multi threaded voxelizer. The fine pages are split into [`SHARDS`] shards
//! by their key, so that every phase of [`MultiThreaded::add_points`] but the
//! (small) update of the compressed pages runs in parallel without locks:
//!
//! 1. The rays are marched in parallel, the voxelizer read only. Every step
//!    that lands in an existing page is recorded as a [`Hit`] in a buffer of
//!    the page's shard, every target as a [`Target`]. Steps in space without
//!    a page are dropped right away, as the single threaded voxelizer drops
//!    them.
//! 2. Every shard applies its hits and then its targets, in parallel with
//!    the others, and compares the pages it touched with their snapshots
//!    taken at the first touch: changed voxels, or with `compress` the
//!    threshold crossings.
//! 3. The crossings update the compressed pages on one thread.
//! 4. Every shard removes its pages that got empty.
//!
//! The result is the one of `SingleThreaded`: all hits of a call
//! before all of its targets, hits only on pages existing when the call
//! starts, each hit and target saturating on its own. Neither saturating
//! subtraction nor saturating (and capped) addition depends on the order, so
//! neither does the result.

use rayon::{
    ThreadPool, ThreadPoolBuildError, ThreadPoolBuilder,
    iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator},
    slice::ParallelSlice,
};
use rustc_hash::FxBuildHasher;

use crate::{
    ALL_POINTS_GROUP_EDGE_BITS, CompressedVoxelGroup, Multipliers, POINTS_IN_VOXEL_GROUP,
    Voxelizer, VoxelizerParams,
    geometry::{PointF, PointI, ZOrder},
};

const BITS_FOR_ZORDER: u32 = 2;
const BITS_FOR_ZORDER_MASK: i32 = 2_i32.pow(BITS_FOR_ZORDER) - 1;
const PAGE_SIZE: u32 = 4096;
const BITS_FOR_COMPRESSED_ZORDER: u32 = PAGE_SIZE.highest_one().unwrap() / 3;
const BITS_FOR_COMPRESSED_ZORDER_MASK: i32 = 2_i32.pow(BITS_FOR_COMPRESSED_ZORDER) - 1;
/// Bits of the index of a voxel in its fine page
const PAGE_BITS: u32 = 3 * BITS_FOR_ZORDER;
const PAGE_LEN: usize = 1 << PAGE_BITS;
const COMPRESSED_PAGE_LEN: usize = 2_usize.pow(3 * BITS_FOR_COMPRESSED_ZORDER);
const SHARD_BITS: u32 = 6;
/// Shards the fine pages are split into. More than threads, so that a
/// shard with many hits does not hold up the others for long.
const SHARDS: usize = 1 << SHARD_BITS;
/// Ray marching tasks per thread, for the same reason
const TASKS_PER_THREAD: usize = 4;
/// Points of a ray marching task at least, fewer cost more to schedule than
/// they take
const MIN_TASK_POINTS: usize = 64;

type HashMap<K, V> = hashbrown::HashMap<K, V, FxBuildHasher>;
type ThreeDim = (i32, i32, i32);
type PagedIndex = (ThreeDim, usize);

static_assertions::const_assert!(BITS_FOR_COMPRESSED_ZORDER == ALL_POINTS_GROUP_EDGE_BITS);
static_assertions::const_assert!(COMPRESSED_PAGE_LEN == POINTS_IN_VOXEL_GROUP);

#[derive(Debug)]
pub struct MultiThreaded {
    shards: Vec<Shard>,
    compressed_voxels: Option<HashMap<ThreeDim, [u64; COMPRESSED_PAGE_LEN]>>,
    /// Buffers of the ray marching tasks, kept between calls so that their
    /// memory is reused
    tasks: Vec<Task>,
    resolution: f32,
    multipliers: Multipliers,
    destroy_threshold: u32,
    construct_threshold: u32,
    max_value: u32,
    zorder: ZOrder,
    thread_pool: ThreadPool,
}

/// A fine page, 4x4x4 voxels.
#[derive(Debug, Clone)]
struct Page {
    key: ThreeDim,
    values: [u32; PAGE_LEN],
    /// Its snapshot is in [`Shard::snapshots`]
    touched: bool,
    /// It is in [`Shard::maybe_empty`]
    maybe_empty: bool,
}

/// The fine pages whose key [`shard_of`] maps to one index, and what the
/// current call did to them.
#[derive(Debug, Default)]
struct Shard {
    /// Page key to its slot in `pages`
    index: HashMap<ThreeDim, u32>,
    pages: Vec<Page>,
    /// Slots of removed pages, reused by new ones
    free: Vec<u32>,
    /// Slot and values of every page touched during the current call, taken
    /// at its first touch
    snapshots: Vec<(u32, [u32; PAGE_LEN])>,
    /// Slots of pages a voxel of which got zeroed or stayed 0 during the
    /// current call
    maybe_empty: Vec<u32>,
    /// Changed voxels of the current call, without `compress`
    points: Vec<PointI>,
    /// Threshold crossings of the current call, with `compress`
    crossings: Vec<Crossing>,
}

/// A step of a ray in an existing page.
#[derive(Debug, Clone, Copy)]
struct Hit {
    /// Slot of the page in its shard, shifted by [`PAGE_BITS`], or the index
    /// of the voxel in the page
    at: u32,
    i: u32,
}

/// The voxel a ray ends in.
#[derive(Debug, Clone, Copy)]
struct Target {
    page: ThreeDim,
    inner: u32,
    /// Intensity times the target multiplier
    add: u32,
}

#[derive(Debug, Clone, Copy)]
struct Crossing {
    page: ThreeDim,
    inner: u32,
    /// Reached `construct_threshold`, otherwise dropped to
    /// `destroy_threshold`
    construct: bool,
}

/// Hits and targets of a ray marching task, one list per shard.
#[derive(Debug)]
struct Task {
    hits: Vec<Vec<Hit>>,
    targets: Vec<Vec<Target>>,
}

impl Default for Task {
    fn default() -> Self {
        Self {
            hits: (0..SHARDS).map(|_| Vec::new()).collect(),
            targets: (0..SHARDS).map(|_| Vec::new()).collect(),
        }
    }
}

/// What ray marching needs, shared by the tasks.
struct March<'a> {
    resolution: f32,
    multipliers: Multipliers,
    sensor_pos_i: PointI,
    sensor_pos: &'a PointF,
    zorder: &'a ZOrder,
}

/// What the shards need to apply a call, shared by them.
struct Update<'a> {
    compress: bool,
    destroy_threshold: u32,
    construct_threshold: u32,
    max_value: u32,
    zorder: &'a ZOrder,
}

impl MultiThreaded {
    #[inline]
    pub fn new(params: VoxelizerParams, thread_count: usize) -> Result<Self, ThreadPoolBuildError> {
        let VoxelizerParams {
            resolution,
            multipliers,
            destroy_threshold,
            construct_threshold,
            max_value,
            compress,
        } = params;
        Ok(Self {
            shards: (0..SHARDS).map(|_| Shard::default()).collect(),
            compressed_voxels: if compress {
                Some(HashMap::default())
            } else {
                None
            },
            tasks: Vec::new(),
            resolution,
            multipliers,
            destroy_threshold,
            construct_threshold,
            max_value,
            zorder: ZOrder::default(),
            thread_pool: ThreadPoolBuilder::new().num_threads(thread_count).build()?,
        })
    }
}

impl Voxelizer for MultiThreaded {
    fn add_points(&mut self, mut sensor_pos: PointF, points: &[PointF]) -> Vec<PointI> {
        sensor_pos /= self.resolution;
        let sensor_pos_i = PointI {
            x: sensor_pos.x.floor() as i32,
            y: sensor_pos.y.floor() as i32,
            z: sensor_pos.z.floor() as i32,
            i: sensor_pos.i as u32,
        };
        let Self {
            shards,
            compressed_voxels,
            tasks,
            resolution,
            multipliers,
            destroy_threshold,
            construct_threshold,
            max_value,
            zorder,
            thread_pool,
        } = self;
        let march = March {
            resolution: *resolution,
            multipliers: *multipliers,
            sensor_pos_i,
            sensor_pos: &sensor_pos,
            zorder,
        };
        let update = Update {
            compress: compressed_voxels.is_some(),
            destroy_threshold: *destroy_threshold,
            construct_threshold: *construct_threshold,
            max_value: *max_value,
            zorder,
        };
        let task_count = (thread_pool.current_num_threads() * TASKS_PER_THREAD)
            .min(points.len().div_ceil(MIN_TASK_POINTS))
            .max(1);
        let chunk = points.len().div_ceil(task_count).max(1);
        let task_count = points.len().div_ceil(chunk);
        if tasks.len() < task_count {
            tasks.resize_with(task_count, Task::default);
        }
        let tasks = &mut tasks[..task_count];

        thread_pool.install(|| {
            tasks
                .par_iter_mut()
                .zip(points.par_chunks(chunk))
                .for_each(|(task, points)| task.march(points, &march, shards));
            let tasks = &*tasks;
            shards
                .par_iter_mut()
                .enumerate()
                .for_each(|(n, shard)| shard.apply(n, tasks, &update));
        });

        let mut result = Vec::new();
        // Compressed pages a word of which got zeroed
        let mut emptied_compressed = Vec::new();
        if let Some(compressed_voxels) = compressed_voxels.as_mut() {
            let mut compressed_dedup = HashMap::default();
            for crossing in shards
                .iter_mut()
                .flat_map(|shard| shard.crossings.drain(..))
            {
                let Crossing {
                    page,
                    inner,
                    construct,
                } = crossing;
                let [x, y, z] = zorder.coord_of(inner as u64);
                let (key, word) = compressed_index(zorder, page);
                let bit = word_bit([x, y, z]);
                // A voxel between the thresholds keeps its state, so a
                // crossing changes it only when it is in the other one
                let v = if construct {
                    let v = &mut compressed_voxels.entry(key).or_insert_with(|| [0; _])[word];
                    if *v & bit != 0 {
                        continue;
                    }
                    *v |= bit;
                    *v
                } else {
                    let Some(words) = compressed_voxels.get_mut(&key) else {
                        continue;
                    };
                    let v = &mut words[word];
                    if *v & bit == 0 {
                        continue;
                    }
                    *v &= !bit;
                    if *v == 0 {
                        emptied_compressed.push(key);
                    }
                    *v
                };
                let upper_bits = z as i32 & 2;
                debug_assert!(upper_bits & !2 == 0);
                compressed_dedup.insert(
                    (
                        page.0 << BITS_FOR_ZORDER,
                        page.1 << BITS_FOR_ZORDER,
                        (page.2 << BITS_FOR_ZORDER) | upper_bits,
                    ),
                    if upper_bits != 0 {
                        (v >> 32) as u32
                    } else {
                        v as u32
                    },
                );
            }
            result.extend(
                compressed_dedup
                    .into_iter()
                    .map(|((x, y, z), i)| PointI { x, y, z, i }),
            );
        } else {
            result.reserve(shards.iter().map(|shard| shard.points.len()).sum());
            for shard in shards.iter_mut() {
                result.append(&mut shard.points);
            }
        }

        let compressed = compressed_voxels.as_ref();
        thread_pool.install(|| {
            shards
                .par_iter_mut()
                .for_each(|shard| shard.remove_emptied(compressed, zorder));
        });
        if let Some(compressed_voxels) = compressed_voxels.as_mut() {
            for key in emptied_compressed {
                if compressed_voxels
                    .get(&key)
                    .is_some_and(|words| words.iter().all(|&w| w == 0))
                {
                    compressed_voxels.remove(&key);
                }
            }
        }
        result
    }
    #[inline]
    fn set_multipliers(&mut self, multipliers: Multipliers) {
        self.multipliers = multipliers;
    }
    fn get_all_points(&self) -> Vec<PointI> {
        self.iter_pages()
            .flat_map(|(page, values)| {
                values
                    .iter()
                    .enumerate()
                    .filter(|(_, v)| **v != 0)
                    .map(move |(i, v)| self.point_from_index((page, i), *v))
            })
            .collect()
    }
    fn get_all_points_compressed(&self) -> Vec<CompressedVoxelGroup> {
        if let Some(compressed) = self.compressed_voxels.as_ref() {
            compressed
                .into_iter()
                .map(|((x, y, z), v)| CompressedVoxelGroup {
                    point: PointI {
                        x: *x,
                        y: *y,
                        z: *z,
                        i: 0,
                    },
                    values: (*v).into(),
                })
                .collect()
        } else {
            // Without compress there is no state between the thresholds, so a
            // voxel counts as constructed once it reaches `construct_threshold`
            let mut groups: HashMap<ThreeDim, Box<[u64; POINTS_IN_VOXEL_GROUP]>> =
                HashMap::default();
            for (page, values) in self.iter_pages() {
                let word = values
                    .iter()
                    .enumerate()
                    .filter(|(_, v)| **v != 0 && **v >= self.construct_threshold)
                    .fold(0, |word, (inner, _)| {
                        word | word_bit(self.zorder.coord_of(inner as u64))
                    });
                if word != 0 {
                    let (key, inner) = compressed_index(&self.zorder, page);
                    groups.entry(key).or_insert_with(|| Box::new([0; _]))[inner] = word;
                }
            }
            groups
                .into_iter()
                .map(|((x, y, z), values)| CompressedVoxelGroup {
                    point: PointI { x, y, z, i: 0 },
                    values,
                })
                .collect()
        }
    }
    #[inline]
    fn multipliers(&self) -> Multipliers {
        self.multipliers
    }
}

impl MultiThreaded {
    /// Page key and index in the page of the voxel `point`.
    #[cfg(test)]
    fn get_index(&self, point: &PointI) -> PagedIndex {
        (page_key(point), inner_index(&self.zorder, point))
    }
    #[inline]
    fn point_from_index(&self, idx: PagedIndex, i: u32) -> PointI {
        point_from_index(&self.zorder, idx, i)
    }
    /// Every fine page with its key.
    fn iter_pages(&self) -> impl Iterator<Item = (ThreeDim, &[u32; PAGE_LEN])> {
        self.shards.iter().flat_map(|shard| {
            shard
                .index
                .iter()
                .map(|(page, &slot)| (*page, &shard.pages[slot as usize].values))
        })
    }
    #[inline]
    fn closest_boundary(upper: bool, position: f32, len: f32) -> f32 {
        let boundary = position.floor() + upper as i32 as f32;
        let result = (boundary - position) / len;
        debug_assert!(result >= 0.0);
        result
    }
    /// `max_[xyz]` of an axis after a step along it. An axis that reached the
    /// target is never stepped again: when the point lies on boundaries of
    /// several axes (or due to float errors) its next crossing can tie with
    /// the last one of another axis, and stepping it would walk past the
    /// target.
    #[inline]
    fn next_boundary(max: f32, delta: f32, reached: bool) -> f32 {
        if reached { f32::INFINITY } else { max + delta }
    }
    /// Calls `visit` with every voxel the ray from `sensor_pos` to `point`
    /// passes through, the target excluded, `i` set to the hit intensity.
    #[inline(always)]
    fn ray_march(march: &March, mut point: PointF, mut visit: impl FnMut(&PointI)) {
        // http://www.cse.yorku.ca/~amana/research/grid.pdf
        let sensor_pos_divided = march.sensor_pos;
        let mut sensor_pos = march.sensor_pos_i.clone();
        point /= march.resolution;
        // dir_[xyz] cannot be calculated after `point -= sensor_pos_divided;`
        // due to desired rounding before subtraction
        let target_point = PointI {
            x: point.x.floor() as i32,
            y: point.y.floor() as i32,
            z: point.z.floor() as i32,
            i: (point.i * march.multipliers.intensity) as u32 * march.multipliers.hit,
        };
        // A hit of 0 changes nothing
        if target_point.i == 0 {
            return;
        }
        let dir_x = (target_point.x - sensor_pos.x).signum();
        let dir_y = (target_point.y - sensor_pos.y).signum();
        let dir_z = (target_point.z - sensor_pos.z).signum();
        sensor_pos.i = target_point.i;
        point -= sensor_pos_divided;
        // Every step moves exactly one voxel towards the target, so this is
        // the exact step count. Used instead of `target_point != sensor_pos`
        // so float errors can't cause an infinite loop
        let steps = (target_point.x - sensor_pos.x).abs()
            + (target_point.y - sensor_pos.y).abs()
            + (target_point.z - sensor_pos.z).abs();
        use std::cmp::Ordering;
        match (dir_x.cmp(&0), dir_y.cmp(&0), dir_z.cmp(&0)) {
            (Ordering::Equal, Ordering::Equal, Ordering::Equal) => (),
            (_, Ordering::Equal, Ordering::Equal) => {
                while target_point.x != sensor_pos.x {
                    visit(&sensor_pos);
                    sensor_pos.x += dir_x;
                }
            }
            (Ordering::Equal, _, Ordering::Equal) => {
                while target_point.y != sensor_pos.y {
                    visit(&sensor_pos);
                    sensor_pos.y += dir_y;
                }
            }
            (Ordering::Equal, Ordering::Equal, _) => {
                while target_point.z != sensor_pos.z {
                    visit(&sensor_pos);
                    sensor_pos.z += dir_z;
                }
            }
            (x, y, Ordering::Equal) => {
                let dx = 1.0 / point.x.abs();
                let dy = 1.0 / point.y.abs();
                let mut max_x = Self::closest_boundary(x.is_gt(), sensor_pos_divided.x, point.x);
                let mut max_y = Self::closest_boundary(y.is_gt(), sensor_pos_divided.y, point.y);
                for _ in 0..steps {
                    debug_assert!(sensor_pos.x == target_point.x || max_x < 1.0 + dx);
                    debug_assert!(sensor_pos.y == target_point.y || max_y < 1.0 + dy);
                    visit(&sensor_pos);
                    if max_x < max_y {
                        sensor_pos.x += dir_x;
                        max_x = Self::next_boundary(max_x, dx, sensor_pos.x == target_point.x);
                    } else {
                        sensor_pos.y += dir_y;
                        max_y = Self::next_boundary(max_y, dy, sensor_pos.y == target_point.y);
                    }
                }
            }
            (x, Ordering::Equal, z) => {
                let dx = 1.0 / point.x.abs();
                let dz = 1.0 / point.z.abs();
                let mut max_x = Self::closest_boundary(x.is_gt(), sensor_pos_divided.x, point.x);
                let mut max_z = Self::closest_boundary(z.is_gt(), sensor_pos_divided.z, point.z);
                for _ in 0..steps {
                    debug_assert!(sensor_pos.x == target_point.x || max_x < 1.0 + dx);
                    debug_assert!(sensor_pos.z == target_point.z || max_z < 1.0 + dz);
                    visit(&sensor_pos);
                    if max_x < max_z {
                        sensor_pos.x += dir_x;
                        max_x = Self::next_boundary(max_x, dx, sensor_pos.x == target_point.x);
                    } else {
                        sensor_pos.z += dir_z;
                        max_z = Self::next_boundary(max_z, dz, sensor_pos.z == target_point.z);
                    }
                }
            }
            (Ordering::Equal, y, z) => {
                let dy = 1.0 / point.y.abs();
                let dz = 1.0 / point.z.abs();
                let mut max_y = Self::closest_boundary(y.is_gt(), sensor_pos_divided.y, point.y);
                let mut max_z = Self::closest_boundary(z.is_gt(), sensor_pos_divided.z, point.z);
                for _ in 0..steps {
                    debug_assert!(sensor_pos.y == target_point.y || max_y < 1.0 + dy);
                    debug_assert!(sensor_pos.z == target_point.z || max_z < 1.0 + dz);
                    visit(&sensor_pos);
                    if max_y < max_z {
                        sensor_pos.y += dir_y;
                        max_y = Self::next_boundary(max_y, dy, sensor_pos.y == target_point.y);
                    } else {
                        sensor_pos.z += dir_z;
                        max_z = Self::next_boundary(max_z, dz, sensor_pos.z == target_point.z);
                    }
                }
            }
            (x, y, z) => {
                let dx = 1.0 / point.x.abs();
                let dy = 1.0 / point.y.abs();
                let dz = 1.0 / point.z.abs();
                let mut max_x = Self::closest_boundary(x.is_gt(), sensor_pos_divided.x, point.x);
                let mut max_y = Self::closest_boundary(y.is_gt(), sensor_pos_divided.y, point.y);
                let mut max_z = Self::closest_boundary(z.is_gt(), sensor_pos_divided.z, point.z);
                for _ in 0..steps {
                    debug_assert!(sensor_pos.x == target_point.x || max_x < 1.0 + dx);
                    debug_assert!(sensor_pos.y == target_point.y || max_y < 1.0 + dy);
                    debug_assert!(sensor_pos.z == target_point.z || max_z < 1.0 + dz);
                    visit(&sensor_pos);
                    if max_x < max_z {
                        if max_x < max_y {
                            sensor_pos.x += dir_x;
                            max_x = Self::next_boundary(max_x, dx, sensor_pos.x == target_point.x);
                        } else {
                            sensor_pos.y += dir_y;
                            max_y = Self::next_boundary(max_y, dy, sensor_pos.y == target_point.y);
                        }
                    } else {
                        if max_y < max_z {
                            sensor_pos.y += dir_y;
                            max_y = Self::next_boundary(max_y, dy, sensor_pos.y == target_point.y);
                        } else {
                            sensor_pos.z += dir_z;
                            max_z = Self::next_boundary(max_z, dz, sensor_pos.z == target_point.z);
                        }
                    }
                }
            }
        };
        // `steps` bounds the loops, so a wrong walk would end silently elsewhere
        debug_assert_eq!(sensor_pos, target_point);
    }
}

impl Task {
    /// Marches the rays of `points`, recording their hits on existing pages
    /// and their targets. `shards` are not changed.
    fn march(&mut self, points: &[PointF], march: &March, shards: &[Shard]) {
        self.hits.iter_mut().for_each(Vec::clear);
        self.targets.iter_mut().for_each(Vec::clear);
        let zorder = march.zorder;
        // Consecutive steps mostly stay in one page: its lookup is reused.
        // No voxel lies in page (i32::MIN, ..), the voxels are i32 too.
        let mut cached: (ThreeDim, Option<(usize, u32)>) = ((i32::MIN, 0, 0), None);
        for point in points {
            MultiThreaded::ray_march(march, point.clone(), |voxel| {
                let page = page_key(voxel);
                if page != cached.0 {
                    let shard = shard_of(page);
                    cached = (
                        page,
                        shards[shard].index.get(&page).map(|&slot| (shard, slot)),
                    );
                }
                if let Some((shard, slot)) = cached.1 {
                    self.hits[shard].push(Hit {
                        at: (slot << PAGE_BITS) | inner_index(zorder, voxel) as u32,
                        i: voxel.i,
                    });
                }
            });
        }
        let resolution = march.resolution;
        let multipliers = march.multipliers;
        for v in points {
            let v = v / resolution;
            let point = PointI {
                x: v.x.floor() as i32,
                y: v.y.floor() as i32,
                z: v.z.floor() as i32,
                i: (v.i * multipliers.intensity) as u32,
            };
            let page = page_key(&point);
            self.targets[shard_of(page)].push(Target {
                page,
                inner: inner_index(zorder, &point) as u32,
                add: point.i * multipliers.target,
            });
        }
    }
}

impl Shard {
    /// Applies the hits and then the targets of `tasks` that belong to this
    /// shard (number `n`), then records what changed.
    fn apply(&mut self, n: usize, tasks: &[Task], update: &Update) {
        for hit in tasks.iter().flat_map(|task| &task.hits[n]) {
            let slot = hit.at >> PAGE_BITS;
            let inner = (hit.at & (PAGE_LEN as u32 - 1)) as usize;
            let page = &mut self.pages[slot as usize];
            if !page.touched {
                page.touched = true;
                self.snapshots.push((slot, page.values));
            }
            let prev = page.values[inner];
            let voxel = prev.saturating_sub(hit.i);
            page.values[inner] = voxel;
            if prev != 0 && voxel == 0 && !page.maybe_empty {
                page.maybe_empty = true;
                self.maybe_empty.push(slot);
            }
        }
        for target in tasks.iter().flat_map(|task| &task.targets[n]) {
            let slot = self.slot(target.page);
            let page = &mut self.pages[slot as usize];
            if !page.touched {
                page.touched = true;
                self.snapshots.push((slot, page.values));
            }
            let voxel = &mut page.values[target.inner as usize];
            *voxel = voxel.saturating_add(target.add).min(update.max_value);
            // A target adding nothing (or capped at 0) can leave a new page
            // all 0
            if *voxel == 0 && !page.maybe_empty {
                page.maybe_empty = true;
                self.maybe_empty.push(slot);
            }
        }
        for (slot, prev) in self.snapshots.drain(..) {
            let page = &mut self.pages[slot as usize];
            page.touched = false;
            if prev == page.values {
                continue;
            }
            for (inner, (&prev, &voxel)) in prev.iter().zip(&page.values).enumerate() {
                if prev == voxel {
                    continue;
                }
                if !update.compress {
                    self.points
                        .push(point_from_index(update.zorder, (page.key, inner), voxel));
                } else if prev < update.construct_threshold && voxel >= update.construct_threshold {
                    self.crossings.push(Crossing {
                        page: page.key,
                        inner: inner as u32,
                        construct: true,
                    });
                } else if prev > update.destroy_threshold && voxel <= update.destroy_threshold {
                    self.crossings.push(Crossing {
                        page: page.key,
                        inner: inner as u32,
                        construct: false,
                    });
                }
            }
        }
    }
    /// Slot of the page `key`, created (all 0) when missing.
    fn slot(&mut self, key: ThreeDim) -> u32 {
        if let Some(&slot) = self.index.get(&key) {
            return slot;
        }
        let page = Page {
            key,
            values: [0; PAGE_LEN],
            touched: false,
            maybe_empty: false,
        };
        let slot = match self.free.pop() {
            Some(slot) => {
                self.pages[slot as usize] = page;
                slot
            }
            None => {
                // `Hit::at` holds the slot above the index of the voxel
                assert!(self.pages.len() < 1 << (u32::BITS - PAGE_BITS));
                self.pages.push(page);
                (self.pages.len() - 1) as u32
            }
        };
        self.index.insert(key, slot);
        slot
    }
    /// Removes pages zeroed during the call that are empty (all voxels 0 and
    /// none constructed).
    fn remove_emptied(
        &mut self,
        compressed_voxels: Option<&HashMap<ThreeDim, [u64; COMPRESSED_PAGE_LEN]>>,
        zorder: &ZOrder,
    ) {
        for slot in self.maybe_empty.drain(..) {
            let page = &mut self.pages[slot as usize];
            page.maybe_empty = false;
            let (key, inner) = compressed_index(zorder, page.key);
            let constructed = compressed_voxels
                .and_then(|compressed_voxels| compressed_voxels.get(&key))
                .is_some_and(|words| words[inner] != 0);
            if !constructed && page.values.iter().all(|&v| v == 0) {
                self.index.remove(&page.key);
                self.free.push(slot);
            }
        }
    }
}

/// Shard of the fine page `page`. Not the hash the maps use, so that the
/// keys of a shard still spread over its map.
#[inline]
fn shard_of((x, y, z): ThreeDim) -> usize {
    let hash = (x as u32).wrapping_mul(0x9E37_79B1)
        ^ (y as u32).wrapping_mul(0x85EB_CA77)
        ^ (z as u32).wrapping_mul(0xC2B2_AE3D);
    (hash >> (u32::BITS - SHARD_BITS)) as usize
}

#[inline]
fn page_key(point: &PointI) -> ThreeDim {
    (
        point.x >> BITS_FOR_ZORDER,
        point.y >> BITS_FOR_ZORDER,
        point.z >> BITS_FOR_ZORDER,
    )
}

#[inline]
fn inner_index(zorder: &ZOrder, point: &PointI) -> usize {
    zorder.index_of([
        (point.x & BITS_FOR_ZORDER_MASK) as u16,
        (point.y & BITS_FOR_ZORDER_MASK) as u16,
        (point.z & BITS_FOR_ZORDER_MASK) as u16,
    ]) as usize
}

#[inline]
fn point_from_index(zorder: &ZOrder, idx: PagedIndex, i: u32) -> PointI {
    let [x, y, z] = zorder.coord_of(idx.1 as u64);
    PointI {
        x: (idx.0.0 << BITS_FOR_ZORDER) | x as i32,
        y: (idx.0.1 << BITS_FOR_ZORDER) | y as i32,
        z: (idx.0.2 << BITS_FOR_ZORDER) | z as i32,
        i,
    }
}

/// Key of the compressed page holding the bits of `page` and index of their
/// word in it.
#[inline]
fn compressed_index(zorder: &ZOrder, page: ThreeDim) -> PagedIndex {
    (
        (
            page.0 >> BITS_FOR_COMPRESSED_ZORDER,
            page.1 >> BITS_FOR_COMPRESSED_ZORDER,
            page.2 >> BITS_FOR_COMPRESSED_ZORDER,
        ),
        zorder.index_of([
            (page.0 & BITS_FOR_COMPRESSED_ZORDER_MASK) as u16,
            (page.1 & BITS_FOR_COMPRESSED_ZORDER_MASK) as u16,
            (page.2 & BITS_FOR_COMPRESSED_ZORDER_MASK) as u16,
        ]) as usize,
    )
}

/// Bit of the voxel at `[x, y, z]` of a fine page in its compressed word.
#[inline]
fn word_bit([x, y, z]: [u16; 3]) -> u64 {
    1 << x << (y << 2) << (z << 4)
}

#[cfg(test)]
mod test {
    use super::{
        BITS_FOR_COMPRESSED_ZORDER, BITS_FOR_COMPRESSED_ZORDER_MASK, BITS_FOR_ZORDER,
        MultiThreaded, PAGE_SIZE, shard_of,
    };
    use crate::{
        Multipliers, Voxelizer, VoxelizerParams,
        geometry::PointI,
        test::{
            Coord, PAGE_LEN, PLAIN, PagedIndex, SCENARIO_PARAMS, SWITCHED, TestVoxelizer,
            assert_snapshot_eq, compressed_bits, compressed_word, cube_faces, get_voxel,
            nonzero_voxels, page_bit, pf, pi, scenario_with, seed, sorted, with_i,
        },
    };

    /// Thread count used by the shared suite.
    const THREADS: usize = 4;

    /// Without a cap on voxel values or scaling of intensities.
    fn new(
        resolution: f32,
        hit_multiplier: u32,
        target_multiplier: u32,
        destroy_threshold: u32,
        construct_threshold: u32,
        compress: bool,
        thread_count: usize,
    ) -> MultiThreaded {
        MultiThreaded::new(
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
            thread_count,
        )
        .expect("thread pool must build")
    }

    impl TestVoxelizer for MultiThreaded {
        const BITS_FOR_ZORDER: u32 = BITS_FOR_ZORDER;
        const BITS_FOR_COMPRESSED_ZORDER: u32 = BITS_FOR_COMPRESSED_ZORDER;
        const BITS_FOR_COMPRESSED_ZORDER_MASK: i32 = BITS_FOR_COMPRESSED_ZORDER_MASK;
        const PAGE_SIZE: u32 = PAGE_SIZE;
        fn with_params(params: VoxelizerParams) -> Self {
            MultiThreaded::new(params, THREADS).expect("thread pool must build")
        }
        fn get_index(&self, point: &PointI) -> PagedIndex {
            MultiThreaded::get_index(self, point)
        }
        fn point_from_index(&self, idx: PagedIndex, i: u32) -> PointI {
            MultiThreaded::point_from_index(self, idx, i)
        }
        fn closest_boundary(upper: bool, position: f32, len: f32) -> f32 {
            MultiThreaded::closest_boundary(upper, position, len)
        }
        fn set_voxel(&mut self, x: i32, y: i32, z: i32, value: u32) {
            let idx = MultiThreaded::get_index(self, &pi(x, y, z, 0));
            let shard = &mut self.shards[shard_of(idx.0)];
            let slot = shard.slot(idx.0);
            shard.pages[slot as usize].values[idx.1] = value;
        }
        fn pages(&self) -> Vec<(Coord, [u32; PAGE_LEN])> {
            let mut out: Vec<_> = self.iter_pages().map(|(k, v)| (k, *v)).collect();
            out.sort_by_key(|(k, _)| *k);
            out
        }
        fn compressed_pages(&self) -> Option<Vec<(Coord, Vec<u64>)>> {
            let mut out: Vec<_> = self
                .compressed_voxels
                .as_ref()?
                .iter()
                .map(|(k, v)| (*k, v.to_vec()))
                .collect();
            out.sort_by_key(|(k, _)| *k);
            Some(out)
        }
        fn pending_results(&self) -> usize {
            self.shards
                .iter()
                .map(|shard| {
                    shard.snapshots.len()
                        + shard.maybe_empty.len()
                        + shard.points.len()
                        + shard.crossings.len()
                })
                .sum()
        }
    }

    crate::test::shared_tests!(MultiThreaded);

    // ------------------------------------------------------------ constructor

    #[test]
    fn new_stores_parameters() {
        let params = VoxelizerParams {
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
        };
        let v = MultiThreaded::new(params, 3).expect("thread pool must build");
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
        assert_eq!(v.thread_pool.current_num_threads(), 3);
        assert!(v.pages().is_empty());
        assert_eq!(v.pending_results(), 0);
        let v = new(1.0, 1, 1, 0, 1, false, 1);
        assert_eq!(v.thread_pool.current_num_threads(), 1);
    }

    #[test]
    fn new_thread_count_zero_uses_default() {
        let v = MultiThreaded::new(PLAIN, 0).expect("0 means default");
        assert!(v.thread_pool.current_num_threads() >= 1);
    }

    #[test]
    fn new_compress_flag() {
        assert!(new(1.0, 1, 1, 0, 1, false, 1).compressed_voxels.is_none());
        let v = new(1.0, 1, 1, 0, 1, true, 1);
        assert!(v.compressed_voxels.as_ref().is_some_and(|m| m.is_empty()));
    }

    // ------------------------------------------------------------ concurrency

    #[test]
    fn same_result_for_any_thread_count() {
        let make =
            |params, threads| MultiThreaded::new(params, threads).expect("thread pool must build");
        for switch in [None, Some(SWITCHED)] {
            for params in SCENARIO_PARAMS {
                let expected = scenario_with(make(params.clone(), 1), switch);
                assert!(expected.results.iter().all(|r| !r.is_empty()));
                for threads in [2, 4, 8, 16] {
                    let got = scenario_with(make(params.clone(), threads), switch);
                    assert_snapshot_eq(
                        &got,
                        &expected,
                        &format!("{params:?}, switch = {switch:?}, threads = {threads}"),
                    );
                }
            }
        }
    }

    #[test]
    fn concurrent_same_target() {
        // Every thread adds to the same target and subtracts from the same
        // three voxels on the way: no update may get lost and the result
        // must hold the final values once.
        const N: u32 = 10_000;
        let mut v = new(1.0, 1, 1, 0, 1, false, 8);
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
        // Targets 8 apart on the faces of a cube: each in its own fine page,
        // none on the path of another ray from the origin.
        let mut v = new(1.0, 1, 1, 0, 1, false, 8);
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
        // Only the targets' pages exist: nothing else got written.
        assert_eq!(v.pages().len(), targets.len());
        assert_eq!(v.pending_results(), 0);
    }

    #[test]
    fn concurrent_compress_fills_block() {
        // From high above every ray enters the z = 1 layer already inside its
        // target column, so no target lies on another ray's path. 16 voxels
        // per block, each sent twice, must merge into one entry per block.
        let mut v = new(1.0, 1, 1, 0, 1, true, 8);
        let blocks = [(0, 0), (4, 4)];
        let mut points = Vec::new();
        let mut voxels = Vec::new();
        for (bx, by) in blocks {
            for x in 0..4 {
                for y in 0..4 {
                    let p = pf((bx + x) as f32 + 0.5, (by + y) as f32 + 0.5, 1.5, 1.0);
                    points.push(p.clone());
                    points.push(p);
                    voxels.push((bx + x, by + y, 1));
                }
            }
        }
        let result = v.add_points(pf(1.5, 1.5, 20.5, 1.0), &points);
        // Whole z = 1 layer of a block: bits 16..32 of the low word half.
        const LAYER: u32 = 0xFFFF_0000;
        assert_eq!(sorted(result), vec![pi(0, 0, 0, LAYER), pi(4, 4, 0, LAYER)]);
        assert_eq!(compressed_word(&v, 0, 0, 1), Some(LAYER as u64));
        assert_eq!(compressed_word(&v, 4, 4, 1), Some(LAYER as u64));
        assert_eq!(compressed_bits(&v), sorted(with_i(&voxels, 1)));
        assert_eq!(nonzero_voxels(&v), sorted(with_i(&voxels, 2)));
        assert_eq!(v.pages().len(), 2);
    }

    #[test]
    fn compress_hits_before_targets() {
        // Ray 1 constructs (1,0,0), ray 2 passes through it into (2,0,0). As
        // in the single threaded voxelizer the hits of a call come before
        // its targets and land only on pages existing when it starts: the
        // page is created by the targets, so the hit on (1,0,0) is dropped
        // whichever thread marches either ray.
        let mut v = new(1.0, 1, 1, 0, 1, true, 4);
        let result = v.add_points(
            pf(0.5, 0.5, 0.5, 1.0),
            &[pf(1.5, 0.5, 0.5, 1.0), pf(2.5, 0.5, 0.5, 1.0)],
        );
        assert_eq!(get_voxel(&v, 1, 0, 0), Some(1));
        assert_eq!(get_voxel(&v, 2, 0, 0), Some(1));
        let word = page_bit(1, 0, 0) | page_bit(2, 0, 0);
        assert_eq!(compressed_word(&v, 0, 0, 0), Some(word));
        assert_eq!(
            compressed_bits(&v),
            sorted(vec![pi(1, 0, 0, 1), pi(2, 0, 0, 1)])
        );
        assert_eq!(result, vec![pi(0, 0, 0, word as u32)]);
    }
}
