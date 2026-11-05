//! Downsampling shared by the implementations. A downsampled voxel stands for
//! a cube of `size^3` voxels, `D = voxel >> log2(size)`, and is present while
//! at least `threshold` voxels of the cube are constructed.
//!
//! The implementations report every change of the occupancy word of a fine
//! page (4x4x4 voxels, bit `x + 4y + 16z`) to [`Downsampling::word_changed`],
//! which keeps the count of constructed voxels of every downsampled voxel and
//! the occupancy of the downsampled pages up to date. Pages whose occupancy
//! changed are remembered with the word last reported, so that
//! [`Downsampling::take_changes`] reports only what differs from it.
//!
//! The state is built lazily: after the params are set or the state is
//! dropped, the next [`Downsampling::take_changes`] builds it from every
//! occupancy word and reports everything present. Until then
//! `word_changed` does nothing.

use hashbrown::hash_map::Entry;
use rustc_hash::FxBuildHasher;

use crate::{
    ALL_POINTS_GROUP_EDGE_BITS, CompressedVoxelGroup, DownsamplingError, POINTS_IN_VOXEL_GROUP,
    geometry::{PointI, ZOrder},
};

type HashMap<K, V> = hashbrown::HashMap<K, V, FxBuildHasher>;
type ThreeDim = (i32, i32, i32);

/// Bits of a coordinate within a fine page
const PAGE_BITS: u32 = 2;
const PAGE_MASK: i32 = (1 << PAGE_BITS) - 1;
const GROUP_MASK: i32 = (1 << ALL_POINTS_GROUP_EDGE_BITS) - 1;
/// Largest downsampled voxel edge
const MAX_SIZE: u8 = 128;
/// Bits of the voxels of the 2x2x2 sub-cube `sx + 2sy + 4sz` of a fine page
/// word, for downsampled voxels of edge 2.
const SUB_CUBES: [u64; 8] = {
    let mut masks = [0; 8];
    let mut sub = 0;
    while sub < 8 {
        let (sx, sy, sz) = (sub & 1, (sub >> 1) & 1, sub >> 2);
        let mut voxel = 0;
        while voxel < 8 {
            let (x, y, z) = (voxel & 1, (voxel >> 1) & 1, voxel >> 2);
            masks[sub] |= 1 << ((2 * sx + x) + 4 * (2 * sy + y) + 16 * (2 * sz + z));
            voxel += 1;
        }
        sub += 1;
    }
    masks
};

#[derive(Debug, Clone)]
pub(crate) struct Downsampling {
    /// log2 of the edge of a downsampled voxel
    shift: u32,
    threshold: u32,
    /// `None` until the next [`Downsampling::take_changes`] builds it
    state: Option<State>,
}

#[derive(Debug, Clone, Default)]
struct State {
    /// Constructed voxels of every downsampled voxel holding any
    counts: HashMap<ThreeDim, u32>,
    /// Present downsampled voxels of every downsampled page holding any,
    /// bit `x + 4y + 16z` like a fine page
    occupancy: HashMap<ThreeDim, u64>,
    /// Downsampled pages changed since the last report, with the occupancy
    /// they had then
    dirty: HashMap<ThreeDim, u64>,
}

impl Downsampling {
    /// Validates the params, see [`crate::Voxelizer::set_downsampling_params`].
    pub(crate) fn new(size: u8, threshold: u32, compress: bool) -> Result<Self, DownsamplingError> {
        if !compress {
            return Err(DownsamplingError::NotCompressed);
        }
        if !(2..=MAX_SIZE).contains(&size) || !size.is_power_of_two() {
            return Err(DownsamplingError::InvalidSize(size));
        }
        let shift = size.trailing_zeros();
        if threshold == 0 || threshold > 1 << (3 * shift) {
            return Err(DownsamplingError::InvalidThreshold(threshold));
        }
        Ok(Self {
            shift,
            threshold,
            state: None,
        })
    }

    /// Drops the state, the next [`Downsampling::take_changes`] builds it again.
    pub(crate) fn reset(&mut self) {
        self.state = None;
    }

    /// The occupancy word of the fine page `page` changed from `old` to `new`.
    #[inline]
    pub(crate) fn word_changed(&mut self, page: ThreeDim, old: u64, new: u64) {
        let Some(state) = self.state.as_mut() else {
            return;
        };
        let threshold = self.threshold;
        for_each_delta(self.shift, page, old, new, |voxel, delta| {
            state.add(voxel, delta, threshold);
        });
    }

    /// Blocks of 4x4x2 downsampled voxels that changed since the last call,
    /// in the format of compress mode `add_points`. Builds the state from
    /// `pages` (every fine page with its occupancy word) when there is none,
    /// then reports every present block.
    pub(crate) fn take_changes<I>(&mut self, pages: impl FnOnce() -> I) -> Vec<PointI>
    where
        I: Iterator<Item = (ThreeDim, u64)>,
    {
        match self.state.as_mut() {
            Some(state) => {
                let mut result = Vec::with_capacity(state.dirty.len());
                for (page, old) in state.dirty.drain() {
                    let new = state.occupancy.get(&page).copied().unwrap_or(0);
                    result.extend(blocks(page, old, new));
                }
                result
            }
            None => {
                let state = State::build(self.shift, self.threshold, pages());
                let result = state
                    .occupancy
                    .iter()
                    .flat_map(|(&page, &word)| blocks(page, 0, word))
                    .collect();
                self.state = Some(state);
                result
            }
        }
    }

    /// Every present downsampled voxel, grouped like
    /// [`crate::Voxelizer::get_all_points_compressed`]. Without a state it is
    /// computed from `pages` and nothing is kept.
    pub(crate) fn all<I>(
        &self,
        zorder: &ZOrder,
        pages: impl FnOnce() -> I,
    ) -> Vec<CompressedVoxelGroup>
    where
        I: Iterator<Item = (ThreeDim, u64)>,
    {
        match self.state.as_ref() {
            Some(state) => groups(zorder, &state.occupancy),
            None => groups(
                zorder,
                &State::build(self.shift, self.threshold, pages()).occupancy,
            ),
        }
    }
}

impl State {
    fn build(shift: u32, threshold: u32, pages: impl Iterator<Item = (ThreeDim, u64)>) -> Self {
        let mut state = Self::default();
        for (page, word) in pages {
            for_each_delta(shift, page, 0, word, |voxel, delta| {
                // Only additions, from 0
                *state.counts.entry(voxel).or_insert(0) += delta as u32;
            });
        }
        for (&voxel, &count) in &state.counts {
            if count >= threshold {
                let (page, bit) = page_and_bit(voxel);
                *state.occupancy.entry(page).or_insert(0) |= bit;
            }
        }
        state
    }

    /// Adds `delta` to the count of the downsampled voxel `voxel`, flips it
    /// when it crosses `threshold`.
    #[inline]
    fn add(&mut self, voxel: ThreeDim, delta: i32, threshold: u32) {
        let (before, after) = match self.counts.entry(voxel) {
            Entry::Occupied(mut entry) => {
                let before = *entry.get();
                debug_assert!(before.checked_add_signed(delta).is_some());
                let after = before.saturating_add_signed(delta);
                if after == 0 {
                    entry.remove();
                } else {
                    *entry.get_mut() = after;
                }
                (before, after)
            }
            Entry::Vacant(entry) => {
                debug_assert!(delta > 0);
                let after = delta.max(0) as u32;
                if after != 0 {
                    entry.insert(after);
                }
                (0, after)
            }
        };
        if (before >= threshold) == (after >= threshold) {
            return;
        }
        let (page, bit) = page_and_bit(voxel);
        let word = self.occupancy.entry(page).or_insert(0);
        self.dirty.entry(page).or_insert(*word);
        *word ^= bit;
        if *word == 0 {
            self.occupancy.remove(&page);
        }
    }
}

/// Calls `f` with every downsampled voxel the change of the occupancy word
/// of `page` from `old` to `new` changes the count of, and the change.
#[inline]
fn for_each_delta(
    shift: u32,
    page: ThreeDim,
    old: u64,
    new: u64,
    mut f: impl FnMut(ThreeDim, i32),
) {
    if shift >= PAGE_BITS {
        // The whole page lies in one downsampled voxel
        let delta = new.count_ones() as i32 - old.count_ones() as i32;
        if delta != 0 {
            let s = shift - PAGE_BITS;
            f((page.0 >> s, page.1 >> s, page.2 >> s), delta);
        }
    } else {
        debug_assert_eq!(shift, 1);
        let changed = old ^ new;
        for (sub, mask) in SUB_CUBES.into_iter().enumerate() {
            if changed & mask == 0 {
                continue;
            }
            let delta = (new & mask).count_ones() as i32 - (old & mask).count_ones() as i32;
            if delta != 0 {
                let sub = sub as i32;
                f(
                    (
                        (page.0 << 1) | (sub & 1),
                        (page.1 << 1) | ((sub >> 1) & 1),
                        (page.2 << 1) | (sub >> 2),
                    ),
                    delta,
                );
            }
        }
    }
}

/// Downsampled page of the downsampled voxel `voxel` and its bit there.
#[inline]
fn page_and_bit(voxel: ThreeDim) -> (ThreeDim, u64) {
    (
        (
            voxel.0 >> PAGE_BITS,
            voxel.1 >> PAGE_BITS,
            voxel.2 >> PAGE_BITS,
        ),
        1 << ((voxel.0 & PAGE_MASK) + 4 * (voxel.1 & PAGE_MASK) + 16 * (voxel.2 & PAGE_MASK)),
    )
}

/// Blocks (4x4x2, halves of `page`) whose bits differ between `old` and `new`.
fn blocks(page: ThreeDim, old: u64, new: u64) -> impl Iterator<Item = PointI> {
    [0, 1].into_iter().filter_map(move |half| {
        let (old, new) = ((old >> (32 * half)) as u32, (new >> (32 * half)) as u32);
        (old != new).then_some(PointI {
            x: page.0 << PAGE_BITS,
            y: page.1 << PAGE_BITS,
            z: (page.2 << PAGE_BITS) | (2 * half),
            i: new,
        })
    })
}

/// Occupancy words of `pages` grouped by 16x16x16 pages.
fn groups(zorder: &ZOrder, pages: &HashMap<ThreeDim, u64>) -> Vec<CompressedVoxelGroup> {
    let mut groups: HashMap<ThreeDim, Box<[u64; POINTS_IN_VOXEL_GROUP]>> = HashMap::default();
    for (&(x, y, z), &word) in pages {
        let key = (
            x >> ALL_POINTS_GROUP_EDGE_BITS,
            y >> ALL_POINTS_GROUP_EDGE_BITS,
            z >> ALL_POINTS_GROUP_EDGE_BITS,
        );
        let inner = zorder.index_of([
            (x & GROUP_MASK) as u16,
            (y & GROUP_MASK) as u16,
            (z & GROUP_MASK) as u16,
        ]) as usize;
        groups.entry(key).or_insert_with(|| Box::new([0; _]))[inner] = word;
    }
    groups
        .into_iter()
        .map(|((x, y, z), values)| CompressedVoxelGroup {
            point: PointI { x, y, z, i: 0 },
            values,
        })
        .collect()
}

/// Every fine page of a map of compressed groups (like the
/// [`crate::Voxelizer::get_all_points_compressed`] ones) with its
/// occupancy word, empty ones skipped.
#[cfg(any(feature = "single_threaded", feature = "multi_threaded"))]
pub(crate) fn compressed_pages<'a, S>(
    zorder: &'a ZOrder,
    map: &'a hashbrown::HashMap<ThreeDim, [u64; POINTS_IN_VOXEL_GROUP], S>,
) -> impl Iterator<Item = (ThreeDim, u64)> + 'a {
    map.iter().flat_map(move |(&(x, y, z), words)| {
        words
            .iter()
            .enumerate()
            .filter(|(_, word)| **word != 0)
            .map(move |(inner, &word)| {
                let [px, py, pz] = zorder.coord_of(inner as u64);
                (
                    (
                        (x << ALL_POINTS_GROUP_EDGE_BITS) | px as i32,
                        (y << ALL_POINTS_GROUP_EDGE_BITS) | py as i32,
                        (z << ALL_POINTS_GROUP_EDGE_BITS) | pz as i32,
                    ),
                    word,
                )
            })
    })
}

#[cfg(test)]
mod test {
    use super::{Downsampling, SUB_CUBES, for_each_delta, page_and_bit};
    use crate::{DownsamplingError, geometry::PointI};

    fn deltas(
        shift: u32,
        page: (i32, i32, i32),
        old: u64,
        new: u64,
    ) -> Vec<((i32, i32, i32), i32)> {
        let mut out = Vec::new();
        for_each_delta(shift, page, old, new, |voxel, delta| {
            out.push((voxel, delta))
        });
        out.sort();
        out
    }

    /// Pages for a state that must be built already.
    fn built() -> std::iter::Empty<((i32, i32, i32), u64)> {
        unreachable!("the state is built")
    }

    fn bit(x: i32, y: i32, z: i32) -> u64 {
        1 << (x + 4 * y + 16 * z)
    }

    #[test]
    fn sub_cubes_partition_the_page() {
        assert_eq!(SUB_CUBES.iter().fold(0, |all, mask| all | mask), u64::MAX);
        for (n, a) in SUB_CUBES.iter().enumerate() {
            assert_eq!(a.count_ones(), 8);
            for b in &SUB_CUBES[n + 1..] {
                assert_eq!(a & b, 0);
            }
        }
        assert_eq!(
            SUB_CUBES[0],
            bit(0, 0, 0)
                | bit(1, 0, 0)
                | bit(0, 1, 0)
                | bit(1, 1, 0)
                | bit(0, 0, 1)
                | bit(1, 0, 1)
                | bit(0, 1, 1)
                | bit(1, 1, 1)
        );
        assert_ne!(SUB_CUBES[7] & bit(3, 3, 3), 0);
        assert_ne!(SUB_CUBES[1] & bit(2, 0, 0), 0);
        assert_ne!(SUB_CUBES[2] & bit(0, 2, 0), 0);
        assert_ne!(SUB_CUBES[4] & bit(0, 0, 2), 0);
    }

    #[test]
    fn deltas_of_whole_pages() {
        // Edge 4: one downsampled voxel per page
        assert_eq!(deltas(2, (1, -1, 5), 0b1, 0b1110), vec![((1, -1, 5), 2)]);
        // Edge 16: 4x4x4 pages, floor for negative keys
        assert_eq!(
            deltas(4, (-1, -4, -5), 0b111, 0b1),
            vec![((-1, -1, -2), -2)]
        );
        // Same count, nothing to report
        assert_eq!(deltas(3, (0, 0, 0), 0b01, 0b10), vec![]);
    }

    #[test]
    fn deltas_of_sub_cubes() {
        let new = bit(0, 0, 0) | bit(1, 1, 1) | bit(3, 0, 0) | bit(2, 3, 3);
        assert_eq!(
            deltas(1, (-1, 0, 2), bit(3, 0, 0) | bit(3, 3, 0), new),
            vec![((-2, 0, 4), 2), ((-1, 1, 4), -1), ((-1, 1, 5), 1),]
        );
    }

    #[test]
    fn page_and_bit_of_negative_voxels() {
        assert_eq!(page_and_bit((-1, 0, 4)), ((-1, 0, 1), bit(3, 0, 0)));
        assert_eq!(page_and_bit((5, -6, -1)), ((1, -2, -1), bit(1, 2, 3)));
    }

    #[test]
    fn new_validates() {
        assert_eq!(
            Downsampling::new(4, 1, false).unwrap_err(),
            DownsamplingError::NotCompressed
        );
        for size in [0, 1, 3, 6, 255] {
            assert_eq!(
                Downsampling::new(size, 1, true).unwrap_err(),
                DownsamplingError::InvalidSize(size)
            );
        }
        for size in [2u8, 4, 8, 16, 32, 64, 128] {
            let full = u32::from(size).pow(3);
            assert!(Downsampling::new(size, 1, true).is_ok());
            assert!(Downsampling::new(size, full, true).is_ok());
            for threshold in [0, full + 1] {
                assert_eq!(
                    Downsampling::new(size, threshold, true).unwrap_err(),
                    DownsamplingError::InvalidThreshold(threshold)
                );
            }
        }
    }

    #[test]
    fn crossings_reported_once() {
        let mut d = Downsampling::new(4, 2, true).unwrap();
        // Pending: changes are ignored until the state is built
        d.word_changed((0, 0, 0), 0, 0b111);
        assert_eq!(d.take_changes(|| [((0, 0, 0), 0b1)].into_iter()), vec![]);
        d.word_changed((0, 0, 0), 0b1, 0b11);
        d.word_changed((5, 0, 2), 0, 0b11);
        let mut changes = d.take_changes(built);
        changes.sort();
        assert_eq!(
            changes,
            vec![
                PointI {
                    x: 0,
                    y: 0,
                    z: 0,
                    i: 1
                },
                PointI {
                    x: 4,
                    y: 0,
                    z: 2,
                    i: 1 << 1
                },
            ]
        );
        assert_eq!(d.take_changes(built), vec![]);
        // Back and forth between two reports
        d.word_changed((0, 0, 0), 0b11, 0b1);
        d.word_changed((0, 0, 0), 0b1, 0b11);
        assert_eq!(d.take_changes(built), vec![]);
        // Below the threshold again
        d.word_changed((5, 0, 2), 0b11, 0b10);
        assert_eq!(
            d.take_changes(built),
            vec![PointI {
                x: 4,
                y: 0,
                z: 2,
                i: 0
            }]
        );
    }
}
