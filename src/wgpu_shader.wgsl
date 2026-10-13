// Number of hashmap buckets, set by the host when the pipelines are created
override BUCKET_COUNT: u32;

@group(0) @binding(0)
var<uniform> params: Params;
// Raw bits, so it can carry both points (f32) and voxels to set (i32 position, u32 value)
@group(0) @binding(1)
var<storage, read> input_points: array<vec4u>;
// Entries touched during the current call
@group(0) @binding(2)
var<storage, read_write> altered: EntryList;
@group(0) @binding(3)
var<storage, read_write> output: Output;
// Entries a voxel of which got zeroed during the current call, removed by `remove_empty` when empty
@group(0) @binding(4)
var<storage, read_write> emptied: EntryList;
// Sized by BUCKET_COUNT * BUCKET_SIZE, can't be one struct since only the last member can be runtime sized
@group(1) @binding(0)
var<storage, read_write> entries: array<Entry>;
@group(1) @binding(1)
var<storage, read_write> voxels: array<Voxel>;

struct Params {
    camera_x: f32,
    camera_y: f32,
    camera_z: f32,
    resolution: f32,
    hit_multiplier: u32,
    target_multiplier: u32,
    destroy_threshold: u32,
    construct_threshold: u32,
    // Number of input points or, for `collect`, of altered entries to process
    count: u32,
    // Index of the first altered entry processed by `collect`
    first: u32,
    compress: u32,
    // Voxel values saturate at it
    max_value: u32,
}

// Every entry is listed at most once
struct EntryList {
    len: atomic<u32>,
    entries: array<u32>,
}

struct Output {
    len: atomic<u32>,
    // Set when a voxel is destroyed without having been constructed
    error: atomic<u32>,
    data: array<vec4i>,
}

struct Voxel {
    value: atomic<u32>,
    // Value reported by the end of the previous call
    prev: u32,
}

const WORKGROUP_SIZE: u32 = 256;

// Dispatches can be 2D when there are more workgroups than allowed in one dimension
@must_use
fn thread_index(global_id: vec3u, num_workgroups: vec3u) -> u32 {
    return global_id.x + global_id.y * num_workgroups.x * WORKGROUP_SIZE;
}

// Z-Order
@must_use
fn expand_bits(bits: u32) -> u32 {
    var v = bits;
    v = (v | (v << u32(16))) & 0x030000FFu;
    v = (v | (v << u32(8))) & 0x0300F00Fu;
    v = (v | (v << u32(4))) & 0x030C30C3u;
    v = (v | (v << u32(2))) & 0x09249249u;
    return v;
}

@must_use
fn get_z_order(pos: vec3i) -> u32 {
    let local = bitcast<vec3u>(pos) & vec3u(BITS_FOR_ENTRY_VOXEL_MASK);
    return (expand_bits(local.z) << u32(2)) //
    | (expand_bits(local.y) << u32(1)) //
    | (expand_bits(local.x));
}
// End of Z-Order

// Hashmap implementation

// https://niessnerlab.org/papers/2013/4hashing/niessner2013hashing.pdf
// https://arxiv.org/pdf/1805.03709

// WGSL atomics are relaxed and there is no device scope fence, so nothing orders accesses
// to different memory locations made by different workgroups (a lock can't protect plain data)
// The hashmap relies only on:
// - atomicity of a read-modify-write of a single location
// - visibility of everything written by a dispatch to the following ones
// During a dispatch every key word is written at most once, by the only compare exchange that finds it unset,
// so once set it is final and all threads inserting the same key make the same decisions
// Open addressing with linear probing, entries are never moved and are removed only by
// `remove_empty` and `reclaim_removed`, which insert nothing,
// so during any other dispatch an entry once found can be used without any synchronization
// A removed entry is left as a tombstone, which matches no key, so lookups probe past it,
// and is freed once no stored key can be probed past it (see `reclaim_removed`)
// No thread ever waits for another one

const BITS_FOR_ENTRY_VOXEL_COUNT: u32 = 2;
const BITS_FOR_ENTRY_VOXEL_MASK: u32 = (1u << BITS_FOR_ENTRY_VOXEL_COUNT) - 1;
const ENTRY_VOXEL_COUNT: u32 = 1u << (3 * BITS_FOR_ENTRY_VOXEL_COUNT);
// Entries of a bucket are the first ones probed for its keys
const BUCKET_SIZE: u32 = 2;
// How many entries are probed for a key
const MAX_PROBE: u32 = 128;

// Returned instead of an entry index
const NO_ENTRY: u32 = 0xFFFFFFFFu;

@must_use
fn entry_count() -> u32 {
    return BUCKET_COUNT * BUCKET_SIZE;
}

alias StoredValue = u32;
@must_use
fn voxel_pos(entry: u32, zorder: u32) -> u32 {
    return (entry << (BITS_FOR_ENTRY_VOXEL_COUNT * 3)) | zorder;
}

// Key word of a free entry
const KEY_UNSET: u32 = 0;
// Set in every written key word, hashable positions take only 30 bits
const KEY_SET: u32 = 1u << 31;
// First key word of a removed entry (tombstone), KEY_SET is clear so it matches no key
const KEY_REMOVED: u32 = 1;
// Entry has been modified during the current call (it is listed in `altered`)
const ENTRY_TOUCHED: u32 = 1;
// A voxel of the entry has been zeroed during the current call (it is listed in `emptied`)
const ENTRY_MAYBE_EMPTY: u32 = 2;
struct Entry {
    // Hashable position (x, y, z) | KEY_SET (see `claim_key_word`), all KEY_UNSET in a free entry
    // and KEY_REMOVED, KEY_UNSET, KEY_UNSET in a tombstone
    // Not a vec3u to save up 4 bytes on padding
    key: array<atomic<u32>, 3>,
    // ENTRY_TOUCHED | ENTRY_MAYBE_EMPTY
    flags: atomic<u32>,
    // Constructed voxels, bit `x + 4y + 16z`, first word holds z = 0..1, second z = 2..3
    occupancy: array<u32, 2>,
}

@must_use
fn get_hashable_pos(pos: vec3i) -> vec3u {
    return vec3u(bitcast<u32>(pos.x) >> BITS_FOR_ENTRY_VOXEL_COUNT, //
    bitcast<u32>(pos.y) >> BITS_FOR_ENTRY_VOXEL_COUNT, //
    bitcast<u32>(pos.z) >> BITS_FOR_ENTRY_VOXEL_COUNT);
}

const P1: u32 = 73856093;
const P2: u32 = 19349669;
const P3: u32 = 83492791;
@must_use
fn hash(pos: vec3u) -> u32 {
    return ((pos.x * P1) ^ (pos.y * P2) ^ (pos.z * P3)) % BUCKET_COUNT;
}

@must_use
fn probe_count() -> u32 {
    return min(MAX_PROBE, entry_count());
}

// Entry probed in the `probe`-th step for keys of `bucket`
@must_use
fn probed_entry(bucket: u32, probe: u32) -> u32 {
    return (bucket * BUCKET_SIZE + probe) % entry_count();
}

// Writes `value` to the key word unless it is already set
// Returns the value the word holds afterwards, which is final
// Entries are accessed by index, since naga does not support storage pointers as function arguments
@must_use
fn claim_key_word(entry: u32, word: u32, value: u32) -> u32 {
    let key_word = &entries[entry].key[word];
    // A set word never changes, so only an unset one needs the exchange
    let current = atomicLoad(key_word);
    if current != KEY_UNSET {
        return current;
    }
    var result = atomicCompareExchangeWeak(key_word, KEY_UNSET, value);
    // Retried only when it failed spuriously, otherwise some other thread has set the word
    while !result.exchanged && result.old_value == KEY_UNSET {
        result = atomicCompareExchangeWeak(key_word, KEY_UNSET, value);
    }
    if result.exchanged {
        return value;
    }
    return result.old_value;
}

// Returns index of the entry containing `pos`, it is inserted if it is not stored yet
// Returns NO_ENTRY if there is no space left for it
// Key words are claimed in order (x, y, z) and only while the previous ones match,
// so whoever sets a word goes on to the next one and no entry is left partially written
// Since every word is set once, all threads inserting the same key end up in the same entry
@must_use
fn find_or_insert(pos: vec3i) -> u32 {
    let hashable_pos = get_hashable_pos(pos);
    let key = hashable_pos | vec3u(KEY_SET);
    let bucket = hash(hashable_pos);
    for (var probe = 0u; probe < probe_count(); probe++) {
        let entry = probed_entry(bucket, probe);
        // Not a single condition, so the order of claims does not depend on short-circuiting
        if claim_key_word(entry, 0, key.x) != key.x {
            continue;
        }
        if claim_key_word(entry, 1, key.y) != key.y {
            continue;
        }
        if claim_key_word(entry, 2, key.z) == key.z {
            return entry;
        }
    }
    return NO_ENTRY;
}

// Returns index of the entry containing `pos` or NO_ENTRY if it is not stored
// Valid only when nothing is inserted during the same dispatch, every entry is then either free or fully written
// No stored key is ever left behind a free entry, so all entries probed before the one containing `pos`
// are occupied or tombstones
@must_use
fn find_existing(pos: vec3i) -> u32 {
    let hashable_pos = get_hashable_pos(pos);
    let key = hashable_pos | vec3u(KEY_SET);
    let bucket = hash(hashable_pos);
    for (var probe = 0u; probe < probe_count(); probe++) {
        let entry = probed_entry(bucket, probe);
        let x = atomicLoad(&entries[entry].key[0]);
        if x == KEY_UNSET {
            return NO_ENTRY;
        }
        if x == key.x //
        && atomicLoad(&entries[entry].key[1]) == key.y //
        && atomicLoad(&entries[entry].key[2]) == key.z {
            return entry;
        }
    }
    return NO_ENTRY;
}

// Lists the entry in `altered` when it is touched for the first time during the current call
fn mark_touched(entry: u32) {
    if (atomicOr(&entries[entry].flags, ENTRY_TOUCHED) & ENTRY_TOUCHED) == 0 {
        altered.entries[atomicAdd(&altered.len, 1)] = entry;
    }
}

// Lists the entry in `emptied` when any of its voxels is zeroed for the first time during the current call
fn mark_maybe_empty(entry: u32) {
    if (atomicOr(&entries[entry].flags, ENTRY_MAYBE_EMPTY) & ENTRY_MAYBE_EMPTY) == 0 {
        emptied.entries[atomicAdd(&emptied.len, 1)] = entry;
    }
}

// Sum of `current` and `val` saturating at `params.max_value`
// `min(val, ~current)` keeps the sum from wrapping around before it is capped
@must_use
fn capped_add(current: u32, val: u32) -> u32 {
    return min(current + min(val, ~current), params.max_value);
}

// A plain atomicAdd followed by atomicMin could wrap around
fn atomic_capped_add(voxel: u32, val: u32) {
    let a = &voxels[voxel].value;
    var current = atomicLoad(a);
    var new_val = capped_add(current, val);
    // Nothing to write once the voxel is at the cap
    while new_val != current && !atomicCompareExchangeWeak(a, current, new_val).exchanged {
        current = atomicLoad(a);
        new_val = capped_add(current, val);
    }
}

// Value is dropped if there is no space left for it
fn add_val(pos: vec3i, val: StoredValue) {
    let entry = find_or_insert(pos);
    // NO_ENTRY at this point means that there is no space left
    if entry != NO_ENTRY && val != 0 {
        atomic_capped_add(voxel_pos(entry, get_z_order(pos)), val);
        mark_touched(entry);
    }
}

// Returns value of the voxel from before the subtraction
@must_use
fn atomic_sat_sub(voxel: u32, val: u32) -> u32 {
    let a = &voxels[voxel].value;
    var current = atomicLoad(a);
    var new_val = current - min(current, val);
    while !atomicCompareExchangeWeak(a, current, new_val).exchanged {
        current = atomicLoad(a);
        new_val = current - min(current, val);
    }
    return current;
}

// Must not be used during a dispatch that inserts (see `find_existing`)
fn sub_val_if_exists(pos: vec3i, val: StoredValue) {
    let entry = find_existing(pos);
    if entry == NO_ENTRY {
        return;
    }
    let prev = atomic_sat_sub(voxel_pos(entry, get_z_order(pos)), val);
    if prev != 0 && val != 0 {
        mark_touched(entry);
    }
    let just_zeroed = prev != 0 && prev <= val;
    if just_zeroed {
        mark_maybe_empty(entry);
    }
}

// End of Hashmap implementation

// Distance (in ray lengths) from `position` to the first voxel boundary on each axis
// Lanes where `ray` is 0 are meaningless and must not be used
@must_use
fn closest_boundary(dir: vec3i, position: vec3f, ray: vec3f) -> vec3f {
    let boundary = floor(position) + vec3f(dir > vec3i(0));
    return (boundary - position) / ray;
}

// Inserts nothing, so it can use `sub_val_if_exists`
@compute @workgroup_size(WORKGROUP_SIZE)
fn ray_march(@builtin(global_invocation_id) global_id: vec3u, @builtin(num_workgroups) num_workgroups: vec3u) {
    let index = thread_index(global_id, num_workgroups);
    if index >= params.count {
        return;
    }
    // http://www.cse.yorku.ca/~amana/research/grid.pdf
    let input = bitcast<vec4f>(input_points[index]);
    let point = input.xyz / params.resolution;
    let hit_value = u32(input.w) * params.hit_multiplier;
    let target_point = vec3i(floor(point));
    let sensor_pos_divided = vec3f(params.camera_x, params.camera_y, params.camera_z) / params.resolution;
    var sensor_pos = vec3i(floor(sensor_pos_divided));
    // dir cannot be calculated from `ray` due to desired rounding before subtraction
    let dir = sign(target_point - sensor_pos);
    let ray = point - sensor_pos_divided;
    // Every iteration moves exactly one voxel towards the target, so this is the exact step count
    // Used instead of `target_point != sensor_pos` so float errors can't cause an infinite loop
    let dist = abs(target_point - sensor_pos);
    let steps = dist.x + dist.y + dist.z;
    let cmp = vec3u(dir == vec3i(0));
    switch (cmp.x << u32(2)) | (cmp.y << u32(1)) | cmp.z {
    // Straight line
    // x
    case 3: {
        while target_point.x != sensor_pos.x {
            sub_val_if_exists(sensor_pos, hit_value);
            sensor_pos.x += dir.x;
        }
    }
    // y
    case 5: {
        while target_point.y != sensor_pos.y {
            sub_val_if_exists(sensor_pos, hit_value);
            sensor_pos.y += dir.y;
        }
    }
    // z
    case 6: {
        while target_point.z != sensor_pos.z {
            sub_val_if_exists(sensor_pos, hit_value);
            sensor_pos.z += dir.z;
        }
    }

    // 2D
    // x, y
    case 1: {
        let delta = 1.0 / abs(ray.xy);
        var t_max = closest_boundary(dir, sensor_pos_divided, ray).xy;
        for (var i = 0; i < steps; i++) {
            sub_val_if_exists(sensor_pos, hit_value);
            if t_max.x < t_max.y {
                sensor_pos.x += dir.x;
                t_max.x += delta.x;
            } else {
                sensor_pos.y += dir.y;
                t_max.y += delta.y;
            }
        }
    }
    // x, z
    case 2: {
        let delta = 1.0 / abs(ray.xz);
        var t_max = closest_boundary(dir, sensor_pos_divided, ray).xz;
        for (var i = 0; i < steps; i++) {
            sub_val_if_exists(sensor_pos, hit_value);
            if t_max.x < t_max.y {
                sensor_pos.x += dir.x;
                t_max.x += delta.x;
            } else {
                sensor_pos.z += dir.z;
                t_max.y += delta.y;
            }
        }
    }
    // y, z
    case 4: {
        let delta = 1.0 / abs(ray.yz);
        var t_max = closest_boundary(dir, sensor_pos_divided, ray).yz;
        for (var i = 0; i < steps; i++) {
            sub_val_if_exists(sensor_pos, hit_value);
            if t_max.x < t_max.y {
                sensor_pos.y += dir.y;
                t_max.x += delta.x;
            } else {
                sensor_pos.z += dir.z;
                t_max.y += delta.y;
            }
        }
    }

    // 3D
    case 0: {
        let delta = 1.0 / abs(ray);
        var t_max = closest_boundary(dir, sensor_pos_divided, ray);
        for (var i = 0; i < steps; i++) {
            sub_val_if_exists(sensor_pos, hit_value);
            if t_max.x < t_max.z {
                if t_max.x < t_max.y {
                    sensor_pos.x += dir.x;
                    t_max.x += delta.x;
                } else {
                    sensor_pos.y += dir.y;
                    t_max.y += delta.y;
                }
            } else {
                if t_max.y < t_max.z {
                    sensor_pos.y += dir.y;
                    t_max.y += delta.y;
                } else {
                    sensor_pos.z += dir.z;
                    t_max.z += delta.z;
                }
            }
        }
    }default /*7*/ : { /*NOOP*/ } //
    }
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn add_point(@builtin(global_invocation_id) global_id: vec3u, @builtin(num_workgroups) num_workgroups: vec3u) {
    let index = thread_index(global_id, num_workgroups);
    if index >= params.count {
        return;
    }
    let input = bitcast<vec4f>(input_points[index]);
    let pos = vec3i(floor(input.xyz / params.resolution));
    let val = u32(input.w) * params.target_multiplier;
    add_val(pos, val);
}

fn append_output(pos: vec3i, value: u32) {
    output.data[atomicAdd(&output.len, 1)] = vec4i(pos, bitcast<i32>(value));
}

// One thread per half (4x4x2 block) of an altered entry
// Reports voxels changed since the previous call (or blocks whose voxels got constructed or destroyed)
// and makes the current values the reference for the next call
// Only the owning thread touches `prev` and `occupancy` of its block, so no synchronization is needed
@compute @workgroup_size(WORKGROUP_SIZE)
fn collect(@builtin(global_invocation_id) global_id: vec3u, @builtin(num_workgroups) num_workgroups: vec3u) {
    let index = thread_index(global_id, num_workgroups);
    if index >= params.count * 2 {
        return;
    }
    let entry = altered.entries[params.first + index / 2];
    let half = index % 2;
    // Shifting left restores the voxel coordinates (with the sign) from the hashable position
    // and drops KEY_SET
    let origin = bitcast<vec3i>(vec3u(atomicLoad(&entries[entry].key[0]), //
    atomicLoad(&entries[entry].key[1]), //
    atomicLoad(&entries[entry].key[2])) << vec3u(BITS_FOR_ENTRY_VOXEL_COUNT));

    var pack = entries[entry].occupancy[half];
    var changed = false;
    for (var bit = 0u; bit < ENTRY_VOXEL_COUNT / 2; bit++) {
        let local = vec3i(vec3u(bit & 3,(bit >> 2) & 3,(bit >> 4) | (half << 1)));
        let voxel = voxel_pos(entry, get_z_order(local));
        let value = atomicLoad(&voxels[voxel].value);
        let prev = voxels[voxel].prev;
        if value == prev {
            continue;
        }
        voxels[voxel].prev = value;
        let mask = 1u << bit;
        if params.compress == 0 {
            append_output(origin + local, value);
        } else if prev < params.construct_threshold && value >= params.construct_threshold {
            pack |= mask;
            changed = true;
        } else if prev > params.destroy_threshold && value <= params.destroy_threshold {
            if (pack & mask) == 0 {
                atomicStore(&output.error, 1);
            }
            pack &= ~mask;
            changed = true;
        }
    }

    if changed {
        entries[entry].occupancy[half] = pack;
        append_output(origin + vec3i(0, 0, i32(half << 1)), pack);
    }
    if half == 0 {
        atomicAnd(&entries[entry].flags, ~ENTRY_TOUCHED);
    }
}

// One thread per entry listed in `emptied`, runs after `collect`
// Removes the entry when it is still empty (`add_point` may have refilled it), leaving a tombstone
// `collect` made every `prev` equal to its value, so an entry with all values 0 and no voxel constructed
// is all zero and only its key and flags need to be reset to be free once reclaimed
// Only the owning thread touches its entry, so no synchronization is needed
@compute @workgroup_size(WORKGROUP_SIZE)
fn remove_empty(@builtin(global_invocation_id) global_id: vec3u, @builtin(num_workgroups) num_workgroups: vec3u) {
    let index = thread_index(global_id, num_workgroups);
    if index >= atomicLoad(&emptied.len) {
        return;
    }
    let entry = emptied.entries[index];
    var empty = (entries[entry].occupancy[0] | entries[entry].occupancy[1]) == 0;
    for (var zorder = 0u; empty && zorder < ENTRY_VOXEL_COUNT; zorder++) {
        empty = atomicLoad(&voxels[voxel_pos(entry, zorder)].value) == 0;
    }
    if !empty {
        atomicAnd(&entries[entry].flags, ~ENTRY_MAYBE_EMPTY);
        return;
    }
    atomicStore(&entries[entry].key[1], KEY_UNSET);
    atomicStore(&entries[entry].key[2], KEY_UNSET);
    atomicStore(&entries[entry].flags, 0);
    atomicStore(&entries[entry].key[0], KEY_REMOVED);
}

// One thread per entry listed in `emptied`, runs after `remove_empty`, so every tombstone is visible
// Frees tombstones no stored key can be probed past, that is the ones followed only by tombstones
// up to a free entry or for `probe_count() - 1` entries (no key is stored further from its bucket)
// Nothing is inserted and the only change is a tombstone getting free, so an entry seen as a tombstone
// or free stays so, whatever order changes made by other threads are seen in,
// and no stored key is ever left behind a free entry
@compute @workgroup_size(WORKGROUP_SIZE)
fn reclaim_removed(@builtin(global_invocation_id) global_id: vec3u, @builtin(num_workgroups) num_workgroups: vec3u) {
    let index = thread_index(global_id, num_workgroups);
    if index >= atomicLoad(&emptied.len) {
        return;
    }
    let entry = emptied.entries[index];
    if atomicLoad(&entries[entry].key[0]) != KEY_REMOVED {
        return;
    }
    // Number of entries, starting with this one, that get free
    var count = 1u;
    for (var i = 1u; i < probe_count(); i++) {
        let key = atomicLoad(&entries[(entry + i) % entry_count()].key[0]);
        if key == KEY_UNSET {
            count = i;
            break;
        }
        if key != KEY_REMOVED {
            // A stored key may be probed past this tombstone
            return;
        }
    }
    for (var i = 0u; i < count; i++) {
        atomicStore(&entries[(entry + i) % entry_count()].key[0], KEY_UNSET);
    }
    // Tombstones right before a free entry are free as well
    for (var i = 1u; i < probe_count(); i++) {
        let previous = (entry + entry_count() - i) % entry_count();
        if atomicLoad(&entries[previous].key[0]) != KEY_REMOVED {
            break;
        }
        atomicStore(&entries[previous].key[0], KEY_UNSET);
    }
}

// Sets values of voxels (inserting their entries when needed), input is (x, y, z, value)
// The new values are not reported as changes
@compute @workgroup_size(WORKGROUP_SIZE)
fn set_voxels(@builtin(global_invocation_id) global_id: vec3u, @builtin(num_workgroups) num_workgroups: vec3u) {
    let index = thread_index(global_id, num_workgroups);
    if index >= params.count {
        return;
    }
    let input = input_points[index];
    let pos = bitcast<vec3i>(input.xyz);
    let entry = find_or_insert(pos);
    if entry != NO_ENTRY {
        let voxel = voxel_pos(entry, get_z_order(pos));
        atomicStore(&voxels[voxel].value, input.w);
        voxels[voxel].prev = input.w;
    }
}
