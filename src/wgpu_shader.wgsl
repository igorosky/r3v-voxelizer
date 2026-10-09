// Number of hashmap buckets, set by the host when the pipelines are created
override BUCKET_COUNT: u32;

@group(0) @binding(0)
var<uniform> params: Params;
// Raw bits, so it can carry both points (f32) and voxels to set (i32 position, u32 value)
@group(0) @binding(1)
var<storage, read> input_points: array<vec4u>;
@group(0) @binding(2)
var<storage, read_write> altered: Altered;
@group(0) @binding(3)
var<storage, read_write> output: Output;
// Sized by BUCKET_COUNT, can't be one struct since only the last member can be runtime sized
@group(1) @binding(0)
var<storage, read_write> buckets: array<Bucket>;
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
    _padding: u32,
}

// Entries touched during the current call, every entry is listed at most once
struct Altered {
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

// RWLock
// Locks of the buckets are accessed directly by bucket index,
// since naga does not support storage pointers as function arguments
const LOCK_FREE: u32 = 0;
const LOCK_WRITE_BIT: u32 = 31;
const LOCK_WRITE_BIT_MASK: u32 = 1 << LOCK_WRITE_BIT;
const LOCK_UNLOCK_BIT_MASK: u32 = ~LOCK_WRITE_BIT_MASK;

// Tries to lock for reading
// returns whether succeed, fails if locked for writing
@must_use
fn try_lock_read(bucket: u32) -> bool {
    let prev = atomicAdd(&buckets[bucket].lock, 1);
    if (prev & LOCK_WRITE_BIT_MASK) != 0 {
        atomicSub(&buckets[bucket].lock, 1);
        return false;
    }
    return true;
}
fn unlock_read(bucket: u32) {
    atomicSub(&buckets[bucket].lock, 1);
}

// Tries to lock for writing
// returns whether succeed, fails if locked for reading or may fail spuriously
@must_use
fn try_lock_write(bucket: u32) -> bool {
    // No need to handle spurious fail since caller will have to have retry mechanism anyway
    return atomicCompareExchangeWeak(&buckets[bucket].lock, LOCK_FREE, LOCK_WRITE_BIT_MASK).exchanged;
}
fn unlock_write(bucket: u32) {
    atomicAnd(&buckets[bucket].lock, LOCK_UNLOCK_BIT_MASK);
}
// End of RWLock

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

const BITS_FOR_ENTRY_VOXEL_COUNT: u32 = 2;
const BITS_FOR_ENTRY_VOXEL_MASK: u32 = (1u << BITS_FOR_ENTRY_VOXEL_COUNT) - 1;
const ENTRY_VOXEL_COUNT: u32 = 1u << (3 * BITS_FOR_ENTRY_VOXEL_COUNT);
const BUCKET_SIZE: u32 = 2;
// How many buckets following the full one are searched for a free entry when extending its list
const MAX_LIST_PROBE: u32 = 64;

// Returned instead of an entry index
const NO_ENTRY: u32 = 0xFFFFFFFFu;
const ENTRY_CONTENDED: u32 = 0xFFFFFFFEu;

@must_use
fn entry_count() -> u32 {
    return BUCKET_COUNT * BUCKET_SIZE;
}

alias StoredValue = u32;
@must_use
fn voxel_pos(entry: u32, zorder: u32) -> u32 {
    return (entry << (BITS_FOR_ENTRY_VOXEL_COUNT * 3)) | zorder;
}

const ENTRY_OCCUPIED_MASK: u32 = u32(1) << u32(31);
const ENTRY_OFFSET_MASK: u32 = ~ENTRY_OCCUPIED_MASK;
// Entry has been modified during the current call (it is listed in `altered`)
const ENTRY_TOUCHED: u32 = 1;
// Any voxel of the entry has ever been constructed
const ENTRY_CONSTRUCTED: u32 = 2;
struct Entry {
    // Not a vec3i to save up 4 bytes on padding
    x: u32,
    y: u32,
    z: u32,
    // Occupied bit + list offset (see `list_entry`)
    // In the last entry of a bucket it is the head of the bucket's list
    // In an entry being a list element it points to the next element
    offset: atomic<u32>,
    // ENTRY_TOUCHED | ENTRY_CONSTRUCTED
    flags: atomic<u32>,
    // Constructed voxels, bit `x + 4y + 16z`, first word holds z = 0..1, second z = 2..3
    occupancy: array<u32, 2>,
}
struct Bucket {
    lock: atomic<u32>,
    entries: array<Entry, BUCKET_SIZE>,
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
fn entry_index(bucket: u32, i: u32) -> u32 {
    return bucket * BUCKET_SIZE + i;
}

// List offsets are relative to the first entry of the list's home bucket, 0 means end of the list
@must_use
fn list_entry(bucket: u32, offset: u32) -> u32 {
    return (entry_index(bucket, 0) + offset) % entry_count();
}

@must_use
fn is_occupied(bucket: u32, i: u32) -> bool {
    return (atomicLoad(&buckets[bucket].entries[i].offset) & ENTRY_OCCUPIED_MASK) != 0;
}

@must_use
fn next_offset(bucket: u32, i: u32) -> u32 {
    return atomicLoad(&buckets[bucket].entries[i].offset) & ENTRY_OFFSET_MASK;
}

@must_use
fn occupied_contains(bucket: u32, i: u32, hashable_pos: vec3u) -> bool {
    return buckets[bucket].entries[i].x == hashable_pos.x //
    && buckets[bucket].entries[i].y == hashable_pos.y //
    && buckets[bucket].entries[i].z == hashable_pos.z;
}

@must_use
fn contains(bucket: u32, i: u32, hashable_pos: vec3u) -> bool {
    return is_occupied(bucket, i) && occupied_contains(bucket, i, hashable_pos);
}

fn set_pos(bucket: u32, i: u32, hashable_pos: vec3u) {
    buckets[bucket].entries[i].x = hashable_pos.x;
    buckets[bucket].entries[i].y = hashable_pos.y;
    buckets[bucket].entries[i].z = hashable_pos.z;
}

// Returns index of the entry containing `hashable_pos` or NO_ENTRY
// Caller must hold a lock (read or write) of `bucket`
@must_use
fn find_entry(bucket: u32, hashable_pos: vec3u) -> u32 {
    // Check the bucket
    for (var i = 0u; i < BUCKET_SIZE; i++) {
        if contains(bucket, i, hashable_pos) {
            return entry_index(bucket, i);
        }
    }

    // Not in the bucket check a link list
    var offset = next_offset(bucket, BUCKET_SIZE - 1);
    while offset != 0 {
        let entry = list_entry(bucket, offset);
        let list_elem_bucket = entry / BUCKET_SIZE;
        let i = entry % BUCKET_SIZE;
        // Since it is pointed to it must contain some element
        if occupied_contains(list_elem_bucket, i, hashable_pos) {
            return entry;
        }
        offset = next_offset(list_elem_bucket, i);
    }
    return NO_ENTRY;
}

// Whether `bucket` has a free entry which can become an element of another bucket's list
@must_use
fn has_free_list_slot(bucket: u32) -> bool {
    for (var i = 0u; i < BUCKET_SIZE - 1; i++) {
        if !is_occupied(bucket, i) {
            return true;
        }
    }
    return false;
}

// Stores `hashable_pos` in a free entry of `bucket` or, if it is full, extends the bucket's list
// Returns index of the new entry, NO_ENTRY if there is no space left
// or ENTRY_CONTENDED if it failed due to contention and should be retried
// Caller must hold the write lock of `bucket` and make sure `hashable_pos` is not stored yet
@must_use
fn insert_entry(bucket: u32, hashable_pos: vec3u) -> u32 {
    // Check the bucket
    for (var i = 0u; i < BUCKET_SIZE; i++) {
        if !is_occupied(bucket, i) {
            set_pos(bucket, i, hashable_pos);
            // Or instead of store, so the list head (in case of the last entry) is not lost
            atomicOr(&buckets[bucket].entries[i].offset, ENTRY_OCCUPIED_MASK);
            return entry_index(bucket, i);
        }
    }

    // Bucket is full - extend its list with a free entry of one of the following buckets
    // Last entries are skipped, since their offset is reserved for their own bucket's list
    // Bucket that gets the entry is write locked, so its readers never see a half written entry
    // It only tries to lock, so it can't deadlock
    // Probing is limited to the other buckets, `bucket` itself is already locked by the caller
    let probes = min(MAX_LIST_PROBE, BUCKET_COUNT - 1);
    var contended = false;
    for (var probe = 1u; probe <= probes; probe++) {
        let other = (bucket + probe) % BUCKET_COUNT;
        // Entries are never freed, so a bucket without a free entry is skipped without locking it
        // Otherwise, once the table is full, threads would keep failing on each other's locks forever
        if !has_free_list_slot(other) {
            continue;
        }
        if !try_lock_write(other) {
            contended = true;
            continue;
        }
        for (var i = 0u; i < BUCKET_SIZE - 1; i++) {
            if !is_occupied(other, i) {
                set_pos(other, i, hashable_pos);
                // New element becomes the head of the list
                let head = &buckets[bucket].entries[BUCKET_SIZE - 1].offset;
                atomicStore(&buckets[other].entries[i].offset, //
                ENTRY_OCCUPIED_MASK | (atomicLoad(head) & ENTRY_OFFSET_MASK));
                unlock_write(other);
                atomicStore(head, ENTRY_OCCUPIED_MASK | (probe * BUCKET_SIZE + i));
                return entry_index(other, i);
            }
        }
        unlock_write(other);
    }
    if contended {
        return ENTRY_CONTENDED;
    }
    return NO_ENTRY;
}

// Returns index of the entry containing `pos` or NO_ENTRY if it is not stored
// or ENTRY_CONTENDED if it failed due to contention and has to be retried
@must_use
fn find_existing(pos: vec3i) -> u32 {
    let hashable_pos = get_hashable_pos(pos);
    let bucket = hash(hashable_pos);
    if !try_lock_read(bucket) {
        return ENTRY_CONTENDED;
    }
    let found = find_entry(bucket, hashable_pos);
    unlock_read(bucket);
    return found;
}

// Returns index of the entry containing `pos`, it is inserted if it is not stored yet
// Returns NO_ENTRY if there is no space left for it
// or ENTRY_CONTENDED if it failed due to contention and has to be retried
@must_use
fn find_or_insert(pos: vec3i) -> u32 {
    let hashable_pos = get_hashable_pos(pos);
    let bucket = hash(hashable_pos);
    if !try_lock_write(bucket) {
        return ENTRY_CONTENDED;
    }

    // Anything that has been learned while lookup should not be considered
    // since while unlocking/locking everything could have changed
    var entry = find_entry(bucket, hashable_pos);
    if entry == NO_ENTRY {
        entry = insert_entry(bucket, hashable_pos);
    }
    unlock_write(bucket);
    return entry;
}

// Returns index of the entry containing `pos`, it is inserted if it is not stored yet
// Returns NO_ENTRY if there is no space left for it
// Retries on contention, every failed attempt means that some other thread holds the lock and makes progress
@must_use
fn get_or_insert(pos: vec3i) -> u32 {
    var entry = find_existing(pos);
    while entry == ENTRY_CONTENDED {
        entry = find_existing(pos);
    }
    if entry != NO_ENTRY {
        return entry;
    }
    // Lock can't be escalated to write since this may cause a deadlock, so it is unlocked and locked
    // Once the entry is known to be missing only the write lock is retried,
    // taking the read lock again on every retry would keep starving the writers of the bucket
    entry = find_or_insert(pos);
    while entry == ENTRY_CONTENDED {
        entry = find_or_insert(pos);
    }
    return entry;
}

// Lists the entry in `altered` when it is touched for the first time during the current call
fn mark_touched(entry: u32) {
    let flags = &buckets[entry / BUCKET_SIZE].entries[entry % BUCKET_SIZE].flags;
    if (atomicOr(flags, ENTRY_TOUCHED) & ENTRY_TOUCHED) == 0 {
        altered.entries[atomicAdd(&altered.len, 1)] = entry;
    }
}

// Value is dropped if there is no space left for it
fn add_val(pos: vec3i, val: StoredValue) {
    let entry = get_or_insert(pos);
    // NO_ENTRY at this point means that there is no space left
    // Entries are never moved nor removed, so once found the voxel can be accessed without a lock
    if entry != NO_ENTRY && val != 0 {
        atomicAdd(&voxels[voxel_pos(entry, get_z_order(pos))].value, val);
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

fn try_remove(bucket: u32, hash_pos: u32, i: u32) {
// NOOP for now - particularly in mostly static env it is not that necessary
// Also it would require to introduce some new variable to bucket element - like non_zero_voxels_count
// while !try_lock_write(bucket) { }
// if atomicLoad(&buckets[bucket].entries[i].offset) & ENTRY_OCCUPIED_MASK == 0 { }
// unlock_write(bucket);
}

// Returns false if it failed due to contention and has to be retried
@must_use
fn sub_val_if_exists(pos: vec3i, val: StoredValue) -> bool {
    let hashable_pos = get_hashable_pos(pos);
    let bucket = hash(hashable_pos);
    if !try_lock_read(bucket) {
        return false;
    }

    let entry = find_entry(bucket, hashable_pos);
    var just_zeroed = false;
    if entry != NO_ENTRY {
        let prev = atomic_sat_sub(voxel_pos(entry, get_z_order(pos)), val);
        if prev != 0 && val != 0 {
            mark_touched(entry);
        }
        just_zeroed = prev != 0 && prev <= val;
    }
    unlock_read(bucket);
    if just_zeroed {
    // TODO
    }
    // No lock failed - not finding an entry just means that such entry does not exist
    // so, everything is fine
    return true;
}

// End of Hashmap implementation

// Distance (in ray lengths) from `position` to the first voxel boundary on each axis
// Lanes where `ray` is 0 are meaningless and must not be used
@must_use
fn closest_boundary(dir: vec3i, position: vec3f, ray: vec3f) -> vec3f {
    let boundary = floor(position) + vec3f(dir > vec3i(0));
    return (boundary - position) / ray;
}

// Hits the voxel, retrying until the bucket's lock is taken
fn hit(pos: vec3i, val: StoredValue) {
    while !sub_val_if_exists(pos, val) { }
}

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
            hit(sensor_pos, hit_value);
            sensor_pos.x += dir.x;
        }
    }
    // y
    case 5: {
        while target_point.y != sensor_pos.y {
            hit(sensor_pos, hit_value);
            sensor_pos.y += dir.y;
        }
    }
    // z
    case 6: {
        while target_point.z != sensor_pos.z {
            hit(sensor_pos, hit_value);
            sensor_pos.z += dir.z;
        }
    }

    // 2D
    // x, y
    case 1: {
        let delta = 1.0 / abs(ray.xy);
        var t_max = closest_boundary(dir, sensor_pos_divided, ray).xy;
        for (var i = 0; i < steps; i++) {
            hit(sensor_pos, hit_value);
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
            hit(sensor_pos, hit_value);
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
            hit(sensor_pos, hit_value);
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
            hit(sensor_pos, hit_value);
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
    let bucket = entry / BUCKET_SIZE;
    let i = entry % BUCKET_SIZE;
    // Shifting left restores the voxel coordinates (with the sign) from the hashable position
    let origin = bitcast<vec3i>(vec3u(buckets[bucket].entries[i].x, //
    buckets[bucket].entries[i].y, //
    buckets[bucket].entries[i].z) << vec3u(BITS_FOR_ENTRY_VOXEL_COUNT));

    var pack = buckets[bucket].entries[i].occupancy[half];
    var changed = false;
    var constructed = false;
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
            constructed = true;
        } else if prev > params.destroy_threshold && value <= params.destroy_threshold {
            if (pack & mask) == 0 {
                atomicStore(&output.error, 1);
            }
            pack &= ~mask;
            changed = true;
        }
    }

    if changed {
        buckets[bucket].entries[i].occupancy[half] = pack;
        append_output(origin + vec3i(0, 0, i32(half << 1)), pack);
    }
    if constructed {
        atomicOr(&buckets[bucket].entries[i].flags, ENTRY_CONSTRUCTED);
    }
    if half == 0 {
        atomicAnd(&buckets[bucket].entries[i].flags, ~ENTRY_TOUCHED);
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
    let entry = get_or_insert(pos);
    if entry != NO_ENTRY {
        let voxel = voxel_pos(entry, get_z_order(pos));
        atomicStore(&voxels[voxel].value, input.w);
        voxels[voxel].prev = input.w;
    }
}
