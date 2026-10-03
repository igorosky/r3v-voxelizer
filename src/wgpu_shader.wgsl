@group(0) @binding(0)
var<uniform, read> params: Params;
@group(0) @binding(1)
var<storage, read> input_points: array<vec4f>;
@group(0) @binding(2)
var<storage, read_write> altered_points: Output;
@group(0) @binding(3)
var<storage, read_write> compressed_points: Output;
@group(1) @binding(0)
var<storage, read_write> hashmap_storage: Hashmap;

struct Params {
    camera_x: f32,
    camera_y: f32,
    camera_z: f32,
    resolution: f32,
    hit_multiplayer: u32,
    target_multiplayer: u32,
}

struct Output {
    len: atomic<u32>,
    data: array<vec4i>,
}

// RWLock
const LOCK_FREE: u32 = 0;
const LOCK_WRITE_BIT: u32 = 31;
const LOCK_WRITE_BIT_MASK: u32 = 1 << LOCK_WRITE_BIT;
const LOCK_UNLOCK_BIT_MASK: u32 = ~LOCK_WRITE_BIT_MASK;

// Tries to lock for reading
// returns whether succeed, fails if locked for writing
@must_use
fn try_lock_read(lock: ptr<atomic<u32>>) -> bool {
    let prev = atomicAdd(lock, 1);
    if prev & LOCK_WRITE_BIT_MASK != 0 {
        atomicSub(lock, 1);
        return false;
    }
    return true;
}
fn unlock_read(lock: ptr<atomic<u32>>) {
    atomicSub(lock, 1);
}

// Tries to lock for writing
// returns whether succeed, fails if locked for reading or may fail spuriously
@must_use
fn try_lock_write(lock: ptr<atomic<u32>>) -> bool {
    // No need to handle spurious fail since caller will have to have retry mechanism anyway
    return atomicCompareExchangeWeak(lock, LOCK_FREE, LOCK_WRITE_BIT_MASK).exchanged;
}
fn unlock_write(lock: ptr<atomic<u32>>) {
    atomicAnd(ptr, LOCK_UNLOCK_BIT_MASK);
}
// End of RWLock

// Z-Order
@must_use
fn expand_bits(v: u32) -> u32 {
    v = (v | (v << u32(16))) & 0x030000FFu;
    v = (v | (v << u32(8))) & 0x0300F00Fu;
    v = (v | (v << u32(4))) & 0x030C30C3u;
    v = (v | (v << u32(2))) & 0x09249249u;
    return v;
}

@must_use
fn get_z_order(pos: vec3i) -> u32 {
    return (expand_bits(pos.z & BITS_FOR_ENTRY_VOXEL_MASK) << u32(2)) //
    | (expand_bits(pos.y & BITS_FOR_ENTRY_VOXEL_MASK) << u32(1)) //
    | (expand_bits(pos.x & BITS_FOR_ENTRY_VOXEL_MASK));
}
// End of Z-Order

// Hashmap implementation

// https://niessnerlab.org/papers/2013/4hashing/niessner2013hashing.pdf
// https://arxiv.org/pdf/1805.03709

const BITS_FOR_ENTRY_VOXEL_COUNT: u32 = 2;
const BITS_FOR_ENTRY_VOXEL_MASK = pow(2, BITS_FOR_ENTRY_VOXEL_COUNT) - 1;
const ENTRY_VOXEL_COUNT = pow(2, 3 * BITS_FOR_ENTRY_VOXEL_COUNT);
const BUCKET_SIZE = 2;
const BUCKET_COUNT = 2 * pow(2, 20);
const STORED_VOXELS = BUCKET_COUNT * BUCKET_SIZE * ENTRY_VOXEL_COUNT;

alias StoredValue = u32;
const VALUE_SIZE = 4;
@must_use
fn voxel_pos(hash: u32, zorder: u32) -> u32 {
    return (hash << (BITS_FOR_ENTRY_VOXEL_COUNT * 3) | zorder) * VALUE_SIZE;
}

const ENTRY_OCCUPIED_MASK: u32 = u32(1) << u32(31);
const ENTRY_OFFSET_MASK: u32 = ~ENTRY_OCCUPIED_MASK;
struct Entry {
    // Not a vec3i to save up 4 bytes on padding
    x: u32,
    y: u32,
    z: u32,
    offset: atomic<u32>,
}
struct Bucket {
    lock: atomic<u32>,
    entries: array<Entry, BUCKET_SIZE>,
}
struct Hashmap {
    buckets: array<Bucket, BUCKET_COUNT>,
    voxels: array<atomic<u32>, STORED_VOXELS>,
}

@must_use
fn get_hashable_pos(pos: vec3i) -> vec3u {
    return vec3u(bitcast<u32>(pos.x) >> BITS_FOR_ENTRY_VOXEL_COUNT, //
    bitcast<u32>(pos.y) >> BITS_FOR_ENTRY_VOXEL_COUNT, //
    bitcast<u32>(pos.z) >> BITS_FOR_ENTRY_VOXEL_COUNT);
}

const P1 = 73856093;
const P2 = 19349669;
const P3 = 83492791;
@must_use
fn hash(pos: vec3u) -> u32 {
    return (pos.x * P1)
        ^ (pos.y * P2)
        ^ (pos.z * P3)
        % BUCKET_COUNT;
}

@must_use
fn is_occupied(hash: u32, i: u32, hashable_pos: vec3u) -> bool {
    return atomicLoad(&hashmap_storage.buckets[hash].entries[i].offset) & ENTRY_OCCUPIED_MASK != 0;
}

@must_use
fn occupied_contains(hash: u32, i: u32, hashable_pos: vec3u) -> bool {
    return hashmap_storage.buckets[hash].entries[i].x == hashable_pos.x //
    && hashmap_storage.buckets[hash].entries[i].y == hashable_pos.y //
    && hashmap_storage.buckets[hash].entries[i].z == hashable_pos.z;
}

@must_use
fn contains(hash: u32, i: u32, hashable_pos: vec3u) -> bool {
    return is_occupied(hash, i, hashable_pos) && occupied_contains(hash, i, hashable_pos);
}

// return 0 - failed spuriously
// return 1 - already occupied
// return 2 - succeeded
@must_use
fn try_occupy(offset: ptr<atomic<u32>>) -> u32 {
    let result = atomicCompareExchangeWeak(&offset, 0, ENTRY_OCCUPIED_MASK);
    if result.exchanged {
        return 2;
    }
    return result.old_value != 0;
}

@must_use
fn add_val(pos: vec3i, val: StoredValue) -> bool {
    let hashable_pos = get_hashable_pos(pos);
    let hash = hash(hashable_pos);
    let bucket = &hashmap_storage.buckets[hash];
    if !try_lock_read(&hashmap_storage.buckets[hash].lock) {
        return false;
    }

    // Check the bucket
    for (var i = 0; i < BUCKET_SIZE; i++) {
        if contains(hash, i, hashable_pos) {
            atomicAdd(&hashmap_storage.voxels[voxel_pos(hash, get_z_order(pos))], val);
            unlock_read(&hashmap_storage.buckets[hash].lock);
            return true;
        }
    }

    // Bucket empty check a link list
    var offset = atomicLoad(&hashmap_storage.buckets[hash].entries[BUCKET_SIZE - 1].offset) & ENTRY_OFFSET_MASK;
    while offset != 0 {
        let list_elem_hash = (hash + offset) / BUCKET_SIZE;
        let i = (hash + offset) % BUCKET_SIZE;
        // Since it is pointed to it must contain some element
        if occupied_contains(list_elem_hash, i, hashable_pos) {
            atomicAdd(&hashmap_storage.voxels[voxel_pos(list_elem_hash, get_z_order(pos))], val);
            unlock_read(&hashmap_storage.buckets[hash].lock);
            return true;
        }
        offset = hashmap_storage.buckets[list_elem_hash].entries[i].offset;
    }

    // This must be unlocked and locked
    // Lock can't be escalated to write since this may caus a deadlock
    unlock_read(&hashmap_storage.buckets[hash].lock);
    if !try_lock_write(&hashmap_storage.buckets[hash].lock) {
        return false;
    }

    // Anything that has been learned while lookup should not be considered
    // since while unlocking/locking everything could have changed
    for (var i = 0; i < BUCKET_SIZE; i++) {
        if is_occupied(hash, i, hashable_pos) {
            if contains(hash, i, hashable_pos) {
                atomicAdd(&hashmap_storage.voxels[voxel_pos(hash, get_z_order(pos))], val);
                unlock_write(&hashmap_storage.buckets[hash].lock);
                return true;
            }
        } else {
            let offset = &hashmap_storage.buckets[hash].entries[i].offset;
            var result = try_occupy(offset);
            while result != 0 {
                result = try_occupy(offset);
            }
            if result == 2 {
                hashmap_storage.buckets[hash].entries[i].x = hashable_pos.x;
                hashmap_storage.buckets[hash].entries[i].y = hashable_pos.y;
                hashmap_storage.buckets[hash].entries[i].z = hashable_pos.z;
                atomicAdd(&hashmap_storage.voxels[voxel_pos(hash, get_z_order(pos))], val);
                unlock_write(&hashmap_storage.buckets[hash].lock);
                return true;
            }

        }
    }

    offset = atomicLoad(&hashmap_storage.buckets[hash].entries[BUCKET_SIZE - 1].offset) & ENTRY_OFFSET_MASK;
    while offset != 0 {
        let list_elem_hash = (hash + offset) / BUCKET_SIZE;
        let i = (hash + offset) % BUCKET_SIZE;
        // Since it is pointed to it must contain some element
        if occupied_contains(hash, i, hashable_pos) {
            atomicAdd(&hashmap_storage.voxels[voxel_pos(hash, get_z_order(pos))], val);
            unlock_write(&hashmap_storage.buckets[hash].lock);
            return true;
        }
        offset = hashmap_storage.buckets[list_elem_hash].entries[i].offset;
    }

    unlock_write(&hashmap_storage.buckets[hash].lock);
    return false;
}

// Returns true if `a` has been set to zero and wan't before
fn atomic_sat_sub(a: ptr<atomic<u32>>, val: u32) -> bool {
    var current = atomicLoad(a);
    var new_val = current - min(current, val);
    while !atomicCompareExchangeWeak(a, current, new_val).exchanged {
        current = atomicLoad(a);
        new_val = current - min(current, val);
    }
    return new_val == 0 && current != 0;
}

fn try_remove(hash: u32, hash_pos: u32, i: u32) {
// NOOP for now - particularly in mostly static env it is not that necessary
// Also it would require to introduce some new variable to bucket element - like non_zero_voxels_count
// while !try_lock_write(&hashmap_storage.buckets[hash].lock) { }
// if atomicLoad(&hashmap_storage.buckets[hash].entries[i].offset) & ENTRY_OCCUPIED_MASK == 0 { }
// unlock_write(&hashmap_storage.buckets[hash].lock);
}

fn sub_val_if_exists(pos: vec3i, val: StoredValue) -> bool {
    let hashable_pos = get_hashable_pos(pos);
    let hash = hash(hashable_pos);
    let bucket = &hashmap_storage.buckets[hash];
    if !try_lock_read(&hashmap_storage.buckets[hash].lock) {
        return false;
    }

    // Check the bucket
    for (var i = 0; i < BUCKET_SIZE; i++) {
        if contains(hash, i, hashable_pos) {
            let just_zeroed = atomic_sat_sub(&hashmap_storage.voxels[voxel_pos(hash, get_z_order(pos))], val);
            unlock_read(&hashmap_storage.buckets[hash].lock);
            if just_zeroed {
            // TODO
            }
            return true;
        }
    }

    // Bucket empty check a link list
    var offset = atomicLoad(&hashmap_storage.buckets[hash].entries[BUCKET_SIZE - 1].offset) & ENTRY_OFFSET_MASK;
    while offset != 0 {
        let list_elem_hash = (hash + offset) / BUCKET_SIZE;
        let i = (hash + offset) % BUCKET_SIZE;
        // Since it is pointed to it must contain some element
        if occupied_contains(list_elem_hash, i, hashable_pos) {
            let just_zeroed = atomic_sat_sub(&hashmap_storage.voxels[voxel_pos(list_elem_hash, get_z_order(pos))], val);
            unlock_read(&hashmap_storage.buckets[hash].lock);
            if just_zeroed {
            // TODO
            }
            return true;
        }
        offset = hashmap_storage.buckets[list_elem_hash].entries[i].offset;
    }
    unlock_read(&hashmap_storage.buckets[hash].lock);
    // No lock failed - getting here just means that no such entry does not exist
    // so, everything is fine
    return true;
}

// End of Hashmap implementation

@compute @workgroup_size(256)
fn ray_march(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let point = input_points[global_id.x] / params.resolution;
    let intensity = u32(point.w);
    let target_point = vec3i(floor(point));
    var sensor_pos = vec3i(params.camera_x, params.camera_y, params.camera_z);
    let dir = sign(target_point - sensor_pos);
    let cmp = dir == 0;
    let hit_value = intensity * params.hit_multiplayer;
    switch (cmp.x << 2) | (cmp.y << 1) | cmp.z {
    // Straight line
    // x
    case 6: {
        while target_point.x != sensor_pos.x {
            sub_val_if_exists(sensor_pos, hit_value);
            sensor_pos += dir.x;
        }
    }
    // y
    case 5: {
        while target_point.y != sensor_pos.y {
            sub_val_if_exists(sensor_pos, hit_value);
            sensor_pos += dir.y;
        }
    }
    // z
    case 3: {
        while target_point.z != sensor_pos.z {
            sub_val_if_exists(sensor_pos, hit_value);
            sensor_pos += dir.z;
        }
    }

    // 2D
    case 1: { } //
    case 2: { } //
    case 3: { } //

    // 3D
    case 0: { } //

    default /*7*/ : { /*NOOP*/ } //
    }
}

@compute @workgroup_size(256)
fn add_point(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let point = input_points[global_id.x] / params.resolution;
    add_val(a
    :)
}
