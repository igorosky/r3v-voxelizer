# Table of the functions that take the most CPU time, from folded stacks
# ("frame;frame;... weight" lines, as cargo flamegraph writes them):
#
#   awk -f benches/hotspots.awk target/flamegraph/stacks.folded
#
# bench.sh --flamegraph runs it. total is the time with the functions called,
# counted once per stack so that recursion does not count twice, self the time
# in the function itself. Weights are taken as nanoseconds (perf cpu-clock),
# summed over every thread. Names are shortened: module paths are dropped down
# to the type, or to the last module of a free function, generic arguments are
# kept. Functions whose short names are the same are counted as one. POSIX awk.

# Drops path segments from every path in name: a::b::c::f -> c::f,
# a::b::Type -> Type.
function short(name, start, rest) {
    if (name in shortened) return shortened[name]
    rest = name
    name = " " name
    # A module followed by another one (not by a function with ::<T>)
    while (match(name, /[^A-Za-z0-9_:][a-z_][a-z0-9_]*::[a-z_][a-z0-9_]*::[A-Za-z_]/)) {
        name = cut(name, RSTART + 1)
    }
    # A module followed by a type
    while (match(name, /[^A-Za-z0-9_:][a-z_][a-z0-9_]*::[A-Z]/)) {
        name = cut(name, RSTART + 1)
    }
    shortened[rest] = substr(name, 2)
    return shortened[rest]
}

# Removes the path segment ("module::") starting at start.
function cut(name, start, rest) {
    rest = substr(name, start)
    match(rest, /^[a-z_][a-z0-9_]*::/)
    return substr(name, 1, start - 1) substr(rest, RLENGTH + 1)
}

{
    weight = $NF
    depth = split(substr($0, 1, length($0) - length($NF) - 1), frame, ";")
    all += weight
    if (depth < 2) next
    # frame[1] is the process name
    split("", seen)
    for (i = 2; i <= depth; i++) {
        name = short(frame[i])
        if (!(name in seen)) {
            seen[name] = 1
            total[name] += weight
        }
    }
    self[name] += weight
}

END {
    if (all == 0) {
        print "no samples"
        exit 1
    }
    printf "%.3f s of CPU time, summed over every thread\n\n", all / 1e9
    printf "%17s %17s  %s\n", "total", "self", "function"
    order = "sort -t '\t' -k1,1nr -k2,2nr | cut -f3-"
    for (name in total) {
        s = (name in self) ? self[name] : 0
        printf "%.0f\t%.0f\t%6.2f%% %8.3f s %6.2f%% %8.3f s  %s\n", total[name], s,
            100 * total[name] / all, total[name] / 1e9, 100 * s / all, s / 1e9,
            name | order
    }
    close(order)
}
