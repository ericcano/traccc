#!/bin/bash
# schedule_simple_wrapper.sh
# Wrapper for schedule_simple.cpp to call traccc_throughput_mt_cuda with translated arguments

set -e


# Default values
threads=1
streams=1
warmupEvents=0
events=-1
errorEnabled=0
errorEventId=-1
verbose=0
launchStrategy="single"
builddir="."

# internal options
internalOption="--input-events 10 "

# Additional arguments to pass through
passthrough_args=()

# Print help
print_help() {
    cat <<EOF
Usage: $0 [options]
Options:
  --numberOfThreads <N>   Number of threads to use (default: 1, use 0 to use all CPU cores)
  --numberOfStreams <N>   Number of concurrent events (default: 0 = numberOfThreads)
  --warmupEvents <N>      Number of events to process before starting the benchmark (default: 0)
  --maxEvents <N>         Number of events to process (default: -1 for all events in the input file)
  --error-on              Enable error in FirstAlgorithm (default: off)
  --error-event <N>       Set the event ID where the error occurs (default: -1)
  --verbose               Enable verbose output (default: off)
  --help                  Show this help message
  --launchStrategy <single|graph|cachedGraphs|straightLaunches|straightDelegated|...>
  --                      Pass all remaining arguments directly to the underlying command
EOF
}


# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --numberOfThreads)
            threads="$2"; shift 2;;
        --numberOfStreams)
            streams="$2"; shift 2;;
        --warmupEvents|--warmup)
            warmupEvents="$2"; shift 2;;
        --maxEvents)
            events="$2"; shift 2;;
        --error-on)
            errorEnabled=1; shift;;
        --error-event)
            errorEventId="$2"; shift 2;;
        --verbose)
            verbose=1; shift;;
        --help)
            print_help; exit 0;;
        --launchStrategy)
            launchStrategy="$2"; shift 2;;
        --builddir)
            builddir="$2"; shift 2;;
        --)
            shift; passthrough_args=("$@"); break;;
        *)
            echo "Unknown option: $1"; print_help; exit 1;;
    esac
done


# Translate to traccc_throughput_mt_cuda arguments
cmd=("$builddir/bin/traccc_throughput_mt_cuda")

cmd+=("--cpu-threads" "$threads")
cmd+=("--cold-run-events" "$warmupEvents")
cmd+=("--processed-events" "$events")
cmd+=("--concurrent-slots" "$streams")

# Add passthrough arguments
if [[ ${#passthrough_args[@]} -gt 0 ]]; then
    cmd+=("${passthrough_args[@]}")
fi


# Call the real binary, print output as it goes, and capture for parsing
tmpfile=$(mktemp)
trap 'rm -f "$tmpfile"' EXIT

"${cmd[@]}" 2>&1 | tee "$tmpfile"

# Parse and print the expected output for run-scan.pl
output=$(cat "$tmpfile")

# Extract event count from arguments
event_count="$events"

# Extract event processing throughput line
event_line=$(awk '/Throughput:/ {p=1; next} p && /Event processing/ {print; exit}' "$tmpfile")

if [[ $event_line =~ ([0-9.]+)\ ms/event,\ ([0-9.eE+-]+)\ events/s ]]; then
    ms_per_event="${BASH_REMATCH[1]}"
    throughput="${BASH_REMATCH[2]}"
    # Calculate total time in seconds
    if [[ -n "$event_count" && "$event_count" =~ ^[0-9]+$ ]]; then
        time=$(awk -v ms="$ms_per_event" -v n="$event_count" 'BEGIN { printf "%.6f", (ms*n)/1000.0 }')
    else
        time="0"
    fi
    echo "Processed $event_count events in $time seconds, throughput $throughput events/s, CPU usage per thread: 0%"
else
    # Fallback: print the throughput block
    awk '/^Throughput:/ {p=1; print; next} p && NF==0 {p=0} p' "$tmpfile"
fi
