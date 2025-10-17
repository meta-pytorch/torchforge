#!/bin/bash

# Shared Memory Monitoring Script
# Optimized for systems with many cores

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# History arrays (store last 60 data points = 1 minute)
HISTORY_SIZE=60
shmem_history=()
devshm_history=()

# Track observed maximums (for stable scaling)
# Initialize to current values instead of 1 to avoid massive initial spike
shmem_max=$(grep "^Shmem:" /proc/meminfo | awk '{printf "%.2f", $2/1024/1024}')
devshm_max=$(df --output=used /dev/shm | tail -1 | awk '{printf "%.2f", $1/1024/1024}')

# Ensure they're not zero
if (( $(echo "$shmem_max < 1" | bc -l) )); then shmem_max=1; fi
if (( $(echo "$devshm_max < 1" | bc -l) )); then devshm_max=1; fi

# Function to draw ASCII sparkline with fixed scale
draw_plot() {
    local -n data=$1
    local label=$2
    local color=$3
    local max_val=$4

    # Avoid division by zero
    if (( $(echo "$max_val == 0" | bc -l) )); then
        max_val=1
    fi

    # Draw the sparkline (using block characters)
    local chars=("▁" "▂" "▃" "▄" "▅" "▆" "▇" "█")
    local line="${color}${label}: "

    for val in "${data[@]}"; do
        # Scale to 0-7 range
        local scaled=$(echo "scale=0; ($val / $max_val) * 7" | bc -l)
        scaled=${scaled%.*}  # Remove decimal
        # Clamp to 0-7 range (in case value exceeds max)
        if [ $scaled -gt 7 ]; then scaled=7; fi
        if [ $scaled -lt 0 ]; then scaled=0; fi
        line+="${chars[$scaled]}"
    done

    line+=" (max: $(printf "%.1f" $max_val) GB)${NC}"
    echo -e "$line"
}

while true; do
    # Capture current values for plotting (in GB)
    shmem_gb=$(grep "^Shmem:" /proc/meminfo | awk '{printf "%.2f", $2/1024/1024}')
    devshm_gb=$(df --output=used /dev/shm | tail -1 | awk '{printf "%.2f", $1/1024/1024}')

    # Update observed maximums (only increase, with slow decay)
    if (( $(echo "$shmem_gb > $shmem_max" | bc -l) )); then
        shmem_max=$shmem_gb
    else
        # Slowly decay max (0.1% per second) to allow adaptation to lower values
        shmem_max=$(echo "$shmem_max * 0.999" | bc -l)
    fi

    if (( $(echo "$devshm_gb > $devshm_max" | bc -l) )); then
        devshm_max=$devshm_gb
    else
        devshm_max=$(echo "$devshm_max * 0.999" | bc -l)
    fi

    # Add to history arrays
    shmem_history+=("$shmem_gb")
    devshm_history+=("$devshm_gb")

    # Trim history to max size
    if [ ${#shmem_history[@]} -gt $HISTORY_SIZE ]; then
        shmem_history=("${shmem_history[@]:1}")
    fi
    if [ ${#devshm_history[@]} -gt $HISTORY_SIZE ]; then
        devshm_history=("${devshm_history[@]:1}")
    fi

    # Build entire output in a variable first (prevents flickering)
    output=""

    output+="${GREEN}=== System Memory Overview ===${NC}\n"
    output+="$(free -h --wide)\n"

    output+="\n${YELLOW}=== /dev/shm Usage ===${NC}\n"
    output+="$(df -h /dev/shm | tail -n +2)\n"
    output+="Largest 5 files/dirs (top-level only):\n"
    output+="$(ls -lhS /dev/shm 2>/dev/null | grep '^[-d]' | head -5 | awk '{printf "  %8s  %s\n", $5, $9}')\n"

    output+="\n${BLUE}=== Shared Memory Segments (IPC) ===${NC}\n"
    output+="$(ipcs -m --human 2>/dev/null | head -20 || ipcs -m | head -20)\n"

    output+="\n${GREEN}=== Top 10 Processes by Memory ===${NC}\n"
    output+="$(ps aux --sort=-%mem | head -11 | awk 'NR==1 {print $0} NR>1 {printf "%-10s %6s %6s %10s %s\n", $1, $2, $4, $6, $11}')\n"

    output+="\n${YELLOW}=== Memory Stats ===${NC}\n"
    output+="MemTotal:     $(grep MemTotal /proc/meminfo | awk '{printf "%.1f GB", $2/1024/1024}')\n"
    output+="MemFree:      $(grep MemFree /proc/meminfo | awk '{printf "%.1f GB", $2/1024/1024}')\n"
    output+="MemAvailable: $(grep MemAvailable /proc/meminfo | awk '{printf "%.1f GB", $2/1024/1024}')\n"
    output+="Shmem:        $(grep "^Shmem:" /proc/meminfo | awk '{printf "%.1f GB", $2/1024/1024}')\n"
    output+="Buffers:      $(grep Buffers /proc/meminfo | awk '{printf "%.1f GB", $2/1024/1024}')\n"
    output+="Cached:       $(grep "^Cached:" /proc/meminfo | awk '{printf "%.1f GB", $2/1024/1024}')\n"

    # Add time-series plots
    output+="\n${CYAN}=== Shared Memory Trend (last ${#shmem_history[@]}s) ===${NC}\n"
    output+="$(draw_plot shmem_history "Shmem (total)   " "$YELLOW" "$shmem_max")\n"
    output+="$(draw_plot devshm_history "/dev/shm (files)" "$GREEN" "$devshm_max")\n"
    output+="Each character = 1 second | Scale adapts slowly to avoid jitter\n"

    output+="\n${BLUE}Last update: $(date '+%H:%M:%S')${NC} | Press Ctrl+C to exit\n"

    # Now clear and display everything at once (atomic operation)
    clear
    echo -e "$output"

    sleep 1
done
