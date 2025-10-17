#!/usr/bin/env python3
"""Track kernel memory categories to find the leak."""

import sys
import time

def read_meminfo():
    """Read /proc/meminfo and return as dict."""
    meminfo = {}
    with open('/proc/meminfo', 'r') as f:
        for line in f:
            if ':' in line:
                key, value = line.split(':', 1)
                # Extract number (KB)
                kb = int(value.strip().split()[0])
                meminfo[key] = kb
    return meminfo

def format_gb(kb):
    """Format KB to GB."""
    return f"{kb / (1024 * 1024):.2f} GB"

def main():
    if len(sys.argv) < 2:
        print("Usage: ./track_kernel_memory.py <interval_seconds>")
        sys.exit(1)

    interval = int(sys.argv[1])

    print("Taking initial snapshot...")
    initial = read_meminfo()

    # Key categories to watch
    categories = [
        'MemTotal',
        'MemFree',
        'MemAvailable',
        'Buffers',
        'Cached',
        'AnonPages',
        'Shmem',
        'Mapped',
        'Active(anon)',
        'Inactive(anon)',
        'Active(file)',
        'Inactive(file)',
        'Slab',
        'KernelStack',
        'PageTables',
        'VmallocUsed',
    ]

    try:
        while True:
            time.sleep(interval)
            current = read_meminfo()

            print(f"\n{'='*100}")
            print(f"Time: {time.strftime('%H:%M:%S')}")
            print(f"{'='*100}")
            print(f"{'Category':<20} {'Current':>15} {'Change':>15} {'% of Total':>15}")
            print(f"{'-'*100}")

            for cat in categories:
                if cat in current:
                    curr_val = current[cat]
                    init_val = initial.get(cat, 0)
                    delta = curr_val - init_val
                    pct = (curr_val / current['MemTotal']) * 100

                    delta_str = f"+{format_gb(delta)}" if delta >= 0 else f"{format_gb(delta)}"

                    # Highlight growing categories
                    marker = "⚠️ " if abs(delta) > 10*1024*1024 else "  "  # > 10 GB change

                    print(f"{marker}{cat:<18} {format_gb(curr_val):>15} {delta_str:>15} {pct:>14.1f}%")

            # Calculate "used" memory (total - available)
            used = current['MemTotal'] - current['MemAvailable']
            init_used = initial['MemTotal'] - initial['MemAvailable']
            used_delta = used - init_used

            print(f"\n{'Total Used (Total - Available)':<20} {format_gb(used):>15} {'+' if used_delta >= 0 else ''}{format_gb(used_delta):>14}")

    except KeyboardInterrupt:
        print("\n\nStopped.")

if __name__ == "__main__":
    main()
