#!/usr/bin/env python3
"""Track memory usage over time to detect leaks."""

import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class MemorySnapshot:
    """Snapshot of memory at a point in time."""
    timestamp: float
    # System memory (KB)
    mem_total: int
    mem_free: int
    mem_available: int
    shmem: int
    anon_pages: int  # Anonymous memory (heap, stack, etc.)
    mapped: int
    cached: int
    # Per-process memory (KB)
    process_rss: Dict[int, int]  # pid -> RSS
    process_vms: Dict[int, int]  # pid -> VMS


def read_meminfo() -> Dict[str, int]:
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


def get_process_memory(pid: int) -> tuple[int, int]:
    """Get RSS and VMS for a process in KB."""
    try:
        status_path = Path(f"/proc/{pid}/status")
        status = status_path.read_text()

        rss = 0
        vms = 0
        for line in status.split('\n'):
            if line.startswith('VmRSS:'):
                rss = int(line.split()[1])
            elif line.startswith('VmSize:'):
                vms = int(line.split()[1])

        return rss, vms
    except:
        return 0, 0


def find_forge_processes() -> List[int]:
    """Find all forge-related process PIDs."""
    uid = os.getuid()
    pids = []

    for pid_dir in Path('/proc').iterdir():
        if not pid_dir.name.isdigit():
            continue

        pid = int(pid_dir.name)
        try:
            if pid_dir.stat().st_uid != uid:
                continue

            cmdline_path = pid_dir / 'cmdline'
            cmdline = cmdline_path.read_text()

            if 'forge' in cmdline or 'monarch' in cmdline:
                pids.append(pid)
        except:
            continue

    return pids


def take_snapshot() -> MemorySnapshot:
    """Take a snapshot of current memory state."""
    meminfo = read_meminfo()
    pids = find_forge_processes()

    process_rss = {}
    process_vms = {}

    for pid in pids:
        rss, vms = get_process_memory(pid)
        if rss > 0:
            process_rss[pid] = rss
            process_vms[pid] = vms

    return MemorySnapshot(
        timestamp=time.time(),
        mem_total=meminfo['MemTotal'],
        mem_free=meminfo['MemFree'],
        mem_available=meminfo['MemAvailable'],
        shmem=meminfo['Shmem'],
        anon_pages=meminfo['AnonPages'],
        mapped=meminfo['Mapped'],
        cached=meminfo.get('Cached', 0),
        process_rss=process_rss,
        process_vms=process_vms
    )


def format_size(kb: int) -> str:
    """Format KB to human readable."""
    if kb >= 1024 * 1024:
        return f"{kb / (1024 * 1024):.2f} GB"
    elif kb >= 1024:
        return f"{kb / 1024:.2f} MB"
    else:
        return f"{kb} KB"


def format_delta(delta_kb: int) -> str:
    """Format delta with +/- sign."""
    sign = "+" if delta_kb >= 0 else ""
    return f"{sign}{format_size(abs(delta_kb))}"


def compare_snapshots(old: MemorySnapshot, new: MemorySnapshot):
    """Compare two snapshots and print differences."""
    elapsed = new.timestamp - old.timestamp

    print(f"\n{'='*80}")
    print(f"Memory Change Over {elapsed:.1f}s")
    print(f"{'='*80}")

    # System memory changes
    print("\nSystem Memory:")
    print(f"  MemFree:      {format_size(old.mem_free):>12} → {format_size(new.mem_free):>12}  ({format_delta(new.mem_free - old.mem_free)})")
    print(f"  MemAvailable: {format_size(old.mem_available):>12} → {format_size(new.mem_available):>12}  ({format_delta(new.mem_available - old.mem_available)})")
    print(f"  Shmem:        {format_size(old.shmem):>12} → {format_size(new.shmem):>12}  ({format_delta(new.shmem - old.shmem)})")
    print(f"  AnonPages:    {format_size(old.anon_pages):>12} → {format_size(new.anon_pages):>12}  ({format_delta(new.anon_pages - old.anon_pages)})")
    print(f"  Mapped:       {format_size(old.mapped):>12} → {format_size(new.mapped):>12}  ({format_delta(new.mapped - old.mapped)})")
    print(f"  Cached:       {format_size(old.cached):>12} → {format_size(new.cached):>12}  ({format_delta(new.cached - old.cached)})")

    # Process memory changes
    all_pids = set(old.process_rss.keys()) | set(new.process_rss.keys())

    # Calculate total process memory
    old_total_rss = sum(old.process_rss.values())
    new_total_rss = sum(new.process_rss.values())
    old_total_vms = sum(old.process_vms.values())
    new_total_vms = sum(new.process_vms.values())

    print(f"\nForge Processes Total:")
    print(f"  RSS (resident): {format_size(old_total_rss):>12} → {format_size(new_total_rss):>12}  ({format_delta(new_total_rss - old_total_rss)})")
    print(f"  VMS (virtual):  {format_size(old_total_vms):>12} → {format_size(new_total_vms):>12}  ({format_delta(new_total_vms - old_total_vms)})")

    # Show top growing processes
    growth = []
    for pid in all_pids:
        old_rss = old.process_rss.get(pid, 0)
        new_rss = new.process_rss.get(pid, 0)
        delta = new_rss - old_rss
        if abs(delta) > 1024:  # Only show > 1MB changes
            growth.append((pid, old_rss, new_rss, delta))

    if growth:
        growth.sort(key=lambda x: x[3], reverse=True)
        print(f"\nTop Process Memory Changes (> 1 MB):")
        print(f"  {'PID':<8} {'Old RSS':>12} {'New RSS':>12} {'Change':>12}")
        print(f"  {'-'*48}")
        for pid, old_rss, new_rss, delta in growth[:10]:
            print(f"  {pid:<8} {format_size(old_rss):>12} {format_size(new_rss):>12} {format_delta(delta):>12}")

    # Leak indicators
    print(f"\n{'='*80}")
    print("Leak Indicators:")
    print(f"{'='*80}")

    indicators = []

    if new_total_rss > old_total_rss + 10*1024:  # > 10MB growth
        indicators.append(f"⚠ Process RSS growing: {format_delta(new_total_rss - old_total_rss)}")

    if new.anon_pages > old.anon_pages + 10*1024:
        indicators.append(f"⚠ Anonymous memory growing: {format_delta(new.anon_pages - old.anon_pages)}")

    if new.shmem > old.shmem + 10*1024:
        indicators.append(f"⚠ Shared memory growing: {format_delta(new.shmem - old.shmem)}")

    if new.mapped > old.mapped + 10*1024:
        indicators.append(f"⚠ Memory-mapped files growing: {format_delta(new.mapped - old.mapped)}")

    if indicators:
        for indicator in indicators:
            print(f"  {indicator}")
    else:
        print("  ✓ No significant memory growth detected")

    print()


def main():
    """Main monitoring loop."""
    if len(sys.argv) < 2:
        print("Usage: ./track_memory_leak.py <interval_seconds> [duration_seconds]")
        print("Example: ./track_memory_leak.py 10 300  # Monitor every 10s for 5 minutes")
        sys.exit(1)

    interval = int(sys.argv[1])
    duration = int(sys.argv[2]) if len(sys.argv) > 2 else None

    print("Taking initial snapshot...")
    snapshots = [take_snapshot()]

    start_time = time.time()

    try:
        while True:
            time.sleep(interval)

            print(f"\nTaking snapshot at {time.strftime('%H:%M:%S')}...")
            snapshot = take_snapshot()
            snapshots.append(snapshot)

            # Compare with previous snapshot
            if len(snapshots) >= 2:
                compare_snapshots(snapshots[-2], snapshots[-1])

            # Also compare with initial snapshot if we have enough history
            if len(snapshots) > 3:
                print(f"\n{'='*80}")
                print(f"Total Change Since Start ({(snapshot.timestamp - snapshots[0].timestamp):.1f}s ago)")
                print(f"{'='*80}")
                compare_snapshots(snapshots[0], snapshots[-1])

            # Check if we should stop
            if duration and (time.time() - start_time) >= duration:
                print("\nMonitoring duration reached. Exiting.")
                break

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")
        if len(snapshots) >= 2:
            print("\nFinal summary:")
            compare_snapshots(snapshots[0], snapshots[-1])


if __name__ == "__main__":
    main()
