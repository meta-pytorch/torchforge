#!/usr/bin/env python3
"""Show shared memory mappings for processes."""

import os
import sys
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Set

# === Constants ===
RED = '\033[0;31m'
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
NC = '\033[0m'  # No Color

TABLE_WIDTH = 92
SEPARATOR = '=' * 80


# === Data Structures ===

@dataclass
class ShmMapping:
    """Represents a single shared memory mapping."""
    addr_range: str
    perms: str
    path: str
    size: int  # KB
    rss: int   # KB
    pss: int   # KB
    shared_clean: int  # KB
    shared_dirty: int  # KB

    @property
    def is_leaked(self) -> bool:
        """Check if this mapping is for a deleted (leaked) file."""
        return '(deleted)' in self.path


@dataclass
class PathAggregation:
    """Aggregated statistics for a shared memory path."""
    path: str
    size: int  # KB
    rss: int   # KB
    pids: Set[int]

    @property
    def is_leaked(self) -> bool:
        return '(deleted)' in self.path


# === Core Functions ===

def get_process_cmdline(pid: int) -> str:
    """Get process command line."""
    try:
        cmdline_path = Path(f"/proc/{pid}/cmdline")
        cmdline = cmdline_path.read_text().replace('\x00', ' ').strip()
        return cmdline or 'unknown'
    except:
        return 'unknown'


def parse_smaps(pid: int) -> List[ShmMapping]:
    """Parse /proc/<pid>/smaps to get /dev/shm memory mappings."""
    smaps_path = Path(f"/proc/{pid}/smaps")
    if not smaps_path.exists():
        return []

    mappings = []
    current = None

    try:
        with open(smaps_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Start of a new mapping (contains address range)
                if '-' in line and not line.startswith(' '):
                    # Save previous mapping if it was /dev/shm
                    if current and '/dev/shm' in current.path:
                        mappings.append(current)

                    # Parse new mapping header
                    parts = line.split()
                    path = ' '.join(parts[5:]) if len(parts) > 5 else '[anonymous]'
                    current = ShmMapping(
                        addr_range=parts[0],
                        perms=parts[1],
                        path=path,
                        size=0, rss=0, pss=0,
                        shared_clean=0, shared_dirty=0
                    )
                elif ':' in line and current:
                    # Parse memory stats
                    key, value = line.split(':', 1)
                    value = value.strip()
                    if 'kB' in value:
                        kb = int(value.split()[0])
                        if key == 'Size':
                            current.size = kb
                        elif key == 'Rss':
                            current.rss = kb
                        elif key == 'Pss':
                            current.pss = kb
                        elif key == 'Shared_Clean':
                            current.shared_clean = kb
                        elif key == 'Shared_Dirty':
                            current.shared_dirty = kb

            # Don't forget last mapping
            if current and '/dev/shm' in current.path:
                mappings.append(current)

    except PermissionError:
        pass  # Silently skip processes we don't own
    except Exception as e:
        print(f"Error reading smaps for PID {pid}: {e}", file=sys.stderr)

    return mappings


def format_size(kb: int) -> str:
    """Format size in KB to human readable."""
    if kb >= 1024 * 1024:
        return f"{kb / (1024 * 1024):.2f} GB"
    elif kb >= 1024:
        return f"{kb / 1024:.2f} MB"
    else:
        return f"{kb} KB"


# === Process Discovery ===

def get_all_pids_with_uid(uid: int) -> Dict[int, str]:
    """Get all PIDs owned by the given UID with their command lines."""
    pids = {}
    for pid_dir in Path('/proc').iterdir():
        if not pid_dir.name.isdigit():
            continue

        pid = int(pid_dir.name)
        try:
            if pid_dir.stat().st_uid == uid:
                pids[pid] = get_process_cmdline(pid)
        except (FileNotFoundError, PermissionError):
            continue

    return pids


def build_process_tree(uid: int) -> Dict[int, List[int]]:
    """Build parent->children map for all processes owned by uid."""
    children_map = defaultdict(list)

    for pid_dir in Path('/proc').iterdir():
        if not pid_dir.name.isdigit():
            continue

        pid = int(pid_dir.name)
        try:
            if pid_dir.stat().st_uid != uid:
                continue

            status_path = pid_dir / 'status'
            status = status_path.read_text()
            for line in status.split('\n'):
                if line.startswith('PPid:'):
                    ppid = int(line.split()[1])
                    children_map[ppid].append(pid)
                    break
        except (FileNotFoundError, PermissionError):
            continue

    return children_map


def get_descendants(root_pids: List[int], children_map: Dict[int, List[int]]) -> Set[int]:
    """Get all descendant PIDs using BFS."""
    descendants = set()
    queue = list(root_pids)

    while queue:
        current = queue.pop(0)
        if current in descendants:
            continue
        descendants.add(current)

        if current in children_map:
            queue.extend(children_map[current])

    return descendants


def find_training_processes() -> List[tuple[int, str]]:
    """Find forge training processes and all their descendants."""
    uid = os.getuid()
    all_pids = get_all_pids_with_uid(uid)

    # Find root forge processes
    root_pids = []
    for pid, cmdline in all_pids.items():
        if any(pattern in cmdline for pattern in [
            'apps/grpo/main', 'apps/sft/main',
            'apps.grpo.main', 'apps.sft.main'
        ]):
            root_pids.append(pid)

    # Fallback: look for any forge/ in command
    if not root_pids:
        root_pids = [
            pid for pid, cmdline in all_pids.items()
            if 'forge/' in cmdline and 'python' in cmdline
        ]

    # Get all descendants
    if root_pids:
        children_map = build_process_tree(uid)
        descendant_pids = get_descendants(root_pids, children_map)
        return [(pid, all_pids.get(pid, 'unknown')) for pid in descendant_pids if pid in all_pids]

    return []


# === Aggregation ===

def group_mappings_by_path(mappings: List[ShmMapping]) -> Dict[str, PathAggregation]:
    """Group mappings by path and aggregate statistics."""
    aggregated = defaultdict(lambda: PathAggregation(path='', size=0, rss=0, pids=set()))

    for m in mappings:
        agg = aggregated[m.path]
        agg.path = m.path
        agg.size += m.size
        agg.rss += m.rss

    return dict(aggregated)


def aggregate_across_processes(
    process_mappings: List[tuple[int, List[ShmMapping]]]
) -> Dict[str, PathAggregation]:
    """Aggregate mappings across multiple processes."""
    aggregated = defaultdict(lambda: PathAggregation(path='', size=0, rss=0, pids=set()))

    for pid, mappings in process_mappings:
        for m in mappings:
            agg = aggregated[m.path]
            agg.path = m.path
            agg.size += m.size
            agg.rss += m.rss
            agg.pids.add(pid)

    return dict(aggregated)


# === Display Functions ===

def print_mapping_table(
    aggregated: Dict[str, PathAggregation],
    title: str,
    is_leaked: bool = False
):
    """Print a formatted table of mappings."""
    if not aggregated:
        return

    color = RED if is_leaked else ''
    reset = NC if is_leaked else ''

    print(f"\n{title:<60} {'Count':>6} {'Size':>12} {'RSS':>12}")
    print('-' * TABLE_WIDTH)

    for path in sorted(aggregated.keys()):
        agg = aggregated[path]
        count = len(agg.pids) if agg.pids else 1
        print(f"{color}{path:<60} {count:>6} {format_size(agg.size):>12} {format_size(agg.rss):>12}{reset}")


def print_object_list(aggregated: Dict[str, PathAggregation], title: str, is_leaked: bool = False):
    """Print detailed list of objects with their PIDs."""
    if not aggregated:
        return

    color = RED if is_leaked else ''
    reset = NC if is_leaked else ''

    print(f"\n{color}{title}:{reset}")
    for path in sorted(aggregated.keys()):
        agg = aggregated[path]
        pids_str = ', '.join(str(p) for p in sorted(agg.pids)) if agg.pids else 'N/A'
        print(f"{color}  {path}{reset}")
        print(f"    Size: {format_size(agg.size)} | RSS: {format_size(agg.rss)} | PIDs: {pids_str}")


def print_summary_stats(
    active_count: int, active_size: int, active_rss: int,
    leaked_count: int, leaked_size: int, leaked_rss: int
):
    """Print summary statistics."""
    total_count = active_count + leaked_count
    total_size = active_size + leaked_size
    total_rss = active_rss + leaked_rss

    print(f"\nActive mappings:  {active_count:>6} | Size: {format_size(active_size):>12} | RSS: {format_size(active_rss):>12}")

    if leaked_count > 0:
        print(f"{RED}Leaked mappings:  {leaked_count:>6} | Size: {format_size(leaked_size):>12} | RSS: {format_size(leaked_rss):>12}{NC}")
    else:
        print(f"Leaked mappings:  {leaked_count:>6} | Size: {format_size(leaked_size):>12} | RSS: {format_size(leaked_rss):>12}")

    print(f"{'-'*80}")
    print(f"TOTAL:            {total_count:>6} | Size: {format_size(total_size):>12} | RSS: {format_size(total_rss):>12}")


# === Command Modes ===

def run_leaked_scan():
    """Scan all processes for leaked /dev/shm objects."""
    print("Scanning all processes for leaked /dev/shm objects...")

    uid = os.getuid()
    all_pids = get_all_pids_with_uid(uid)

    # Collect leaked mappings
    process_mappings = []
    for pid in all_pids:
        mappings = parse_smaps(pid)
        leaked = [m for m in mappings if m.is_leaked]
        if leaked:
            process_mappings.append((pid, leaked))

    if not process_mappings:
        print("\n✓ No leaked /dev/shm objects found!")
        return

    # Aggregate and display
    aggregated = aggregate_across_processes(process_mappings)
    print(f"\n⚠ Found {len(aggregated)} leaked /dev/shm objects:\n")

    total_size = sum(agg.size for agg in aggregated.values())
    total_rss = sum(agg.rss for agg in aggregated.values())

    for path in sorted(aggregated.keys()):
        agg = aggregated[path]
        print(f"{RED}{path}{NC}")
        print(f"  Total: {format_size(agg.size)} (RSS: {format_size(agg.rss)})")
        print(f"  Held by {len(agg.pids)} process(es):")
        for pid in sorted(agg.pids):
            cmdline = get_process_cmdline(pid)
            print(f"    PID {pid}: {cmdline[:80]}")
        print()

    print(f"\n{SEPARATOR}")
    print(f"TOTAL LEAKED: {format_size(total_size)} (RSS: {format_size(total_rss)})")
    print(f"{SEPARATOR}\n")


def run_pid_scan(pid: int):
    """Show shared memory mappings for a specific PID."""
    cmdline = get_process_cmdline(pid)
    if cmdline == 'unknown':
        print(f"Process {pid} not found or not accessible")
        return

    print(f"\n{SEPARATOR}")
    print(f"PID: {pid}")
    print(f"Command: {cmdline[:100]}{'...' if len(cmdline) > 100 else ''}")
    print(f"{SEPARATOR}")

    mappings = parse_smaps(pid)
    if not mappings:
        print("No /dev/shm mappings found")
        return

    # Group and separate leaked vs active
    aggregated = group_mappings_by_path(mappings)
    leaked = {k: v for k, v in aggregated.items() if v.is_leaked}
    active = {k: v for k, v in aggregated.items() if not v.is_leaked}

    print(f"\nShared Memory Mappings: {len(mappings)} total")

    print_mapping_table(leaked, "LEAKED (deleted but still open)", is_leaked=True)
    print_mapping_table(active, "Active mappings")

    # Print total
    total_size = sum(m.size for m in mappings)
    total_rss = sum(m.rss for m in mappings)
    print('-' * TABLE_WIDTH)
    print(f"{'TOTAL':<60} {len(mappings):>6} {format_size(total_size):>12} {format_size(total_rss):>12}")

    # Detailed view if not too many
    if len(mappings) <= 20:
        print(f"\nDetailed Mappings:")
        print(f"{'Address Range':<20} {'Perms':<6} {'Size':>12} {'RSS':>12} Path")
        print('-' * 100)
        for m in mappings:
            print(f"{m.addr_range:<20} {m.perms:<6} {format_size(m.size):>12} {format_size(m.rss):>12} {m.path}")


def run_training_scan():
    """Find and analyze training processes."""
    print("Searching for training processes with /dev/shm mappings...")

    processes = find_training_processes()
    if not processes:
        print("No training processes found")
        return

    # Collect mappings for each process
    process_mappings = []
    for pid, cmdline in processes:
        mappings = parse_smaps(pid)
        if mappings:
            process_mappings.append((pid, mappings))

    if not process_mappings:
        print("\nFound processes but none have /dev/shm mappings:")
        for pid, cmdline in processes[:10]:
            print(f"  PID {pid}: {cmdline[:80]}")
        return

    # Show per-process details
    for pid, mappings in process_mappings:
        cmdline = get_process_cmdline(pid)
        run_pid_scan(pid)

    # Print summary across all processes
    print(f"\n{SEPARATOR}")
    print("SUMMARY ACROSS ALL TRAINING PROCESSES")
    print(f"{SEPARATOR}")

    aggregated = aggregate_across_processes(process_mappings)
    active = {k: v for k, v in aggregated.items() if not v.is_leaked}
    leaked = {k: v for k, v in aggregated.items() if v.is_leaked}

    # Calculate stats
    all_mappings = [m for _, mappings in process_mappings for m in mappings]
    active_mappings = [m for m in all_mappings if not m.is_leaked]
    leaked_mappings = [m for m in all_mappings if m.is_leaked]

    active_size = sum(m.size for m in active_mappings)
    active_rss = sum(m.rss for m in active_mappings)
    leaked_size = sum(m.size for m in leaked_mappings)
    leaked_rss = sum(m.rss for m in leaked_mappings)

    print(f"\nTotal processes scanned: {len(processes)}")
    print_summary_stats(
        len(active_mappings), active_size, active_rss,
        len(leaked_mappings), leaked_size, leaked_rss
    )

    # List objects
    print_object_list(active, "ACTIVE OBJECTS")
    print_object_list(leaked, "LEAKED OBJECTS", is_leaked=True)
    print()


# === Main Entry Point ===

def main():
    """Main entry point - dispatch to appropriate command mode."""
    if len(sys.argv) > 1 and sys.argv[1] == '--leaked':
        run_leaked_scan()
    elif len(sys.argv) > 1:
        try:
            pid = int(sys.argv[1])
            run_pid_scan(pid)
        except ValueError:
            print(f"Invalid PID: {sys.argv[1]}")
            sys.exit(1)
    else:
        run_training_scan()


if __name__ == "__main__":
    main()
