#!/bin/bash
# Force the kernel to reclaim inactive memory

echo "Before reclaim:"
free -h | grep Mem

echo ""
echo "Dropping caches and reclaiming inactive pages..."
# This requires sudo/root
sync  # Flush any pending writes
sudo sh -c 'echo 1 > /proc/sys/vm/drop_caches'  # Drop page cache
sudo sh -c 'echo 2 > /proc/sys/vm/drop_caches'  # Drop dentries and inodes
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'  # Drop everything

echo ""
echo "After reclaim:"
free -h | grep Mem

echo ""
echo "Inactive(anon) pages:"
grep "Inactive(anon)" /proc/meminfo
