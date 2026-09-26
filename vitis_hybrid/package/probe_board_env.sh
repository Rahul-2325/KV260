#!/bin/sh
# probe_board_env.sh -- what is available for the three open items:
# real power measurement, an accuracy dataset, and thread behaviour.
echo "=== hwmon sensors (real rail monitors on the SOM) ==="
for d in /sys/class/hwmon/hwmon*; do
    [ -d "$d" ] || continue
    echo "  $d : $(cat $d/name 2>/dev/null)"
done

echo
echo "=== power / current / voltage inputs ==="
ls /sys/class/hwmon/hwmon*/power*_input /sys/class/hwmon/hwmon*/curr*_input \
   /sys/class/hwmon/hwmon*/in*_input 2>/dev/null | head -40

echo
echo "=== labels for those rails ==="
for f in /sys/class/hwmon/hwmon*/*_label; do
    [ -f "$f" ] && echo "  $f = $(cat $f)"
done 2>/dev/null | head -40

echo
echo "=== xmutil platformstats ==="
xmutil platformstats 2>&1 | head -30

echo
echo "=== images available on the board ==="
ls /home/root/*.jpg /home/root/*.png 2>/dev/null | head
echo "  dataset dirs:"
ls -d /home/root/dataset /home/root/images /home/root/val* 2>/dev/null

echo
echo "=== cpu count (for pipelining) ==="
nproc
echo DONE
