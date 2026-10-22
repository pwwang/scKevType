#!/bin/bash
# On-node I/O probe for a hung/slow ask step. Run it inside the job's own allocation:
#   srun --jobid=<id> --overlap bash $W/logs/io-probe.sh
P=$(ps -eo pid,args | grep 'venv-train/bin/python' | grep 'kev_probe' | grep -v grep | awk '{print $1}' | head -1)
echo "PID=$P"
[ -n "$P" ] || { echo "no kev_probe process on this node"; exit 1; }
echo "--- io (first read)"
cat /proc/"$P"/io 2>&1 | head -4
echo "--- open files (last 6)"
ls -l /proc/"$P"/fd 2>/dev/null | tail -6
echo "--- state"
grep -E '^State|^VmRSS|^Threads' /proc/"$P"/status
sleep 30
echo "--- io (after 30s; read_bytes not advancing = blocked, advancing = just slow)"
cat /proc/"$P"/io 2>&1 | head -4
