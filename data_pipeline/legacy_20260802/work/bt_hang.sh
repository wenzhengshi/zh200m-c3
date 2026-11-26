#!/usr/bin/env bash
which strace gdb 2>&1
PID=$(pgrep -f "llama-cli -m" | head -1)
echo "PID=$PID"
if [ -n "$PID" ]; then
  gdb -batch -ex "thread apply all bt 15" -p "$PID" 2>&1 | head -70
fi
