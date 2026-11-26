#!/usr/bin/env bash
echo "--- llama-cli processes ---"
ps -o pid,stat,%cpu,etime,wchan:28,cmd -C llama-cli 2>/dev/null | head -5
PID=$(pgrep -f "llama-cli -m" | head -1)
if [ -n "$PID" ]; then
  echo "--- /proc/$PID/stack ---"
  cat /proc/$PID/stack 2>/dev/null | head -15
  echo "--- threads ---"
  for t in /proc/$PID/task/*; do
    echo "$t: $(cat $t/comm 2>/dev/null) $(cat $t/wchan 2>/dev/null)"
  done
fi
echo "--- output file size ---"
wc -c /tmp/llama_local.txt 2>/dev/null
echo "--- output head ---"
head -30 /tmp/llama_local.txt 2>/dev/null
