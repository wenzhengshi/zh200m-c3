#!/usr/bin/env bash
set -euo pipefail
D="$1"
cd "$D"
find . -type f ! -name SHA256SUMS.txt -printf '%P\n' | sort | xargs -r sha256sum > SHA256SUMS.txt
sha256sum -c SHA256SUMS.txt
echo "entries: $(wc -l < SHA256SUMS.txt)"
