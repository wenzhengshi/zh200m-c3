#!/usr/bin/env bash
curl -s "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=40" \
  | grep -E '"tag_name"|browser_download_url' \
  | grep -B1 -E "ubuntu-x64" | head -40
