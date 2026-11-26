#!/usr/bin/env bash
for TAG in b7434 b6811 b6099 b5000; do
  echo "### $TAG"
  curl -s "https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/$TAG" \
    | grep -E "browser_download_url.*ubuntu-x64.tar.gz" | head -2
done
