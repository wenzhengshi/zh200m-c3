#!/usr/bin/env bash
set -e
mkdir -p /home/xmkjsoft/llamacpp
cd /home/xmkjsoft/llamacpp
if [ ! -f llama-b10223-bin-ubuntu-x64.tar.gz ]; then
  curl -sL -o llama-b10223-bin-ubuntu-x64.tar.gz \
    https://github.com/ggml-org/llama.cpp/releases/download/b10223/llama-b10223-bin-ubuntu-x64.tar.gz
fi
tar -xzf llama-b10223-bin-ubuntu-x64.tar.gz
ls -la
