#!/usr/bin/env bash
/home/xmkjsoft/llama.cpp-src/build/bin/llama-bench --help 2>&1 | grep -E "^\s+-[mpntcr]" | head -20
