#!/usr/bin/env bash
# Build the etrade C++ library, demo and tests.
# Run tests afterwards with: ctest --test-dir build --output-on-failure
set -euo pipefail
cd "$(dirname "$0")"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
