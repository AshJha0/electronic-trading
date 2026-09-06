#!/bin/bash
# Run the etrade demo (build first with: bash build.sh).
set -euo pipefail
cd "$(dirname "$0")"
java -cp out/main com.quant.etrade.Demo
