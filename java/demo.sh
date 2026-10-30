#!/usr/bin/env bash
# Run the etrade demo with Maven.
set -euo pipefail
cd "$(dirname "$0")"
mvn -q -B compile exec:java
