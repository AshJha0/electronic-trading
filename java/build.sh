#!/usr/bin/env bash
# Build the etrade Java library and tests with Maven (any OS with JDK 25).
set -euo pipefail
cd "$(dirname "$0")"
mvn -q -B -DskipTests package
echo "build OK: target/etrade-*.jar"
