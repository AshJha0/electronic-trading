#!/usr/bin/env bash
# Run the JUnit suite with Maven.
set -euo pipefail
cd "$(dirname "$0")"
mvn -B test
