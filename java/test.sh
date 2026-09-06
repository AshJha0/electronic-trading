#!/bin/bash
# Run all JUnit 4 test classes (build first with: bash build.sh).
set -euo pipefail
cd "$(dirname "$0")"

JUNIT=/usr/share/java/junit4.jar
HAMCREST=/usr/share/java/hamcrest.jar

CLASSES=$(find out/test -name '*Test.class' \
    | sed -e 's|^out/test/||' -e 's|\.class$||' -e 's|/|.|g' | sort)

java -cp "out/main:out/test:$JUNIT:$HAMCREST" org.junit.runner.JUnitCore $CLASSES
