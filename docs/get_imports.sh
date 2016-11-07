#! /usr/bin/env bash

w="a-zA-Z0-9_\."
src="../bb_tracking"
from=$(find $src -name "*.py" | xargs cat  |  sed -n "s/^from \([$w]*\) .*$/\1/p")
import=$(find $src -name "*.py" | xargs cat  | sed -n "s/^import \([$w]*\).*/\1/p")

echo "["
for m in $from $import; do
    echo "    \"$m\","
done
# hack to fix json
echo "    \"last item must have no comma\""
echo "]"
