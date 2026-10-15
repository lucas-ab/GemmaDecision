#!/bin/zsh
# Train the given modes sequentially (default: frozen lora full), then compare all finished runs.
set -e
cd "$(dirname "$0")/.."
mkdir -p runs
modes=(${@:-frozen lora full})
for m in $modes; do
  uv run gd-train configs/$m.yaml > runs/$m.out 2>&1
done
done_runs=(runs/*/eval_test.json(N:h))
uv run gd-compare $done_runs > runs/compare.md
uv run gd-compare $done_runs --slice type=score >> runs/compare.md
