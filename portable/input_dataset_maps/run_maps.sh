#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
CONFIG="${AFOLU_MAP_CONFIG:-${SCRIPT_DIR}/input_dataset_maps.config.json}"
OUTPUT_DIR="${AFOLU_MAP_OUTPUT_DIR:-${SCRIPT_DIR}/outputs/maps_${STAMP}}"
CACHE_DIR="${AFOLU_MAP_CACHE_DIR:-${SCRIPT_DIR}/cache}"

echo "Output directory: ${OUTPUT_DIR}"
python3 "${SCRIPT_DIR}/create_input_dataset_maps.py" \
  --config "${CONFIG}" \
  --output-dir "${OUTPUT_DIR}" \
  --cache-dir "${CACHE_DIR}" \
  "$@"

