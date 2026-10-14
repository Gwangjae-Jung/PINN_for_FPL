#!/bin/bash

# Configuration
TARGET="bkw"
MODEL="oppinn"

# Execute common experiment runner
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/run_experiments__common.sh" --target "$TARGET" --model "$MODEL" "$@"
