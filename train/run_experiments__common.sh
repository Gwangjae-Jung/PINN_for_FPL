#!/bin/bash

# ==============================================================================
# Common Experiment Runner for FPL PINN / opPINN
# ==============================================================================

# Determine project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DRY_RUN=false

# Parse arguments and environment variables
while [[ $# -gt 0 ]]; do
    case "$1" in
        --target)
            TARGET="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --cuda|--cuda_index)
            CUDA_INDEX="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift 1
            ;;
        -*)
            echo "Unknown option: $1"
            exit 1
            ;;
        *)
            if [ -z "$TARGET" ]; then
                TARGET="$1"
            elif [ -z "$MODEL" ]; then
                MODEL="$1"
            elif [ -z "$CUDA_INDEX" ]; then
                CUDA_INDEX="$1"
            fi
            shift
            ;;
    esac
done

# Validate required variables
if [ -z "$TARGET" ] || [ -z "$MODEL" ]; then
    echo "Usage: $0 [TARGET] [MODEL] [CUDA_INDEX] [--dry-run]"
    echo "       $0 --target [TARGET] --model [MODEL] [--cuda CUDA_INDEX] [--dry-run]"
    echo ""
    echo "Available targets: bimaxwellian, bkw, maxwellian"
    echo "Available models:  oppinn, specpinn"
    exit 1
fi

# Normalize model name and configure model-specific parameters
NORMALIZED_MODEL=$(echo "$MODEL" | tr '[:upper:]' '[:lower:]')

case "$NORMALIZED_MODEL" in
    oppinn)
        MODEL_NAME="opPINN"
        MODEL_TAG="oppinn"
        EXTRA_ARGS="--surrogate"
        DEFAULT_CUDA=1
        ;;
    specpinn|pinn)
        MODEL_NAME="specPINN"
        MODEL_TAG="pinn"
        EXTRA_ARGS=""
        DEFAULT_CUDA=0
        ;;
    *)
        echo "Error: Unknown model '$MODEL'. Available options: oppinn, specpinn"
        exit 1
        ;;
esac

CUDA_INDEX="${CUDA_INDEX:-$DEFAULT_CUDA}"

# Normalize target and execute experiment
NORMALIZED_TARGET=$(echo "$TARGET" | tr '[:upper:]' '[:lower:]')
SEEDS=(${SEEDS:-0 1 2 3 4})

case "$NORMALIZED_TARGET" in
    bimaxwellian)
        GAMMAS=(${GAMMAS:--3.0 -2.0 -1.0 0.0 1.0})
        for SEED in "${SEEDS[@]}"; do
            for gamma in "${GAMMAS[@]}"; do
                current_work=$(printf "Training %s for seed %d and gamma %.1f" "$MODEL_NAME" "$SEED" "$gamma")
                path_config=$(printf "../config/config__2d__gamma%.1f__bimaxwellian.yaml" "$gamma")
                output_log=$(printf "output__%s2d__gamma%.1f__seed%d.log" "$MODEL_TAG" "$gamma" "$SEED")

                echo "$path_config"
                echo "$current_work"
                if [ "$DRY_RUN" = true ]; then
                    echo "[DRY-RUN] nohup python run_train.py --cuda_index $CUDA_INDEX --dim 2 --seed $SEED --path_config $path_config $EXTRA_ARGS > $output_log 2>&1 &"
                else
                    nohup python run_train.py \
                        --cuda_index "$CUDA_INDEX" \
                        --dim 2 \
                        --seed "$SEED" \
                        --path_config "$path_config" \
                        $EXTRA_ARGS > "$output_log" 2>&1 &
                    wait
                fi
            done
        done
        ;;
    bkw)
        gamma="0.0"
        for SEED in "${SEEDS[@]}"; do
            current_work=$(printf "Training %s for seed %d and gamma %.1f" "$MODEL_NAME" "$SEED" "$gamma")
            path_config=$(printf "../config/config__2d__gamma%.1f__bkw.yaml" "$gamma")
            output_log=$(printf "output__%s2d__bkw__seed%d.log" "$MODEL_TAG" "$SEED")

            echo "$path_config"
            echo "$current_work"
            if [ "$DRY_RUN" = true ]; then
                echo "[DRY-RUN] nohup python run_train.py --cuda_index $CUDA_INDEX --dim 2 --seed $SEED --path_config $path_config $EXTRA_ARGS > $output_log 2>&1 &"
            else
                nohup python run_train.py \
                    --cuda_index "$CUDA_INDEX" \
                    --dim 2 \
                    --seed "$SEED" \
                    --path_config "$path_config" \
                    $EXTRA_ARGS > "$output_log" 2>&1 &
                wait
            fi
        done
        ;;
    maxwellian)
        gamma="0.0"
        for SEED in "${SEEDS[@]}"; do
            current_work=$(printf "Training %s for seed %d and gamma %.1f" "$MODEL_NAME" "$SEED" "$gamma")
            path_config=$(printf "../config/config__2d__gamma%.1f__maxwellian.yaml" "$gamma")
            output_log=$(printf "output__%s2d__maxwellian__seed%d.log" "$MODEL_TAG" "$SEED")

            echo "$path_config"
            echo "$current_work"
            if [ "$DRY_RUN" = true ]; then
                echo "[DRY-RUN] nohup python run_train.py --cuda_index $CUDA_INDEX --dim 2 --seed $SEED --path_config $path_config $EXTRA_ARGS > $output_log 2>&1 &"
            else
                nohup python run_train.py \
                    --cuda_index "$CUDA_INDEX" \
                    --dim 2 \
                    --seed "$SEED" \
                    --path_config "$path_config" \
                    $EXTRA_ARGS > "$output_log" 2>&1 &
                wait
            fi
        done
        ;;
    *)
        echo "Error: Unknown target '$TARGET'. Available options: bimaxwellian, bkw, maxwellian"
        exit 1
        ;;
esac

echo "All jobs submitted!"
