CUDA_INDEX__PINN=2
CUDA_INDEX__OPPINN=3

for SEED in 0 1 2 3 4
do
    gamma=0.0
    current_work__pinn=$(printf "Training PINN for seed %d and gamma %.1f" "$SEED" "$gamma")
    current_work__oppinn=$(printf "Training opPINN for seed %d and gamma %.1f" "$SEED" "$gamma")
    path_config=$(printf "config/config__2d__gamma%.1f__bkw.yaml" "$gamma")
    output_pinn=$(printf "output__pinn2d__bkw__seed%d.log" "$SEED")
    output_oppinn=$(printf "output__oppinn2d__bkw__seed%d.log" "$SEED")
    echo $path_config
    echo $current_work__pinn
    nohup python experiment.py \
        --cuda_index $CUDA_INDEX__OPPINN \
        --dim 2 \
        --seed $SEED \
        --path_config $path_config &
    echo $current_work__oppinn
    nohup python experiment.py \
        --cuda_index $CUDA_INDEX__PINN \
        --dim 2 \
        --seed $SEED \
        --path_config $path_config \
        --surrogate &
    wait
    # Wait until both models are trained
done

echo "All jobs submitted!"