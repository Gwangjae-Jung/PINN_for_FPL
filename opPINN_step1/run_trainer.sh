# batch_size=300
for SEED in 0
do
    echo "Starting Jobs for SEED: $SEED"
    
    for gamma in -3.0 -2.0 -1.0 0.0 1.0
    do
        # Train the operators D and F
        nohup python run_trainer.py \
            --seed $SEED \
            --cuda_index 2 \
            --dimension 2 \
            --sample_v 64 \
            --gamma $gamma \
            --op D \
            > output__2d__gamma${gamma%1f}__op_D__seed${SEED}.log &
        nohup python run_trainer.py \
            --seed $SEED \
            --cuda_index 3 \
            --dimension 2 \
            --sample_v 64 \
            --gamma $gamma \
            --op F \
            > output__2d__gamma${gamma%1f}__op_F__seed${SEED}.log &
        wait
    done
done

echo "All jobs submitted!"