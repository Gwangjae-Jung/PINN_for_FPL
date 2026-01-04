for gamma in -3.0 -2.0 -1.0 0.0 1.0
do
    echo "Generating a dataset for gamma: $gamma"
    # Train the operator D and F
    nohup python generate_dataset.py --gamma $gamma > data_generation__2d.log &
    wait
done

echo "All jobs submitted!"