gamma=0.0
echo "Generating a dataset for gamma: $gamma (Maxwellian molecules only)"
# Train the operator D and F
nohup python generate_dataset.py > data_generation__2d.log &
wait
echo "All jobs submitted!"