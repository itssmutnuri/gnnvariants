#!/bin/bash

# Function to capture timing information for each script and append it to times.txt
execute_and_time() {
    script_name=$1
    start_time=$(date +%s)  # Get the current time in seconds since the epoch
    python "$script_name"  # Execute the script
    end_time=$(date +%s)    # Get the current time again after the script finishes

    # Compute the elapsed time
    elapsed_time=$((end_time - start_time))

    # Convert the elapsed time to minutes and seconds format
    minutes=$((elapsed_time / 60))
    seconds=$((elapsed_time % 60))

    # Append the timing info to times.txt
    echo "$script_name: $minutes minutes $seconds seconds" >> times.txt
}

# Initialize the times.txt with a header
echo "Script Execution Timings:" > times.txt

# List of scripts to run
scripts=("GNNScript.py" 
        "Ablations/wosT2_GNNScript.py" "Ablations/wosT1_GNNScript.py"
        "Ablations/t1_GNNScript.py" "Ablations/t3_GNNScript.py" "Ablations/t4_GNNScript.py" 
        "Ablations/enc0_GNNScript.py" "Ablations/enc1_GNNScript.py" "Ablations/enc2_GNNScript.py" "Ablations/enc3a_GNNScript.py")

# Execute each script in the list and capture its timing
for script in "${scripts[@]}"; do
    echo "Starting $script"
    execute_and_time "$script"
done

echo "All scripts executed."
