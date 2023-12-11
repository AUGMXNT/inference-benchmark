max=0
while true; do
    # Get total memory usage across all GPUs
    total_usage=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{sum += $1} END {print sum}')

    # Update max if current total_usage is greater
    if [ "$total_usage" -gt "$max" ]; then
        max=$total_usage
    fi

    # Print the current maximum
    echo "Max GPU Memory Used: $max MiB"
    
    # Wait for 1 second
    sleep 1
done
