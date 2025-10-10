#!/bin/bash
# Run BioDiscoveryAgent with MY_CUSTOM_SCREEN
echo "🚀 Running BioDiscoveryAgent with MY_CUSTOM_SCREEN..."
python3 research_assistant.py \
    --task perturb-genes-brief \
    --model claude-3-5-sonnet-20240620 \
    --run_name my_custom_screen_run \
    --data_name MY_CUSTOM_SCREEN \
    --steps 3 \
    --num_genes 64 \
    --log_dir my_custom_screen_logs
echo "✅ Run completed! Check my_custom_screen_logs/ for results."
