#!/bin/bash

activities=("01-Illegal_Activitiy" "02-HateSpeech" "03-Malware_Generation" "06-Fraud" "07-Sex" "09-Privacy_Violence")

for activity in "${activities[@]}"; do
    python /data/chenhang_cui/ysh/mPLUG-Owl/mPLUG-Owl2/infer_mmsfb_wo_se.py \
        --input_file /data/c/benchmarks/MM-SafetyBench/data/processed_questions/$activity.json \
        --image_dir /data/private_models/dpo_models/ysh/image/mmsafetybench/MM-SafetyBench_imgs/$activity/SD_TYPO \
        --output_file /data/c/LLaVA-Align/experiments/eval/sampling/json/mmsfb_mplug_wo_defence/$activity.json
done
