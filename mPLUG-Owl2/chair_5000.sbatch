#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=15:29:00
#SBATCH --partition=compute
#SBATCH --job-name=sgd_64
source activate /data/public_models/eval/env/ysh_mplug_owl2
# conda env 

cd /data/chenhang_cui/ysh/mPLUG-Owl/mPLUG-Owl2

python /data/chenhang_cui/ysh/mPLUG-Owl/mPLUG-Owl2/sgd_chair.py --max_new_token 64 --output_file /data/chenhang_cui/ysh/LLaVA-Align/experiments/eval/sampling/chair/mplug_owl2/dse_500_64_panal2_promptnoeos.jsonl