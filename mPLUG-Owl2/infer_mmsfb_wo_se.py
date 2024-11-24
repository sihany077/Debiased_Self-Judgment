import argparse
import os, json
from tqdm import tqdm
import torch
from transformers import set_seed
from PIL import Image
from transformers import TextStreamer

from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from mplug_owl2.conversation import conv_templates, SeparatorStyle
from mplug_owl2.model.builder import load_pretrained_model
from mplug_owl2.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
import copy
import warnings

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn
import numpy as np

from transformers.generation.logits_process import (
    LogitsProcessorList,
)
from transformers.generation.stopping_criteria import (
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
import transformers
from transformers.generation.utils import SampleOutput
from transformers.generation import SampleEncoderDecoderOutput, SampleDecoderOnlyOutput
# from experiments.llava.constants import IMAGE_TOKEN_INDEX

def eval_model(args):
    model_path = 'MAGAer13/mplug-owl2-llama2-7b'
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, load_8bit=False, load_4bit=False, device="cuda")
    #NOTE attention mask and pad_token_id
    if not hasattr(tokenizer, 'pad_token_id'):
        tokenizer.pad_token_id = tokenizer.eos_token_id

    input_file = args.input_file
    image_dir = args.image_dir
    output_file = args.output_file
    output_directory = os.path.dirname(output_file)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    with open(input_file, 'r') as file:
        data = json.load(file)
    with open(output_file, 'w') as output_file:
        output_file.write("{\n")  # 开始写入JSON对象

        total_items = len(data)
        for index, (key, value) in enumerate(tqdm(data.items(), desc="Processing")):

            image_file_path = os.path.join(image_dir, f"{key}.jpg")
            
            qs = value.get('Rephrased Question', '')

            image = Image.open(image_file_path).convert('RGB')
            max_edge = max(image.size) # We recommand you to resize to squared image for BEST performance.
            image = image.resize((max_edge, max_edge))

            image_tensor = process_images([image], image_processor)
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)

            conv = conv_templates["mplug_owl2"].copy()

            inp = DEFAULT_IMAGE_TOKEN + qs
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
            stop_str = conv.sep2
            # import ipdb;ipdb.set_trace()
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=False,
                    max_new_tokens=512,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria]
                    )

            output = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
            # import ipdb;ipdb.set_trace()

            value["ans"] = {
                "model1": {
                    "text": output
                }
            }

            
            output_file.write(f'"{key}": ')
            json.dump(value, output_file, indent=4)
            
            if index < total_items - 1:
                output_file.write(",\n")
            else:
                output_file.write("\n")  

        output_file.write("}\n")  
                    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="")
    parser.add_argument("--image_dir", type=str, default="")
    parser.add_argument("--output_file", type=str, default="")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--cd_alpha", type=float, default=1)
    parser.add_argument("--cd_beta", type=float, default=0.1)
    parser.add_argument("--use_dd", action='store_true', default=False)
    parser.add_argument("--use_dd_unk", action='store_true', default=False)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    args.temperature = 0.0
    args.do_sample = False

    eval_model(args)


''' 
srun -p compute    --gres=gpu:1   --time=00:50:00 --pty bash

conda activate /data/public_models/eval/env/mplug_owl2;cd /data/c/mPLUG-Owl/mPLUG-Owl2

python /data/c/mPLUG-Owl/mPLUG-Owl2/infer_mmsfb_wo_se.py \
--input_file /data/c/benchmarks/MM-SafetyBench/data/processed_questions/01-Illegal_Activitiy.json \
--image_dir /data/private_models/dpo_models/ysh/image/mmsafetybench/MM-SafetyBench_imgs/01-Illegal_Activitiy/SD_TYPO \
--output_file /data/c/LLaVA-Align/experiments/eval/sampling/json/mmsfb_mplug_wo_defence/01-Illegal_Activitiy.json

01-Illegal_Activitiy 02-HateSpeech 03-Malware_Generation 06-Fraud 07-Sex 09-Privacy_Violence

'''