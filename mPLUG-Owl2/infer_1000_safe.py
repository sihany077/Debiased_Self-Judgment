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
model_path = 'MAGAer13/mplug-owl2-llama2-7b'
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, load_8bit=False, load_4bit=False, device="cuda")
#NOTE attention mask and pad_token_id
if not hasattr(tokenizer, 'pad_token_id'):
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Load the JSON file
input_file = '/data/chenhang_cui/ysh/LLaVA-Align/rlhfv_data/selected_items_1100_2.json'
output_file = '/data/chenhang_cui/ysh/LLaVA-Align/rlhfv_data/mplug_selected_items_1100_2.json'

with open(input_file, 'r') as file:
    data = json.load(file)

# Loop through each item in the JSON and process with InstructBlip
for item in tqdm(data):
    # Construct the image path
    image_path = "/data/public_models/ysh/image/rlhfv_data/" + item['image']
    
    # Load the image
    image = Image.open(image_path).convert("RGB").convert('RGB')
    max_edge = max(image.size) # We recommand you to resize to squared image for BEST performance.
    image = image.resize((max_edge, max_edge))

    image_tensor = process_images([image], image_processor)
    image_tensor = image_tensor.to(model.device, dtype=torch.float16)
    
    # Get the question (prompt)
    prompt = item['qs']
    
    # Process the image and prompt
    conv = conv_templates["mplug_owl2"].copy()

    inp = DEFAULT_IMAGE_TOKEN + prompt
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

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()

    # Update the item with the new generated answer
    item['answer'] = outputs

# Save the updated items back to a new JSON file
with open(output_file, 'w') as file:
    json.dump(data, file, indent=4)

print(f"Processed data has been saved to {output_file}")

'''
conda activate /data/public_models/eval/env/ysh_mplug_owl2;cd /data/chenhang_cui/ysh/mPLUG-Owl/mPLUG-Owl2

python /data/chenhang_cui/ysh/mPLUG-Owl/mPLUG-Owl2/infer_1000_safe.py
'''