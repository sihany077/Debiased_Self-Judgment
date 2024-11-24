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
def sample(
    self,
    input_ids: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    logits_warper: Optional[LogitsProcessorList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    **model_kwargs,
) -> Union[SampleOutput, torch.LongTensor]:
    # init values
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id

    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
    output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
    output_attentions = (
        output_attentions if output_attentions is not None else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
    )

    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

    this_peer_finished = False  # used by synced_gpus only
    model_kwargs_cd, model_kwargs_dd = None, None
    # auto-regressive generation
    while True:
        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                break

        # prepare model inputs
        
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # forward pass to get next token
        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        if synced_gpus and this_peer_finished:
            continue  # don't waste resources running the code we don't need

        next_token_logits = outputs.logits[:, -1, :]
        
        
        ## For contrastive decoding initial
        use_cd = model_kwargs.get("images_cd") != None
        use_dd = model_kwargs.get("use_dd")
        use_dd_unk = model_kwargs.get('use_dd_unk')
        output_attentions_wo_img = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states_wo_img = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        
        if use_cd or use_dd or use_dd_unk:
            if use_cd:
                model_kwargs_cd = model_kwargs.copy()
                model_inputs_cd = self.prepare_inputs_for_generation_cd(input_ids, **model_kwargs_cd)
            else: 
                model_kwargs_cd = model_kwargs.copy() if model_kwargs_cd is None else model_kwargs_cd
                if use_dd_unk:
                    input_ids_dd = copy.deepcopy(input_ids)
                    input_ids_dd[input_ids_dd == IMAGE_TOKEN_INDEX] = 0 # unk
                elif use_dd:
                    indices_to_keep = torch.where(input_ids != IMAGE_TOKEN_INDEX)[0]
                    attention_mask = model_kwargs['attention_mask']
                    model_kwargs_cd['attention_mask'] = attention_mask[:,indices_to_keep]
                    input_ids_dd = input_ids[input_ids != IMAGE_TOKEN_INDEX].unsqueeze(0)
                model_inputs_cd = self.prepare_inputs_for_generation_cd(input_ids_dd, **model_kwargs_cd)
            
            outputs_cd = self(
                **model_inputs_cd,
                return_dict=True,
                output_attentions=output_attentions_wo_img,
                output_hidden_states=output_hidden_states_wo_img,
            )
            next_token_logits_cd = outputs_cd.logits[:, -1, :]

            if use_dd_unk and use_dd: # outputs_cd now is just for use_dd_unk if both is used
                model_kwargs_dd = model_kwargs.copy() if model_kwargs_dd is None else model_kwargs_dd
                indices_to_keep = torch.where(input_ids != IMAGE_TOKEN_INDEX)[0]
                attention_mask = model_kwargs['attention_mask']
                model_kwargs_dd['attention_mask'] = attention_mask[:,indices_to_keep]
                input_ids_dd = input_ids[input_ids != IMAGE_TOKEN_INDEX].unsqueeze(0)
                model_inputs_dd = self.prepare_inputs_for_generation_cd(input_ids_dd, **model_kwargs_dd)
                outputs_dd = self(
                        **model_inputs_dd,
                        return_dict=True,
                        output_attentions=output_attentions_wo_img,
                        output_hidden_states=output_hidden_states_wo_img,
                    )
                next_token_logits_dd_none = outputs_dd.logits[:, -1, :]
                next_token_logits_cd = (next_token_logits_cd + next_token_logits_dd_none) / 2

            ## cd_comments: pre-process logits from contrastive inputs
            cd_alpha = model_kwargs.get("cd_alpha") if model_kwargs.get("cd_alpha") is not None else 1
            cd_beta = model_kwargs.get("cd_beta") if model_kwargs.get("cd_beta") is not None else 0.1
            
            cutoff = torch.log(torch.tensor(cd_beta)) + next_token_logits.max(dim=-1, keepdim=True).values
            
            diffs = (1+cd_alpha)*next_token_logits - cd_alpha*next_token_logits_cd
            # next_token_logits_origin = next_token_logits

            next_token_logits = diffs.masked_fill(next_token_logits < cutoff, -float("inf"))

            ## cd_comments: apply temperature warping and top-k filtering in contrastive decoding
            cd_logits = logits_processor(input_ids, next_token_logits)
            cd_logits = logits_warper(input_ids, cd_logits)

            next_token_scores = cd_logits
            probs = nn.functional.softmax(cd_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                # scores += (next_token_scores,)
                # NOTE changed 现在的logit没有除以temp
                scores += (next_token_logits,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )
        ## cd_comments: update model_kwargs_cd for contrastive decoding
        if use_cd or use_dd or use_dd_unk:
            model_kwargs_cd = self._update_model_kwargs_for_generation(
                outputs_cd, model_kwargs_cd, is_encoder_decoder=self.config.is_encoder_decoder
            )
        if use_dd and use_dd_unk:
            model_kwargs_dd = self._update_model_kwargs_for_generation(
                outputs_dd, model_kwargs_dd, is_encoder_decoder=self.config.is_encoder_decoder
            )

        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                this_peer_finished = True

        # stop if we exceed the maximum length
        if stopping_criteria(input_ids, scores):
            this_peer_finished = True

        if this_peer_finished and not synced_gpus:
            break

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return SampleEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
            )
        else:
            return SampleDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )
    else:
        return input_ids

def evolve_vcd_sampling():
    transformers.generation.utils.GenerationMixin.sample = sample
evolve_vcd_sampling()

LABEL_DICT = {0: ['yes',], 1: ['maybe',],2: ['no',]}

def extract_topk_token_logits(logits, tokenizer, apply_softmax=True):
    probs = logits.float().cpu() if not apply_softmax else torch.softmax(logits, dim=-1).float().cpu()
    top_probs, top_tokens = torch.topk(probs, k=10)
    temp = {}
    for prob, token in zip(top_probs[0], top_tokens[0]):
        str_token = tokenizer.decode(token.item())
        str_token = str_token.lower().strip()
        if str_token not in temp.keys():
            temp[str_token] = prob.item()
        else:
            pass
    return temp

def get_required_token_logits(top_token_probs, label_dict=LABEL_DICT):
    p_y = [0] * len(label_dict)
    for i, answers in label_dict.items():
        prob = 0
        for a in answers:
            a = a.lower()
            if a not in top_token_probs.keys():
                prob += 0
            else:
                prob += top_token_probs[a]
        p_y[i] = prob
    return p_y

def eval_model(args):
    model_path = 'MAGAer13/mplug-owl2-llama2-7b'
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, load_8bit=False, load_4bit=False, device="cuda")
    # import ipdb;ipdb.set_trace()
    #NOTE attention mask and pad_token_id
    # if not hasattr(tokenizer, 'pad_token_id'):
    tokenizer.pad_token_id = tokenizer.eos_token_id


    output_file = args.output_file
    input_dir = args.input_dir

    with open(output_file, "a+") as f:

        for filename in tqdm(os.listdir(input_dir)[:500]):
            if filename.endswith((".jpg", ".jpeg", ".png")):
                qs = "Describe this image in detail."
                cur_prompt = qs

                image = Image.open(os.path.join(args.input_dir, filename)).convert('RGB')
                max_edge = max(image.size) # We recommand you to resize to squared image for BEST performance.
                image = image.resize((max_edge, max_edge))

                image_tensor = process_images([image], image_processor)
                image_tensor = image_tensor.to(model.device, dtype=torch.float16)

                conv = conv_templates["mplug_owl2"].copy()
                roles = conv.roles

                accumulated_sentences = ''

                inp = DEFAULT_IMAGE_TOKEN + qs
                conv.append_message(conv.roles[0], inp)
                conv.append_message(conv.roles[1], accumulated_sentences)
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
                        max_new_tokens=args.max_new_token,
                        use_cache=True,
                        stopping_criteria=[stopping_criteria]
                        )

                outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()

                

                
                result = {"id": filename, "question": cur_prompt, "answer": outputs, "model": "LLaVA-1.5-7b_vdd_tot"}
                json.dump(result, f)
                f.write('\n')
                f.flush()

    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--input_dir", type=str, default="/data/chenhang_cui/dataset_for_attack/val2014")#ori_caption.jsonl
    parser.add_argument("--output_file", type=str, default="/data/chenhang_cui/ysh/LLaVA-Align/experiments/eval/sampling/chair/mplug_owl2/origin_64.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_token", type=int, default=64)
    
    args = parser.parse_args()
    set_seed(args.seed)

    eval_model(args)

'''
conda activate /data/public_models/eval/env/ysh_mplug_owl2;cd /data/chenhang_cui/ysh/mPLUG-Owl/mPLUG-Owl2
python /data/chenhang_cui/ysh/mPLUG-Owl/mPLUG-Owl2/origin.py --max_new_token 64 --output_file /data/chenhang_cui/ysh/LLaVA-Align/experiments/eval/sampling/chair/mplug_owl2/greedy_64.jsonl

'''