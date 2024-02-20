# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# from accelerate import init_empty_weights, load_checkpoint_and_dispatch

import fire
import os
import sys
import time

import torch
from transformers import LlamaTokenizer

import json

from peft import PeftModel
from transformers import LlamaForCausalLM

from tqdm import trange

# Function to load the main model for text generation
def load_model(model_name, quantization):
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        return_dict=True,
        load_in_8bit=quantization,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    return model

# Function to load the PeftModel for performance optimization
def load_peft_model(model, peft_model):
    peft_model = PeftModel.from_pretrained(model, peft_model)
    return peft_model

def my_open_w(fpath):
    if not os.path.exists(os.path.dirname(fpath)):
        os.makedirs(os.path.dirname(fpath))
    return open(fpath, 'w')

def main(
    model_name,
    peft_model: str=None,
    quantization: bool=False,
    include_full_outputs: bool=False,
    max_new_tokens: int=100, #The maximum numbers of tokens to generate
    prompt_file: str=None,
    target_file: str=None,
    seed: int=42, #seed value for reproducibility
    do_sample: bool=True, #Whether or not to use sampling ; use greedy decoding otherwise.
    min_length: int=None, #The minimum length of the sequence to be generated, input prompt + min_new_tokens
    use_cache: bool=True, #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float=1.0, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float=1.0, # [optional] The value used to modulate the next token probabilities.
    top_k: int=50, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float=1.0, #The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int=1, #[optional] Exponential penalty to the length that is used with beam-based generation. 
    use_fast_kernels: bool = False, # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    batch_size: int=1,  # Batch_size parameter
    **kwargs
):
    args = dict(locals())
    if prompt_file is not None:
        assert os.path.exists(
            prompt_file
        ), f"Provided Prompt file does not exist {prompt_file}"
        extension = os.path.splitext(prompt_file)[1].strip(".")
        if extension.lower() in ["json"]:
            with open(prompt_file, "r") as f:
                user_prompts = json.load(f)
                assert isinstance(user_prompts, list), "JSON content is not a list"
        elif extension.lower()=="txt":
            with open(prompt_file, "r") as f:
                user_prompt = "\n".join(f.readlines())
            user_prompts = [user_prompt]
        else:
            assert False, f"Error: unrecognized Prompt file extension '{extension}'!"

    elif not sys.stdin.isatty():
        user_prompt = "\n".join(sys.stdin.readlines())
        user_prompts = [user_prompt]
    else:
        print("No user prompt provided. Exiting.")
        sys.exit(1)
    
    # Set the seeds for reproducibility
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    
    model = load_model(model_name, quantization)
    if peft_model:
        model = load_peft_model(model, peft_model)

    model.eval()
    
    if use_fast_kernels:
        """
        Setting 'use_fast_kernels' will enable
        using of Flash Attention or Xformer memory-efficient kernels 
        based on the hardware being used. This would speed up inference when used for batched inputs.
        """
        try:
            from optimum.bettertransformer import BetterTransformer
            model = BetterTransformer.transform(model)    
        except ImportError:
            print("Module 'optimum' not found. Please install 'optimum' it before proceeding.")

    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # Add padding token
    tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    model.resize_token_embeddings(len(tokenizer))
    
    output_texts=[]
    output_texts_answer_only=[]
    output_texts_answer_only_cleaned=[]
    output_texts_answer_only_binary=[]
    # Process prompts in batches
    for i in trange(0, len(user_prompts), batch_size):
        batch_prompts = user_prompts[i:i+batch_size]

        # Tokenize the batch
        batch = tokenizer(batch_prompts, padding="longest", truncation=True, return_tensors="pt")
        batch = {k: v.to("cuda") for k, v in batch.items()}
            
        # batch = tokenizer(user_prompt, padding='max_length', truncation=True, max_length=max_padding_length, return_tensors="pt")
        # batch = {k: v.to("cuda") for k, v in batch.items()}

        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                min_length=min_length,
                use_cache=use_cache,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                **kwargs 
            )
        #e2e_inference_time = (time.perf_counter()-start)*1000
        #print(f"the inference time is {e2e_inference_time} ms")

        for inpt, output in zip(batch_prompts,outputs):
            output_text = tokenizer.decode(output, skip_special_tokens=True)
            output_text_answer_only=output_text[len(inpt):]
            
            output_texts.append(output_text)
            output_texts_answer_only.append(output_text_answer_only)
            output_cleaned = output_text_answer_only.strip().strip(".,!?;'\"").strip().strip(".,!?;'\"").lower()
            output_texts_answer_only_cleaned.append(output_cleaned)
            if output_cleaned == "no":
                output_texts_answer_only_binary.append(0)
            elif output_cleaned == "yes":
                output_texts_answer_only_binary.append(1)
            elif "negative" in output_cleaned and "positive" not in output_cleaned:
                output_texts_answer_only_binary.append(0)
            elif "positive" in output_cleaned and "negative" not in output_cleaned:
                output_texts_answer_only_binary.append(1)
            else:
                output_texts_answer_only_binary.append(2)

    output_data={
        "settings": args,
        "outputs": output_texts_answer_only,
        "outputs_cleaned": output_texts_answer_only_cleaned,
        "outputs_binary": output_texts_answer_only_binary,
        "output_stats": {
            "negative": len([x for x in output_texts_answer_only_binary if x==0]),
            "positive": len([x for x in output_texts_answer_only_binary if x==1]),
            "other": len([x for x in output_texts_answer_only_binary if x==2])
        }
    }
    if include_full_outputs:
        output_data["outputs_full"] = output_texts

    if target_file is not None:
        assert os.path.exists(
            target_file
        ), f"Provided target file does not exist {target_file}"
        extension = os.path.splitext(target_file)[1].strip(".")
        if extension.lower() in ["json"]:
            with open(target_file, "r") as f:
                targets = json.load(f)
                assert isinstance(targets, list), "JSON content is not a list"
        else:
            assert False, f"Error: unrecognized target file extension '{extension}'!"
        
        assert len(output_texts_answer_only_binary)==len(targets), f"Provided target file is not the sema lenght as the inputs"

        correct=0
        misclassifed=[]
        for target, output, output_full in zip(targets, output_texts_answer_only_binary, output_texts):
            if target == output:
                correct+=1
            else:
                misclassifed.append(output_full)
        output_data["accuracy"]=int(10000*correct/len(targets))/100
        print(f"Accuracy: {output_data['accuracy']}%")
        output_data["misclassifed"]=misclassifed
    
    output_fpath = os.path.join("model_outputs",f'{os.path.splitext(os.path.basename(prompt_file))[0].strip(".")}_{os.path.basename(model_name.rstrip("/"))}.json')
    with my_open_w(output_fpath) as file:
        json.dump(output_data, file, indent=4)
             

if __name__ == "__main__":
    fire.Fire(main)