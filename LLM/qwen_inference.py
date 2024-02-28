import fire
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from tqdm import tqdm
from typing import Union
import sys
sys.path.append("..")
from utils import my_open

def main(
    model_name: str,
    prompt_file: str,
    quantization: bool=False,
    include_full_outputs: bool=False,
    max_new_tokens: int=100, #The maximum numbers of tokens to generate
    target_file: Union[None, str]=None,
    seed: int=42, #seed value for reproducibility
    use_fast_kernels: bool = False # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
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
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        return_dict=True,
        load_in_8bit=quantization,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )

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

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, pad_token='<|endoftext|>')
    
    output_texts=[]
    output_texts_cleaned=[]
    output_texts_binary=[]
    # Process prompts in batches
    for prompt in tqdm(user_prompts):
        if isinstance(prompt,list):
            if prompt[0]["role"] == "system":
                sys_prompt=prompt[0]["content"]+" Answer only with one word!"
            else:
                sys_prompt=""

            with torch.no_grad():
                output_text, _ = model.chat(
                    tokenizer,
                    prompt[-1]["content"],
                    history=None,
                    system=sys_prompt,
                    max_new_tokens=max_new_tokens,
                )
        else:
            sys_prompt=" Answer only with one word!"

            with torch.no_grad():
                output_text, _ = model.chat(
                    tokenizer,
                    prompt,
                    history=None,
                    system=sys_prompt,
                    max_new_tokens=max_new_tokens,
                )
        
        output_texts.append(output_text)
        output_cleaned = output_text.strip().strip(".,!?;'\"").strip().strip(".,!?;'\"").lower()
        output_texts_cleaned.append(output_cleaned)
        if output_cleaned == "no":
            output_texts_binary.append(0)
        elif output_cleaned == "yes":
            output_texts_binary.append(1)
        elif "negative" in output_cleaned and "positive" not in output_cleaned:
            output_texts_binary.append(0)
        elif "positive" in output_cleaned and "negative" not in output_cleaned:
            output_texts_binary.append(1)
        else:
            output_texts_binary.append(2)

    output_data={
        "settings": args,
        "outputs": output_texts,
        "outputs_cleaned": output_texts_cleaned,
        "outputs_binary": output_texts_binary,
        "output_stats": {
            "negative": len([x for x in output_texts_binary if x==0]),
            "positive": len([x for x in output_texts_binary if x==1]),
            "other": len([x for x in output_texts_binary if x==2])
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
        
        assert len(output_texts_binary)==len(targets), f"Provided target file is not the sema lenght as the inputs"

        correct=0
        misclassifed=[]
        for target, output, output_full in zip(targets, output_texts_binary, output_texts):
            if target == output:
                correct+=1
            else:
                misclassifed.append(output_full)
        output_data["accuracy"]=int(10000*correct/len(targets))/100
        print(f"Accuracy: {output_data['accuracy']}%")
        output_data["misclassifed"]=misclassifed
    
    output_fpath = os.path.join("model_outputs",f'{os.path.splitext(os.path.basename(prompt_file))[0].strip(".")}_{os.path.basename(model_name.rstrip("/"))}.json')
    with my_open(output_fpath) as file:
        json.dump(output_data, file, indent=4)
             

if __name__ == "__main__":
    fire.Fire(main)