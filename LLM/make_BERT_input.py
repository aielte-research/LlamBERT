import os
import json
import random
from typing import Optional
import fire

import sys
sys.path.append("..")
from utils import my_open

def main(
   prompt_fpath: str = "model_inputs/UMLS/train_concepts_prompts.json",
   LLM_output_fpath: str = "model_outputs/UMLS/train_concepts_prompts_meta-llama_Llama-2-70b-chat-hf.json",
   target_labels_fpath: Optional[str]=None,
   output_folder: str = "../BERT/data/UMLS_regions_10k/short_prompt",
   output_fname: str = "train",
   a_pos: Optional[int]=None,
   b_pos: Optional[int]=None,
   shuffle: bool = False,
   **kwargs
):
   if len(kwargs) > 0:
      raise ValueError(f"Unknown argument(s): {kwargs}")

   with open(prompt_fpath) as f:
      prompts = json.loads(f.read())
   if isinstance(prompts[0], dict):
      prompts = [x["txt"] for x in prompts]

   if target_labels_fpath is not None:
      with open(target_labels_fpath) as f:
         labels = json.loads(f.read())
   else:
      with open(LLM_output_fpath) as f:
         labels = json.loads(f.read())["outputs_binary"]
   
   data=[]
   for prompt, label in zip(prompts, labels):
      if label in [0,1]:
         data.append({"txt": prompt[a_pos:b_pos], "label": label})
   
   if shuffle:
      random.shuffle(data)

   output_fpath = f"{os.path.join(output_folder, output_fname)}.json"
   with my_open(output_fpath, 'w') as outfile:
      json.dump(data, outfile, indent=3)

if __name__ == '__main__':
   fire.Fire(main)