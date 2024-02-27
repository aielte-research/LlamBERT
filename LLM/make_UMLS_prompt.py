import argparse, os
import json
from typing import Union
import fire

def my_open(fpath,mode="w"):
   dirname=os.path.dirname(fpath)
   if len(dirname)>0 and not os.path.exists(dirname):
      os.makedirs(dirname)
   return open(fpath, mode)

def get_sys_prompt_plain():
   return "Please answer with a 'yes' or a 'no' only!"

def get_sys_prompt():
   return {
      "role": "system",
      "content": get_sys_prompt_plain()
   }

def get_user_prompt_plain(cui, syns):
   return f"Decide if the term: {'; '.join([x.replace('&#x7C;', 'XXXYYYZZZ') for x in syns])} is related to the human nervous system. Exclude the only vascular structures, even if connected to the nervous system. If multiple examples or terms with multiple words are given, treat them all as a whole and make your decision based on that."

def get_user_prompt(cui, syns):
   return {
      "role": "user",
      "content": get_user_prompt_plain(cui, syns)
   }

def get_assistant_prompt(content):
   return {
      "role": "assistant",
      "content": content
   }

def main(
      input_fpath: str="../UMLS/regions/train_concepts.json",
      output_folder: str="model_inputs/UMLS/",
      output_fname: Union[str, None]=None,
      shots: int=1,
      plain_format: bool=False,
      llama_format: bool=False
   ):
   with open(input_fpath) as f:
      concepts = json.loads(f.read())

   if output_fname is None:
      output_fname = f'{os.path.basename(input_fpath).split(".")[0]}_prompts.json'
   
   prompts=[]
   if plain_format:
      prompts=[get_user_prompt_plain(cui, syns) for cui, syns in concepts.items()]
   elif llama_format:
      for cui, syns in concepts.items():
         prompt = ""
         if shots>=1:
            prompt += f"[INST] <<SYS>>\n{get_sys_prompt_plain()}\n<</SYS>>\n{get_user_prompt_plain('C2328354', ['C4 branch to right iliocostalis cervicis'])}[/INST]\nyes\n"
         if shots>=2:
            prompt += f"[INST] <<SYS>>\n{get_sys_prompt_plain()}\n<</SYS>>\n{get_user_prompt_plain('C1514049', ['Neoplastic Neuroepithelial Cell and Neoplastic Perineural Cell'])}[/INST]\nno\n"
         if shots>=3:
            print("Only options 0, 1 and 2 shot are implemented.")
         prompt += f"[INST] <<SYS>>\n{get_sys_prompt_plain()}\n<</SYS>>\n{get_user_prompt_plain(cui, syns)}[/INST]"
         prompts.append(prompt)
   else:
      for cui, syns in concepts.items():
         prompt = []
         if shots>=1:
            prompt.append(get_sys_prompt())
            prompt.append(get_user_prompt('C2328354', ['C4 branch to right iliocostalis cervicis']))
            prompt.append(get_assistant_prompt("yes"))
         if shots>=2:
            prompt.append(get_sys_prompt())
            prompt.append(get_user_prompt('C1514049', ['Neoplastic Neuroepithelial Cell and Neoplastic Perineural Cell']))
            prompt.append(get_assistant_prompt("no"))
         if shots>=3:
            print("Only options 0, 1 and 2 shot are implemented.")
         prompt.append(get_sys_prompt())
         prompt.append(get_user_prompt(cui, syns))
         prompts.append(prompt)
         
   with my_open(os.path.join(output_folder, output_fname), 'w') as outfile:
      json.dump(prompts, outfile, indent=3)

if __name__ == '__main__':
   fire.Fire(main)
