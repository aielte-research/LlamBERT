import os, sys
import json
from tqdm import tqdm
import fire
import time
from typing import Union

from openai import OpenAI
client = OpenAI()
import sys
sys.path.append("..")
from utils import my_open

def json_pretty_print(text, indent=4):
   level = 0
   list_level = 0
   inside_apostrophe = 0
   last_backslash_idx = -2
   ret = ""
   for i,c in enumerate(text):
      if c=="}" and inside_apostrophe % 2 == 0:
         level -= 1
         ret += "\n" + " "*(level*indent)
      ret += c
      if c=="{" and inside_apostrophe % 2 == 0:
         level += 1
         ret += "\n" + " "*(level*indent)
      elif c=="[" and inside_apostrophe % 2 == 0:
         list_level += 1
      elif c=="]" and inside_apostrophe % 2 == 0:
         list_level -= 1
      elif c=='"' and last_backslash_idx != i-1:
         inside_apostrophe += 1
      elif c=="\\":
         last_backslash_idx=i
      elif c=="," and inside_apostrophe % 2 == 0 and list_level<2:
         ret += "\n" + " "*(level*indent)
   return ret

def main(
   prompt_file: str,
   model_name: str="text-embedding-3-small",
   output_path: str="model_inputs/IMDB",
   dim: Union[None, int]=None,
   **kwargs
):
   if len(kwargs) > 0:
      raise ValueError(f"Unknown argument(s): {kwargs}")
   if prompt_file is None:
      print("No user prompt provided (option --promt-file). Exiting.")
      sys.exit(1)

   assert os.path.exists(prompt_file), f"Provided Prompt file does not exist {prompt_file}"
   extension = os.path.splitext(prompt_file)[1].strip(".")
   if extension.lower() in ["json"]:
      with open(prompt_file, "r") as f:
         user_prompts = json.load(f)
      assert isinstance(user_prompts, list), "JSON content is not a list"
   else:
      assert False, f"Error: unrecognized Prompt file extension '{extension}'!"

   if dim is None:
      get_emb = lambda entry: client.embeddings.create(input = [entry["txt"]], model=model_name).data[0].embedding
   else:
      get_emb = lambda entry: client.embeddings.create(input = [entry["txt"]], model=model_name, dimensions=dim).data[0].embedding
   for entry in tqdm(user_prompts):
      try:
         entry["embedding"] = get_emb(entry)
      except Exception as error:
         print(error)
         print("Inference failed, will try again in a minute...")
         time.sleep(60)
         try:
            entry["embedding"] = get_emb(entry)
         except Exception as error:
            print(error)
            print("Inference failed, will try again in a 10 minutes...")
            time.sleep(600)
            try:
               entry["embedding"] = get_emb(entry)
            except Exception as error:
               print(error)
               print(f"Inference failed again, terminating session...")
               break
      
   output_fpath = os.path.join(output_path,f'{os.path.splitext(os.path.basename(prompt_file))[0].strip(".")}_{os.path.basename(model_name.rstrip("/"))}_dim-{dim}.json')
   with my_open(output_fpath) as outfile:
      outfile.write(json_pretty_print(json.dumps(user_prompts,separators=(',', ': '))))
      #json.dump(user_prompts, file, indent=4)

if __name__ == "__main__":
   fire.Fire(main)