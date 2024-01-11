import argparse, os
import json
import random

def my_open(fpath,mode="w"):
   dirname=os.path.dirname(fpath)
   if len(dirname)>0 and not os.path.exists(dirname):
      os.makedirs(dirname)
   return open(fpath, mode)

def make_data(name="train", makelarge=False, balanced=True):
   with open(f"/home/projects/DeepNeurOntology/IMDB_data/BERT_inputs/DEV/Gold/{name}.json") as f:
      gold_data = json.loads(f.read())
      
   gold_prompts = [x["txt"] for x in gold_data]
   gold_labels = [x["label"] for x in gold_data]

   with open(f"/home/projects/DeepNeurOntology/IMDB_data/BERT_inputs/DEV/Llama_output_train/{name}.json") as f:
      llama_data = json.loads(f.read())

   llama_prompts = [x["txt"] for x in llama_data]
   llama_labels = [x["label"] for x in llama_data]
   
   data_same=[]
   data_different=[]
   for gold_prompt, gold_label in zip(gold_prompts, gold_labels):
      for llama_prompt, llama_label in zip(llama_prompts, llama_labels):
         if gold_prompt==llama_prompt:
            sentiment = "N/A"
            if llama_label==1:
               sentiment = "positive"
            elif llama_label==0:
               sentiment = "negative"
            if llama_label==gold_label:
               data_same.append({"txt": f"{gold_prompt}\nPredicted sentiment: {sentiment}.", "label": 1})
            else:
               data_different.append({"txt": f"{gold_prompt}\nPredicted sentiment: {sentiment}.", "label": 0})
            break

   print(f"{name}_data_same: {len(data_same)}, {name}_data_different: {len(data_different)}")
   
   if makelarge:
      while len(data_different)<len(data_same):
         data_different += data_different

   if balanced:
      cut_len=min(len(data_same), len(data_different))
      data = data_same[:cut_len].copy() + data_different[:cut_len].copy()
   else:
      data = data_same.copy() + data_different.copy()
   random.shuffle(data)

   with my_open(f"/home/projects/DeepNeurOntology/IMDB_data/BERT_inputs/DEV/EasyOrHard_unbalanced/{name}.json", 'w') as outfile:
      json.dump(data, outfile, indent=3)

def main():
   make_data("train", balanced=False)
   make_data("dev")
   make_data("test", balanced=False)

if __name__ == '__main__':
   main()
