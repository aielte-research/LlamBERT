import argparse, os
import json
import random

def my_open(fpath,mode="w"):
   dirname=os.path.dirname(fpath)
   if len(dirname)>0 and not os.path.exists(dirname):
      os.makedirs(dirname)
   return open(fpath, mode)

def main(prompt_fpath, LLM_output_fpath, output_fpath, shuffle=True):
   with open(prompt_fpath) as f:
      gold_data = json.loads(f.read())
      
   prompts = [x["txt"] for x in gold_data]
   gold_labels = [x["label"] for x in gold_data]

   with open(LLM_output_fpath) as f:
      labels = json.loads(f.read())["outputs_binary"]
   
   data=[]
   for prompt, label, gold_label in zip(prompts, labels, gold_labels):
      if label==gold_label:
         data.append({"txt": prompt, "label": label})

   print(f"Supergold data size: {len(data)} out of {len(labels)}")
   
   if shuffle:
      random.shuffle(data)

   with my_open(output_fpath, 'w') as outfile:
      json.dump(data, outfile, indent=3)

if __name__ == '__main__':
   parser = argparse.ArgumentParser(prog='Sentiment promt maker', description='This script prepares a promts for sentiment analysis for Llama 2')
   parser.add_argument('-s', '--shuffle', type=bool, required=False, default=True)
   args = parser.parse_args()
   
   main(
      prompt_fpath = "/home/projects/DeepNeurOntology/IMDB_data/train.json",
      LLM_output_fpath = "model_outputs/IMDB/train_prompts_meta-llama_Llama-2-70b-chat-hf.json",
      output_fpath = "/home/projects/DeepNeurOntology/IMDB_data/BERT_inputs/SuperGold/train.json",
      shuffle = args.shuffle
   )
