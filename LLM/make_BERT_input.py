import argparse, os
import json

def my_open(fpath,mode="w"):
   dirname=os.path.dirname(fpath)
   if len(dirname)>0 and not os.path.exists(dirname):
      os.makedirs(dirname)
   return open(fpath, mode)

def main(prompt_fpath, LLM_output_fpath, target_labels_fpath, output_fpath):
   with open(prompt_fpath) as f:
      prompts = json.loads(f.read())

   if target_labels_fpath is not None:
      with open(target_labels_fpath) as f:
         labels = json.loads(f.read())
   else:
      with open(LLM_output_fpath) as f:
         labels = json.loads(f.read())["outputs_binary"]
   
   data=[]
   for prompt, label in zip(prompts, labels):
      data.append({"txt": prompt, "label": label})

   with my_open(output_fpath, 'w') as outfile:
      json.dump(data, outfile, indent=3)

if __name__ == '__main__':
   parser = argparse.ArgumentParser(prog='Sentiment promt maker', description='This script prepares a promts for sentiment analysis for Llama 2')
   parser.add_argument('-p', '--prompt_fpath', required=False, default="model_inputs/UMLS/train_concepts_prompts.json")
   parser.add_argument('-l', '--LLM_output_fpath', required=False, default="model_outputs/UMLS/train_concepts_prompts_meta-llama_Llama-2-70b-chat-hf.json")
   parser.add_argument('-t', '--target_labels_fpath', required=False, default=None)
   parser.add_argument('-o', '--output_folder', required=False, default="../BERT/data/UMLS_regions_10k/full_prompt")
   parser.add_argument('-f', '--output_fname', required=False, default="train.json")
   args = parser.parse_args()
      
   output_fpath = os.path.join(args.output_folder, args.output_fname)
   
   main(prompt_fpath=args.prompt_fpath, LLM_output_fpath = args.LLM_output_fpath, target_labels_fpath=args.target_labels_fpath, output_fpath = output_fpath)