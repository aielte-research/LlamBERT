import argparse, os
import json

def my_open(fpath,mode="w"):
   dirname=os.path.dirname(fpath)
   if len(dirname)>0 and not os.path.exists(dirname):
      os.makedirs(dirname)
   return open(fpath, mode)

def main(reiviews_fpath, output_folder, output_fname):
   with open(reiviews_fpath) as f:
      concepts = json.loads(f.read())
   
   prompts=[]
   for cui, syns in concepts.items():
      prompts.append(f"[INST] <<SYS>>\nPlease answer with a 'yes' or a 'no' only!\n<</SYS>>\nDecide if the term: C4 branch to right iliocostalis cervicis is related to the human nervous system. Exclude the only vascular structures, even if connected to the nervous system. If multiple examples or terms with multiple words are given, treat them all as a whole and make your decision based on that. [/INST]\nyes\n[INST] <<SYS>>\nPlease answer with a 'yes' or a 'no' only!\n<</SYS>>\nDecide if the term: Neoplastic Neuroepithelial Cell and Neoplastic Perineural Cell is related to the human nervous system. Exclude the only vascular structures, even if connected to the nervous system. If multiple examples or terms with multiple words are given, treat them all as a whole and make your decision based on that. [/INST]\nno\n[INST] <<SYS>>\nPlease answer with a 'yes' or a 'no' only!\n<</SYS>>\nDecide if the term: {'; '.join(syns)} is related to the human nervous system. Exclude the only vascular structures, even if connected to the nervous system. If multiple examples or terms with multiple words are given, treat them all as a whole and make your decision based on that. [/INST]")
      #prompts.append(f"[INST] <<SYS>>\nPlease answer with a 'yes' or a 'no' only!\n<</SYS>>\nDecide if the term: {'; '.join(syns)} is related to the human nervous system. Exclude the only vascular structures, even if connected to the nervous system. If multiple examples or terms with multiple words are given, treat them all as a whole and make your decision based on that. [/INST]")
         
   with my_open(os.path.join(output_folder, f"{output_fname}_prompts.json"), 'w') as outfile:
      json.dump(prompts, outfile)#, indent=4)

if __name__ == '__main__':
   parser = argparse.ArgumentParser(prog='Sentiment promt maker', description='This script prepares a promts for sentiment analysis for Llama 2')
   parser.add_argument('-i', '--input_fpath', required=False, default="../UMLS/train_concepts.json")
   parser.add_argument('-o', '--output_folder', required=False, default="model_inputs/UMLS/")
   parser.add_argument('-f', '--output_fname', required=False, default=None)
   args = parser.parse_args()
   
   output_fname = args.output_fname
   if output_fname is None:
      output_fname = os.path.basename(args.input_fpath).split(".")[0]
   
   main(reiviews_fpath=args.input_fpath, output_folder = args.output_folder, output_fname = output_fname)
