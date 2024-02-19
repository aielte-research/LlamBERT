import argparse, os
import json

def my_open(fpath,mode="w"):
   dirname=os.path.dirname(fpath)
   if len(dirname)>0 and not os.path.exists(dirname):
      os.makedirs(dirname)
   return open(fpath, mode)

def main(reiviews_fpath, output_folder, output_fname, shots=0):
   with open(reiviews_fpath) as f:
      concepts = json.loads(f.read())
   
   prompts=[]
   for cui, syns in concepts.items():
      if shots==0:
         prompts.append([
            {"role": "system", "content": "Please answer with a 'yes' or a 'no' only!"},
            {"role": "user", "content": f"Decide if the term: {'; '.join([x.replace('&#x7C;', 'XXXYYYZZZ') for x in syns])} is related to the human nervous system. Exclude the only vascular structures, even if connected to the nervous system. If multiple examples or terms with multiple words are given, treat them all as a whole and make your decision based on that."}
         ])            
      elif shots==1:
         prompts.append([
            {"role": "system", "content": "Please answer with a 'yes' or a 'no' only!"},
            {"role": "user", "content": "Decide if the term: C4 branch to right iliocostalis cervicis is related to the human nervous system. Exclude the only vascular structures, even if connected to the nervous system. If multiple examples or terms with multiple words are given, treat them all as a whole and make your decision based on that."},
            {"role": "assistant", "content": "yes"},
            {"role": "system", "content": "Please answer with a 'yes' or a 'no' only!"},
            {"role": "user", "content": f"Decide if the term: {'; '.join([x.replace('&#x7C;', 'XXXYYYZZZ') for x in syns])} is related to the human nervous system. Exclude the only vascular structures, even if connected to the nervous system. If multiple examples or terms with multiple words are given, treat them all as a whole and make your decision based on that."}
         ])
      else:
         print("Only options 0 and 1 shot are implemented.")
   
   with my_open(os.path.join(output_folder, f"{output_fname}_{shots}-shot_prompts.json"), 'w') as outfile:
      json.dump(prompts, outfile, indent=3)

if __name__ == '__main__':
   parser = argparse.ArgumentParser(prog='Sentiment promt maker', description='This script prepares a promts for sentiment analysis for Llama 2')
   parser.add_argument('-i', '--input_fpath', required=False, default="../UMLS/test_concepts.json")
   parser.add_argument('-o', '--output_folder', required=False, default="model_inputs/UMLS/")
   parser.add_argument('-f', '--output_fname', required=False, default=None)
   parser.add_argument('-s', '--shots', required=False, type=int, default=0)
   args = parser.parse_args()
   
   output_fname = args.output_fname
   if output_fname is None:
      output_fname = os.path.basename(args.input_fpath).split(".")[0]
   
   main(reiviews_fpath=args.input_fpath, output_folder = args.output_folder, output_fname = output_fname, shots = args.shots)
