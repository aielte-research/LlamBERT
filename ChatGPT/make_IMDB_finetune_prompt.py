import argparse, os
import json

def my_open(fpath,mode="w"):
   dirname=os.path.dirname(fpath)
   if len(dirname)>0 and not os.path.exists(dirname):
      os.makedirs(dirname)
   return open(fpath, mode)

def main(reiviews_fpath, output_folder, output_fname, shots=0):
   with open(reiviews_fpath) as f:
      reviews = json.loads(f.read())
   
   train_data=[]
   for rev in reviews:
      if isinstance(rev, str):
         rev_str=rev
      else:
         rev_str=rev["txt"]

      if isinstance(rev, dict) and "label" in rev:
         if rev["label"]==0:
            completion="negative"
         elif rev["label"]==1:
            completion="positive"
         else:
            raise
      else:
         raise

      if shots==0:
         train_data.append(
            {"messages": [
               {"role": "system", "content": "Please answer with 'positive' or 'negative' only!"},
               {"role": "user", "content": f"Decide if the following movie review is positive or negative: \n{rev_str}\n If the movie review is positive please answer 'positive', if the movie review is negative please answer 'negative'. Make your decision based on the whole text."},
               {"role": "assistant", "content": completion}
            ]}
         )
      else:
         print("Only option 0-shot is implemented.")
         raise
   
   with my_open(os.path.join(output_folder, f"{output_fname}_finetune_data.jsonl"), 'w') as outfile:
      for entry in train_data:
         json.dump(entry, outfile)
         outfile.write('\n')

if __name__ == '__main__':
   parser = argparse.ArgumentParser(prog='Sentiment promt maker', description='This script prepares a promts for sentiment analysis for Llama 2')
   parser.add_argument('-i', '--input_fpath', required=False, default="/home/projects/DeepNeurOntology/IMDB_data/promt_eng.json")
   parser.add_argument('-o', '--output_folder', required=False, default="model_inputs/IMDB/")
   parser.add_argument('-f', '--output_fname', required=False, default=None)
   parser.add_argument('-s', '--shots', required=False, type=int, default=0)
   args = parser.parse_args()
   
   output_fname = args.output_fname
   if output_fname is None:
      output_fname = os.path.basename(args.input_fpath).split(".")[0]
   
   main(reiviews_fpath=args.input_fpath, output_folder = args.output_folder, output_fname = output_fname, shots = args.shots)
