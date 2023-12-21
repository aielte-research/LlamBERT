import argparse, os
import json

def my_open(fpath,mode="w"):
   dirname=os.path.dirname(fpath)
   if len(dirname)>0 and not os.path.exists(dirname):
      os.makedirs(dirname)
   return open(fpath, mode)

def main(reiviews_fpath, output_folder, output_fname):
   with open(reiviews_fpath) as f:
      reviews = json.loads(f.read())
   
   prompts=[]
   labels=[]
   for rev in reviews:
      prompts.append(f"[INST] <<SYS>>\nPlease answer with 'positive' or 'negative' only!\n<</SYS>>\n Does the following movie review have a positive sentiment? If the movie review is positive please answer 'positive', if movie review is negative please anwer 'negative'! Make your decision based on th whole text! \n{rev['txt']}[/INST]\n")
      if "label" in rev:
         labels.append(rev["label"])
   
   with my_open(os.path.join(output_folder, f"{output_fname}_prompts.json"), 'w') as outfile:
      json.dump(prompts, outfile)#, indent=4)

   if labels!=[]:
      with my_open(os.path.join(output_folder, f"{output_fname}_labels.json"), 'w') as outfile:
         json.dump(labels, outfile)#, indent=4)

if __name__ == '__main__':
   parser = argparse.ArgumentParser(prog='Sentiment promt maker', description='This script prepares a promts for sentiment analysis for Llama 2')
   parser.add_argument('-i', '--input_fpath', required=False, default="/home/projects/DeepNeurOntology/IMDB_data/promt_eng.json")
   parser.add_argument('-o', '--output_folder', required=False, default="model_inputs/IMDB/")
   parser.add_argument('-f', '--output_fname', required=False, default=None)
   args = parser.parse_args()
   
   output_fname = args.output_fname
   if output_fname is None:
      output_fname = os.path.basename(args.input_fpath).split(".")[0]
   
   main(reiviews_fpath=args.input_fpath, output_folder = args.output_folder, output_fname = output_fname)
