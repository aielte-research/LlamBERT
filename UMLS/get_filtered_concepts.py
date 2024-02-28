import os
import json
import fire
import sys
sys.path.append("..")
from utils import my_open

def main(
      concepts_fpath: str="/home/projects/DeepNeurOntology/UMLS/regions/all_concepts.json",
      labels_fpath: str="/home/projects/DeepNeurOntology/UMLS/regions/all_concepts_labels_Llama-10k-microsoft-BiomedBERT.json"
   ):
   with open(concepts_fpath) as f:
      concepts = json.loads(f.read())
   
   with open(labels_fpath) as f:
      labels = json.loads(f.read())

   assert(len(concepts)==len(labels))

   filtered = {}
   kept, excluded = 0, 0
   for (cui, syns), label in zip(concepts.items(), labels):
      if label == 0:
         excluded+=1
      elif label == 1:
         kept+=1
         filtered[cui]=syns
      else:
         print(f"Unexpected label: {label}")
         raise
   print(f"{kept} concepts kept and {excluded} concepts excluded form {len(concepts)}.")

   output_fname = f'{os.path.basename(concepts_fpath).split(".")[0]}_filtered.json'
   with my_open(output_fname, 'w') as outfile:
      json.dump(filtered, outfile, indent=3)
 
if __name__ == '__main__':
   fire.Fire(main)