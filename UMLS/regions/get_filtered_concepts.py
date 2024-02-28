import pandas as pd
import os
import random
import json
from tqdm import tqdm
import fire

def my_open(fpath,mode="w"):
   dirname=os.path.dirname(fpath)
   if len(dirname)>0 and not os.path.exists(dirname):
      os.makedirs(dirname)
   return open(fpath, mode)

def export(cuis, MRCONSO, output_fpath):
   MRCONSO_sampled = MRCONSO[MRCONSO['CUI'].isin(cuis)].copy()
   abv_tty = ['AA', 'AB', 'ACR', 'AM', 'CA2', 'CA3', 'CDA', 'CS', 'DEV', 'DS', 'DSV', 'ES', 'HS', 'ID', 'MTH_ACR', 'NS', 'OAM', 'OA', 'OSN', 'PS', 'QAB', 'QEV', 'QSV', 'RAB', 'SSN', 'SS', 'VAB']
   MRCONSO_sampled = MRCONSO_sampled[~MRCONSO_sampled['TTY'].isin(abv_tty)]
   ret = {}
   for cui in tqdm(cuis):
      synonyms = sorted(sorted(list(set(MRCONSO_sampled.loc[MRCONSO_sampled['CUI'] == cui]["STR"].str.lower().tolist()))), key=len)
      ret[cui] = synonyms

   with my_open(output_fpath,mode="w") as f:
      f.write(json.dumps(ret, indent=3).replace("&#x7c;", "|"))

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