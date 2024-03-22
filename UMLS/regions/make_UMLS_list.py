import pandas as pd
import os
import random
import json
from tqdm import tqdm
import fire

import sys
sys.path.append("../..")
from utils import my_open

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
   META_path: str = "/home/projects/DeepNeurOntology/UMLS/2023AA/META/",
   seed: int = 42,
   **kwargs
):
   if len(kwargs) > 0:
      raise ValueError(f"Unknown argument(s): {kwargs}")
   random.seed(seed)
   
   ### MRCONSO ###
   MRCONSO_fpath = os.path.join(META_path, "MRCONSO.RRF")
   print(f"Reading '{MRCONSO_fpath}'...")
   cols=["CUI","LAT","TS","LUI","STT","SUI","ISPREF","AUI","SAUI","SCUI","SDUI","SAB","TTY","CODE","STR","SRL","SUPPRESS","CVF"]
   dtypes={col_nam:"string" for col_nam in cols}
   MRCONSO = pd.read_csv(MRCONSO_fpath, sep="|", names=cols, dtype=dtypes, index_col=False)
   MRCONSO = MRCONSO[MRCONSO["LAT"] == 'ENG']
   print(MRCONSO)

   print("Collecting unique CUIs...")
   cui_list = MRCONSO['CUI'].unique().tolist()
   print(f"{len(cui_list)} uniquie CUIs found.")

   ### MRSTY ###
   MRSTY_fpath = os.path.join(META_path, "MRSTY.RRF")
   print(f"Reading '{MRSTY_fpath}'...")
   MRSTY = pd.read_csv(MRSTY_fpath, sep='|', header=None, names=['CUI', 'TUI', 'STN', 'STY', 'ATUI', 'CVF'], index_col=False)
   MRSTY = MRSTY[MRSTY['CUI'].isin(cui_list)]
   print(MRSTY)

   desired_tui_values = [
      'T017', 'T021', 'T022', 'T023', 'T024', 'T025',
      'T026', 'T029', 'T030', 'T031', 'T125', 'T192'
   ]

   print(f"Filtering MRSTY using TUI list {desired_tui_values}...")
   MRSTY = MRSTY[MRSTY['TUI'].isin(desired_tui_values)]
   print(MRSTY)

   print("Collecting unique CUIs in filtered MRSTY...")
   # Convert the unique CUI values to a Python list
   cui_tui_list = list(MRSTY['CUI'].unique())
   print(f"{len(cui_tui_list)} uniquie CUIs found.")
   
   ### MRREL ###
   MRREL_fpath = os.path.join(META_path, "MRREL.RRF")
   print(f"Reading '{MRREL_fpath}'...")
   cols=["CUI1","AUI1","STYPE1","REL","CUI2","AUI2","STYPE2","RELA","RUI","SRUI","SAB","SL","RG","DIR","SUPPRESS","CVF"]
   dtypes={col_nam:"string" for col_nam in cols}
   MRREL = pd.read_csv(MRREL_fpath, sep="|", names=cols, dtype=dtypes, index_col=False)
   MRREL = MRREL[MRREL['CUI1'].isin(cui_tui_list) & MRREL['CUI2'].isin(cui_tui_list)]
   MRREL = MRREL[MRREL["REL"] == 'CHD']
   MRREL = MRREL[MRREL["CUI1"] != MRREL["CUI2"]]
   print(MRREL)

   print("Collecting CUIs not isolated in MRREL...")
   cui_list_all = list(set(MRREL['CUI1'].unique().tolist() + MRREL['CUI2'].unique().tolist()))
   print(f"{len(cui_list_all)} uniquie CUIs found.")

   export(cui_list_all, MRCONSO, "all_concepts.json")

   ### Samplig Data ###
   with open('test_regions_cui_list.txt') as f:
      test_cuis = f.read().splitlines()
   
   export(test_cuis, MRCONSO, "test_concepts.json")

   with open('train_gold_regions_cui_list.txt') as f:
      train_gold_cuis = f.read().splitlines()
   
   export(train_gold_cuis, MRCONSO, "train_gold_concepts.json")

   print("Discarding test and train_gold CUIs...")
   cui_list_non_test = list(set(cui_list_all)-set(test_cuis)-set(train_gold_cuis))
   print(f"{len(cui_list_non_test)} CUIs left.")

   print("Sampling CUIs...")
   random.shuffle(cui_list_non_test)
   sampled_cuis = cui_list_non_test[:10000]

   export(sampled_cuis, MRCONSO, "train_concepts.json")
 
if __name__ == '__main__':
   fire.Fire(main)