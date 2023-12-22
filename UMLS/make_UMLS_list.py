import pandas as pd
import argparse, os
import random
import json
from tqdm import tqdm

def my_open(fpath,mode="w"):
   dirname=os.path.dirname(fpath)
   if len(dirname)>0 and not os.path.exists(dirname):
      os.makedirs(dirname)
   return open(fpath, mode)

def main(META_path, output_fpath):
   ### MRCONSO ###
   MRCONSO_fpath = os.path.join(META_path, "MRCONSO.RRF")
   print(f"Reading '{MRCONSO_fpath}'...")
   MRCONSO = pd.read_csv(MRCONSO_fpath, sep="|", names=["CUI","LAT","TS","LUI","STT","SUI","ISPREF","AUI","SAUI","SCUI","SDUI","SAB","TTY","CODE","STR","SRL","SUPPRESS","CVF"], index_col=False)
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
   MRREL = pd.read_csv(MRREL_fpath, sep="|", index_col=False, names=["CUI1","AUI1","STYPE1","REL","CUI2","AUI2","STYPE2","RELA","RUI","SRUI","SAB","SL","RG","DIR","SUPPRESS","CVF"])
   MRREL = MRREL[MRREL['CUI1'].isin(cui_tui_list) & MRREL['CUI2'].isin(cui_tui_list)]
   MRREL = MRREL[MRREL["REL"] == 'CHD']
   MRREL = MRREL[MRREL["CUI1"] != MRREL["CUI2"]]
   print(MRREL)

   print("Collecting CUIs not isolated in MRREL...")
   cui_list_all = list(set(MRREL['CUI1'].unique().tolist() + MRREL['CUI2'].unique().tolist()))
   print(f"{len(cui_list_all)} uniquie CUIs found.")

   ### Samplig Data ###
   with open('1000_regions_test_cui_list.txt') as f:
      test_cuis = f.read().splitlines()

   print("Discarding test CUIs...")
   cui_list_non_test = list(set(cui_list_all)-set(test_cuis))
   print(f"{len(cui_list_non_test)} CUIs left.")

   print("Sampling CUIs...")
   sampled_cuis = random.sample(cui_list_non_test, 10000)

   MRCONSO_sampled = MRCONSO[MRCONSO['CUI'].isin(sampled_cuis)].copy()
   abv_tty = ['AA', 'AB', 'ACR', 'AM', 'CA2', 'CA3', 'CDA', 'CS', 'DEV', 'DS', 'DSV', 'ES', 'HS', 'ID', 'MTH_ACR', 'NS', 'OAM', 'OA', 'OSN', 'PS', 'QAB', 'QEV', 'QSV', 'RAB', 'SSN', 'SS', 'VAB']
   MRCONSO_sampled = MRCONSO_sampled[~MRCONSO_sampled['TTY'].isin(abv_tty)]
   ret = {}
   for cui in tqdm(sampled_cuis):
      synonyms = [x.replace("&#x7C;", "|") for x in set(MRCONSO_sampled.loc[MRCONSO_sampled['CUI'] == cui]["STR"].str.lower().tolist())]
      ret[cui] = synonyms

   with open(output_fpath,mode="w") as f:
      f.write(json.dumps(ret))
 
if __name__ == '__main__':
   parser = argparse.ArgumentParser(prog='Sentiment promt maker', description='This script prepares a promts for sentiment analysis for Llama 2')
   parser.add_argument('-i', '--META_path', required=False, default="/home/projects/DeepNeurOntology/UMLS/2022AB/META/")
   parser.add_argument('-f', '--output_fpath', required=False, default="train_concepts.json")
   args = parser.parse_args()

   main(META_path=args.META_path, output_fpath = args.output_fpath)
