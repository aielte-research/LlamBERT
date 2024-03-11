import pandas as pd
import os
import json
from tqdm import tqdm
import fire
import networkx as nx

def main(
   META_path: str="/home/projects/DeepNeurOntology/UMLS/2023AA/META/",
   concepts_fpath: str="regions/region_concepts_combined-microsoft-BiomedBERT_PubMed-filtered-counts.json",
   output_fpath: str="regions/region_concepts_graph_combined-microsoft-BiomedBERT_PubMed-filtered.gexf",
   **kwargs
):
   if len(kwargs) > 0:
      raise ValueError(f"Unknown argument(s): {kwargs}")
   
   print(f"Reading '{concepts_fpath}'...")
   with open(concepts_fpath) as f:
      concepts = json.loads(f.read())
   
   ### MRREL ###
   MRREL_fpath = os.path.join(META_path, "MRREL.RRF")
   print(f"Reading '{MRREL_fpath}'...")
   cols=["CUI1","AUI1","STYPE1","REL","CUI2","AUI2","STYPE2","RELA","RUI","SRUI","SAB","SL","RG","DIR","SUPPRESS","CVF"]
   dtypes={col_nam:"string" for col_nam in cols}
   MRREL = pd.read_csv(MRREL_fpath, sep="|", names=cols, dtype=dtypes, index_col=False)
   MRREL = MRREL[MRREL['CUI1'].isin(concepts.keys()) & MRREL['CUI2'].isin(concepts.keys())]
   MRREL = MRREL[MRREL["REL"] == 'CHD']
   MRREL = MRREL[MRREL["CUI1"] != MRREL["CUI2"]]
   print(MRREL)

   G = nx.DiGraph()
   G.add_nodes_from(concepts.keys())
   for _, row in tqdm(MRREL.iterrows(), total=len(MRREL)):
      G.add_edge(row['CUI1'], row['CUI2'])

   nx.write_gexf(G, output_fpath)

if __name__ == '__main__':
   fire.Fire(main)