import pandas as pd
from time import time
import re
import os
import fire
import json
import logging
logger = logging.getLogger("data_preproc")
import sys
sys.path.append('../')
from utils import getTime, human_format, my_open
import ahocorasick
from tqdm import tqdm
from collections import defaultdict 

def filter_by_lang(lang,df):
    start_time = time()
    df_len_save = len(df)
    logger.info(f"Filtering {human_format(len(df))} abstracts by language '{lang}'...")
    ret = df.loc[df['Language'] == lang]
    logger.info(f"{human_format(len(ret))} abstracts kept, {human_format(df_len_save - len(ret))} discarded. {getTime(time() - start_time)}")
    return ret

def filter_by_kw_aho_fast(df,automaton,concepts):
    found=[]
    for abst in tqdm(df['Abstract']):
        found_cuis=[]
        for _, (cui, original_value) in automaton.iter(abst):
            if re.search(r'\b' + re.escape(original_value) + r'\b',abst) and not cui in found_cuis:
                found_cuis.append(cui)
                print(f"Found '{original_value}' (cui:{cui})")
        if len(found_cuis)>0:
            for cui in found_cuis:
                for syn in concepts[cui]:
                    automaton.remove_word(syn)
            automaton.make_automaton()
            found+=found_cuis
            print(f"Found in chunk: {len(found)}")

    return found,automaton

def filter_by_kw_aho(df,automaton,concepts,syn2cui):
    for abst in tqdm(df['Abstract']):
        found_cuis=set()
        found_syns=set()
        for _, (cui, syn) in automaton.iter(abst):
            if re.search(r'\b' + re.escape(syn) + r'\b',abst) and not syn in found_syns:
                for dup_cui in syn2cui[syn]:
                    found_cuis.add(dup_cui)
                    concepts[dup_cui]["synonyms"][syn]+=1
                found_syns.add(syn)
        if len(found_cuis)>0:
            for cui in found_cuis:
                concepts[cui]["count"]+=1

    return concepts,automaton

def load_chunk(
        csv_path: str,
        automaton,
        concepts,
        syn2cui,
        index: int,
        language: str="eng",
    ):
    # load csv
    start_time = time()
    logger.info(f"Loading '{csv_path}'...")
    df = pd.read_csv(csv_path,dtype = {'Language': "string", 'Year ': int, 'Title': "string", 'Abstract': "string"})
    logger.info(f"{human_format(len(df))} abstracts loaded in chunk {index:02d}. {getTime(time() - start_time)}")

    # Duplicate the title column, so we can keep the original values (An idea to have better pairs is to look for pairs in the og titles)
    df['Title_original'] = df['Title']

    # lang filter
    df = filter_by_lang(language,df)

    # making the abstracts lowercase
    start_time = time()
    logger.info(f"Making the abstracts lowercase in chunk {index:02d}...")
    df['Abstract']=df['Abstract'].str.lower()
    #df['Title']=df['Title'].fillna("")
    df['Title']=df['Title'].str.lower()
    logger.info(f"Done. {getTime(time() - start_time)}")
        
    # filter by concepts
    start_time = time()
    logger.info(f"Searching {len(concepts)} concepts in {human_format(len(df))} abstracts in chunk {index:02d}...")
    concepts, automaton = filter_by_kw_aho(df, automaton, concepts, syn2cui)

    return concepts, automaton

def get_syn2cui(concepts):
    ret = defaultdict(lambda:[])
    for cui, syns in concepts.items():
        for syn in syns:
            ret[syn].append(cui)
    return dict(ret)

def main(
        csv_path: str="/home/projects/Brainvectors/PubMed/2024/pubmed_abstracts/",
        concepts_fpath: str="regions/region_concepts_combined-microsoft-BiomedBERT.json",
        output_path: str="regions",
        loglevel: int=20,
        language: str="eng",
        **kwargs
    ):
    if len(kwargs) > 0:
        raise ValueError(f"Unknown argument(s): {kwargs}")
    logger.setLevel(loglevel)
    streamHandler = logging.StreamHandler()
    logger.addHandler(streamHandler)
    streamHandler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"))
    logger.propagate = False

    overall_start_time = time()

    # load concepts
    with open(concepts_fpath) as f:
        concepts = json.loads(f.read())

    syn2cui = get_syn2cui(concepts)
    output_fname = f'{os.path.basename(concepts_fpath).split(".")[0]}_duplicates.json'
    with my_open(os.path.join(output_path,output_fname), 'w') as outfile:
        json.dump({syn:cuis for syn,cuis in syn2cui.items() if len(cuis)>1}, outfile, indent=3)

    automaton = ahocorasick.Automaton()    
    for cui, entry in concepts.items():
        for syn in entry:
            if len(syn)>2:
                automaton.add_word(syn, (cui, syn))    
    automaton.make_automaton()

    concepts = {cui: {"count":0,"synonyms":{s:0 for s in syns}} for cui, syns in concepts.items()}
    
    if os.path.isdir(csv_path):
        file_names = sorted(os.listdir(csv_path))
        csv_fpaths = [os.path.join(csv_path, file_name) for file_name in file_names if ".csv" in file_name]
    else:
        csv_fpaths = [csv_path]

    for i, csv_fpath in enumerate(csv_fpaths):
        concepts, automaton = load_chunk(csv_fpath, automaton, concepts, syn2cui, index=i+1, language=language)
    
    concepts_res = {cui:entry for cui,entry in concepts.items() if entry["count"]>0}
    logger.info(f"{len(concepts_res)} concepts kept out of {len(concepts)}.")
    output_fname = f'{os.path.basename(concepts_fpath).split(".")[0]}_PubMed-filtered-counts.json'
    with my_open(os.path.join(output_path,output_fname), 'w') as outfile:
        json.dump(concepts_res, outfile, indent=3)

    logger.info(f"Data prapared. Overall time: {getTime(time() - overall_start_time)}")

# parse command line arguments
if __name__ == "__main__":    
    fire.Fire(main)