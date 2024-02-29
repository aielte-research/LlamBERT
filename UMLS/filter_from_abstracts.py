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

def filter_by_lang(lang,df):
    start_time = time()
    df_len_save = len(df)
    logger.info(f"Filtering {human_format(len(df))} abstracts by language '{lang}'...")
    ret = df.loc[df['Language'] == lang]
    logger.info(f"{human_format(len(ret))} abstracts kept, {human_format(df_len_save - len(ret))} discarded. {getTime(time() - start_time)}")
    return ret

def filter_by_kw_aho_(df,automaton,concepts,strict=True):
    found=[]
    for abst in tqdm(df['Abstract']):
        found_match=True
        while found_match:
            found_match=False
            for end_index, (cui, original_value) in automaton.iter(abst):
                if not strict or re.search(r'\b' + re.escape(original_value) + r'\b',abst):
                    found_match=True  # found a match
                    for syn in concepts[cui]:
                        automaton.remove_word(syn)
                    automaton.make_automaton()
                    found.append(cui)
                    print(f"Found '{original_value}' (cui:{cui}). Found in chunk: {len(found)}")
                    #input()
                    break

    return found,automaton

def filter_by_kw_aho(df,automaton,concepts,strict=True):
    found=[]
    for abst in tqdm(df['Abstract']):
        found_cuis=[]
        for end_index, (cui, original_value) in automaton.iter(abst):
            if (not strict or re.search(r'\b' + re.escape(original_value) + r'\b',abst)) and not cui in found_cuis:
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

def load_chunk(
        csv_path: str,
        automaton,
        concepts,
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
    found_cuis, automaton = filter_by_kw_aho(df, automaton, concepts)
    logger.info(f"{len(found_cuis)} concepts found. {getTime(time() - start_time)}")
    return found_cuis, automaton

def main(
        csv_path: str="/home/projects/Brainvectors/PubMed/2024/pubmed_abstracts/",
        concepts_fpath: str="regions/region_concepts_combined-microsoft-BiomedBERT.json",
        output_path: str="regions",
        loglevel: int=20,
        language: str="eng"
    ):
    logger.setLevel(loglevel)
    streamHandler = logging.StreamHandler()
    logger.addHandler(streamHandler)
    streamHandler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"))
    logger.propagate = False

    overall_start_time = time()

    # load concepts
    with open(concepts_fpath) as f:
        concepts = json.loads(f.read())
    
    if os.path.isdir(csv_path):
        file_names = sorted(os.listdir(csv_path))
        csv_fpaths = [os.path.join(csv_path, file_name) for file_name in file_names if ".csv" in file_name]
    else:
        csv_fpaths = [csv_path]

    automaton = ahocorasick.Automaton()    
    for cui, syns in concepts.items():
        for syn in syns:
            if len(syn)>2:
                automaton.add_word(syn, (cui, syn))    
    automaton.make_automaton()
    found_cuis=[]
    for i, csv_fpath in enumerate(csv_fpaths):
        found, automaton = load_chunk(csv_fpath, automaton, concepts, index=i+1, language=language)
        found_cuis+=found

    logger.info(f"{len(found_cuis)} concepts found out of {len(concepts)}.")

    output_fname = f'{os.path.basename(concepts_fpath).split(".")[0]}_PubMed-filtered.json'
    with my_open(os.path.join(output_path,output_fname), 'w') as outfile:
        json.dump({cui:syns for cui,syns in concepts.items() if cui in found_cuis}, outfile, indent=3)

    logger.info(f"Data prapared. Overall time: {getTime(time() - overall_start_time)}")

# parse command line arguments
if __name__ == "__main__":    
    fire.Fire(main)