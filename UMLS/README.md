Code for preparing UMLS data for LlamBERT, and filtering the concepts with LlamBERT and appearance in PubMed abstracts.

## Requirements
`make_UMLS_list.py`:

- Downloaded UMLS archive: https://www.nlm.nih.gov/research/umls/licensedcontent/umlsarchives04.html

`get_filtered_concepts.py`:

- UMLS concepts
- LlamBERT concept annotation labels 

`filter_from_abstracts.py`:

- UMLS concepts filtered by LlamBERT
- PubMed abstracts: https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/
  converted to csv.

`get_concept_graph.py`:

- UMLS concepts filtered by LlamBERT
- Downloaded UMLS archive: https://www.nlm.nih.gov/research/umls/licensedcontent/umlsarchives04.html
