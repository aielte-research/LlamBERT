# LlamBERT
<p align="center">
  <img src="./plots/LlamBERT_DALL-E-3_gen.png" alt="Model Architecture">
</p>

Efficiently extracting subontologies from the UMLS graph using natural language queries.

## Project Description
LlamBERT is a Python library that implements a semi-supervised approach for text classification using BERT-based models. 
This repository contains the code implementation for the method described in the research paper titled LlamBERT: Leveraging Semi-Supervised Learning for Text Classification.

Method Overview
Given a large corpus of unlabeled natural language data, LlamBERT follows these steps:

1. Annotation: Annotate a subset of the corpus using Llama,2 and a natural language prompt.
2. Parsing: Parse Llama,2 responses into desired categories.
3. Data Filtering: Discard data not classifiable into specified categories.
4. Fine-tuning: Perform supervised fine-tuning on a BERT classifier using resulting labels.
5. Annotation: Apply the fine-tuned BERT classifier to annotate the original unlabeled corpus.


## Hardware Dependencies
+ **Llama-2-7b-chat:** Requires a single A100 40GB GPU.
+ **Llama-2-70b-chat:** Requires four A100 80GB GPUs
+ **gpt-4-0613:** Requires no GPU.

## How to use LLamBERT

### Install environment
```
conda env create --file=environment.yml
```

### Generating labels with LLamBERT
```
insert code here
```

### Generating labels with ChatGPT
```
insert code here
```

## Citation
If you use this code in your research, please cite the corresponding paper:

InsertCitationHere

### Contributors
- Bálint Csanády (csbalint@protonmail.ch)
- Lajos Muzsai (muzsailajos@protonmail.com)
- Péter Vedres (vedrespeter0000@gmail.com)
- Zoltán Nádasdy (zoltan@utexas.edu)
- András Lukács (andras.lukacs@ttk.elte.hu)

## License
This project is licensed under the MIT License - see the LICENSE file for details.
