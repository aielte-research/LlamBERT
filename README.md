# LlamBERT
<p align="center">
  <img src="./plots/LlamBERT_DALL-E-3.png" alt="Model Architecture">
</p>

## Project Description
LlamBERT implements a semi-supervised approach for text classification using BERT-based models. 
We use it for efficiently extracting subontologies from the UMLS graph using natural language queries.
This repository contains the code implementation for the method described in the research paper titled LlamBERT: Leveraging Semi-Supervised Learning for Text Classification.

Method Overview
Given a large corpus of unlabeled natural language data, LlamBERT follows these steps:

1. Annotation: Annotate a subset of the corpus using Llama,2 and a natural language prompt.
2. Parsing: Parse Llama,2 responses into desired categories.
3. Data Filtering: Discard data not classifiable into specified categories.
4. Fine-tuning: Perform supervised fine-tuning on a BERT classifier using resulting labels.
5. Annotation: Apply the fine-tuned BERT classifier to annotate the original unlabeled corpus.

### Comparison BERT test accuracies on the IMDb data.

| BERT model    | Baseline train | LlamBERT train | LlamBERT train&extra | Combined extra+train |
|---------------|----------------|----------------|----------------------|----------------------|
| distilbert-base | 91.23         | 90.77         | 92.12               | **92.53**           |
| bert-base       | 92.35         | 91.58         | 92.76               | **93.47**           |
| bert-large      | 94.29         | 93.31         | 94.07               | **95.03**           |
| roberta-base    | 94.74         | 93.53         | 94.28               | **95.23**           |
| roberta-large   | 96.54         | 94.83         | 94.98               | **96.68**           |

## Hardware Dependencies
+ **Llama-2-7b-chat:** Requires a single A100 40GB GPU.
+ **Llama-2-70b-chat:** Requires four A100 80GB GPUs
+ **gpt-4-0613:** Requires no OpenAI API access.

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
