# LlamBERT
<p align="center">
  <img src="./plots/LlamBERT_DALL-E-3.png" alt="Model Architecture">
</p>

## Project Description
LlamBERT implements a hybrid approach approach for text classification that leverages LLMs to annotate a small subset of large, unlabeled databases and uses the results for fine-tuning transformer encoders like BERT and RoBERTa. 
This strategy is evaluated on two diverse datasets: the IMDb review dataset and the UMLS Meta-Thesaurus.
We use it for efficiently extracting subontologies from the UMLS graph using natural language queries.
This repository implements the method described in the research paper titled LlamBERT: Leveraging Semi-Supervised Learning for Text Classification.

Method Overview
Given a large corpus of unlabeled natural language data, LlamBERT follows these steps:

1. Annotate a reasonably sized, randomly selected subset of the corpus utilizing an LLM and a prompt reflecting the labeling criteria;
2. Parse the Llama 2 responses into the desired categories;
3. Discard any data that fails to classify into any of the specified categories;
4. Employ the resulting labels to perform supervised fine-tuning on a BERT classifier;
5. Apply the fine-tuned BERT classifier to annotate the original unlabeled corpus.

### Comparison BERT test accuracies on the IMDb data.

| BERT model    | Baseline train | LlamBERT train | LlamBERT train&extra | Combined extra+train |
|---------------|----------------|----------------|----------------------|----------------------|
| distilbert-base | 91.23         | 90.77         | 92.12               | **92.53**           |
| bert-base       | 92.35         | 91.58         | 92.76               | **93.47**           |
| bert-large      | 94.29         | 93.31         | 94.07               | **95.03**           |
| roberta-base    | 94.74         | 93.53         | 94.28               | **95.23**           |
| roberta-large   | 96.54         | 94.83         | 94.98               | **96.68**           |

### Accuracy comparison of different training data for the UMLS classification
95th percentile confidence interval measured on 5 different random seeds.

| Model            | Baseline         | LlamBERT        | Combined        |
|------------------|------------------|-----------------|-----------------|
| bert-large       | 94.84 (±0.25)    | 95.70 (±0.21)   | 96.14 (±0.42)   |
| roberta-large    | 95.00 (±0.18)    | 96.02 (±0.12)   | 96.64 (±0.14)   |
| BiomedBERT-large | 96.72 (±0.17)    | 96.66 (±0.13)   | 96.92 (±0.10)   |

## Hardware Requirements
+ **Llama-2-7b-chat:** Requires a single A100 40GB GPU.
+ **Llama-2-70b-chat:** Requires four A100 80GB GPUs
+ **gpt-4-0613:** Requires OpenAI API access.

## Installation
```
conda env create --file=environment.yml
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
