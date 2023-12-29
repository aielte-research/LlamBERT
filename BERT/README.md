<!-- USAGE EXAMPLES -->

## Environment variables

```sh
export NEPTUNE_API_TOKEN="<Your personal api token>"
export NEPTUNE_PROJECT="aielte/DNeurOn"
export TOKENIZERS_PARALLELISM="false"
```

## Usage

```sh
   time CUDA_VISIBLE_DEVICES=6 python bert_finetune.py -c conf/UMLS_region_10k_quicktest.yaml 
```
