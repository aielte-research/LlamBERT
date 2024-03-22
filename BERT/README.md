<!-- USAGE EXAMPLES -->

## Environment variables

With `neptune.ai` logging:
```sh
export NEPTUNE_API_TOKEN="<Your personal api token>"
export NEPTUNE_PROJECT="<Neptune project ID>"
export TOKENIZERS_PARALLELISM="false"
```

Without `neptune.ai` logging:
```sh
export TOKENIZERS_PARALLELISM="false"
```

## Usage

```sh
   time CUDA_VISIBLE_DEVICES=0 python bert_finetune.py -c conf/UMLS/region_10k_quicktest.yaml 
```
