`llama_inference.py`:

<!-- USAGE EXAMPLES -->

## Downloading weigths

```sh
bash <(curl -sSL https://g.bodaay.io/hfd) -m "MODEL PATH ON HUGGINGFACE"
```

## Usage
Assuming downloaded Llama 2 weights.

Full llama-2
```sh
   time CUDA_VISIBLE_DEVICES=0,1,2,3 python llama_inference.py --model-name /home/projects/llama/meta-llama_Llama-2-70b-chat-hf/ --prompt_file model_inputs/IMDB/promt_eng_0-shot_prompts.json --target_file model_inputs/IMDB/promt_eng_0-shot_labels.json --max-new-tokens 5 --batch-size 1
```

7B llama-2
```sh
   time CUDA_VISIBLE_DEVICES=0 python llama_inference.py --model-name /home/projects/llama/meta-llama_Llama-2-7b-chat-hf/ --prompt_file model_inputs/IMDB/promt_eng_0-shot_prompts.json --target_file model_inputs/IMDB/promt_eng_0-shot_labels.json --max-new-tokens 5 --batch-size 1
```