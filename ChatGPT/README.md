<!-- USAGE EXAMPLES -->

## Environment variables

```sh
export OPENAI_API_KEY="<Your personal api token>"
```

## Usage

Full llama-2
```sh
   time python ChatGPT4_inference.py --model-name /home/projects/llama/meta-llama_Llama-2-70b-chat-hf/ --prompt_file model_inputs/IMDB/promt_eng_0-shot_prompts.json --target_file model_inputs/IMDB/promt_eng_0-shot_labels.json --max-new-tokens 5 --batch-size 1
```