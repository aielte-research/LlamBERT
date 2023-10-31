<!-- USAGE EXAMPLES -->
## Usage

Full llama-2
```sh
   time CUDA_VISIBLE_DEVICES=4,5,6,7 python llama_inference.py --model-name /home/projects/llama/meta-llama_Llama-2-70b-chat-hf/ --prompt_file model_inputs/cui_formatted_llama.json --max-new-tokens 1000 --batch-size 1
```

Quantized llama-2
```sh
   time CUDA_VISIBLE_DEVICES=3 python llama_inference.py --model-name /home/projects/llama/TheBloke_Platypus2-70B-Instruct-GPTQ/ --prompt_file model_inputs/cui_formatted_llama.json --max-new-tokens 1000 --batch-size 1
```