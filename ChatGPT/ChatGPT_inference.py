import fire
import os
import sys
import time
from typing import Union
import json
from openai import OpenAI
from tqdm import trange
import sys
sys.path.append("..")
from utils import my_open, read_json_list

def main(
   prompt_file: str,
   model_name: str="gpt-4",    
   target_file: Union[str, None]=None,
   **kwargs
):
   if len(kwargs) > 0:
      raise ValueError(f"Unknown argument(s): {kwargs}")
   
   args = dict(locals())
   client = OpenAI()
   
   user_prompts = read_json_list(prompt_file)

   if target_file is not None:
      targets = read_json_list(target_file)

   questions=[]
   answers=[]
   answers_cleaned=[]
   answers_binary=[]
   # Process prompts in batches
   for i in trange(len(user_prompts)):
      if isinstance(user_prompts[i], list):
         messages:list[dict] = user_prompts[i]
      else:
         messages=[
            {"role": "user", "content": user_prompts[i]}
         ]

      get_answer = lambda messages: client.chat.completions.create(model=model_name, messages = messages, stream=False).choices[0].message.content
      try:
         answer = get_answer(messages)
         if answer is None:
            raise
      except Exception as error:
         print(error)
         print("Inference failed, will try again in a minute...")
         time.sleep(60)
         try:
            answer = get_answer(messages)
            if answer is None:
               raise
         except Exception as error:
            print(error)
            print("Inference failed, will try again in a 10 minutes...")
            time.sleep(600)
            try:
               answer = get_answer(messages)
               if answer is None:
                  raise
            except Exception as error:
               print(error)
               print(f"Inference failed again, terminating session at prompt {i}...")
               break
      #print(answer)

      questions.append(messages[-1]["content"])
      answers.append(answer)
      output_cleaned = answer.strip().strip(".,!?;'\"").strip().strip(".,!?;'\"").lower()
      answers_cleaned.append(output_cleaned)
      if output_cleaned == "no":
         answers_binary.append(0)
      elif output_cleaned == "yes":
         answers_binary.append(1)
      elif "negative" in output_cleaned and "positive" not in output_cleaned:
         answers_binary.append(0)
      elif "positive" in output_cleaned and "negative" not in output_cleaned:
         answers_binary.append(1)
      else:
         answers_binary.append(2)

   output_data={
      "settings": args,
      "outputs": answers,
      "outputs_cleaned": answers_cleaned,
      "outputs_binary": answers_binary,
      "output_stats": {
         "negative": len([x for x in answers_binary if x==0]),
         "positive": len([x for x in answers_binary if x==1]),
         "other": len([x for x in answers_binary if x==2])
      }
   }

   if target_file is not None:        
      correct=0
      misclassifed=[]
      for target, output, question in zip(targets[:len(answers_binary)], answers_binary, questions):
         if target == output:
            correct+=1
         else:
            misclassifed.append({"question":question,"answer": output})
      output_data["accuracy"]=int(100000*correct/len(targets))/1000
      print(f"Accuracy: {output_data['accuracy']}%")
      output_data["misclassifed"]=misclassifed
   
   output_fpath = os.path.join("model_outputs",f'{os.path.splitext(os.path.basename(prompt_file))[0].strip(".")}_{os.path.basename(model_name.rstrip("/"))}.json')
   with my_open(output_fpath) as file:
      json.dump(output_data, file, indent=4)
             

if __name__ == "__main__":
   fire.Fire(main)