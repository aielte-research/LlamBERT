import fire
import os
import sys
import time

import json

from openai import OpenAI

from tqdm import trange

def my_open_w(fpath):
    if not os.path.exists(os.path.dirname(fpath)):
        os.makedirs(os.path.dirname(fpath))
    return open(fpath, 'w')

def main(
    model_name: str="gpt-4",
    prompt_file: str=None,
    target_file: str=None,
    **kwargs
):
    args = dict(locals())

    client = OpenAI()
    
    if prompt_file is not None:
        assert os.path.exists(
            prompt_file
        ), f"Provided Prompt file does not exist {prompt_file}"
        extension = os.path.splitext(prompt_file)[1].strip(".")
        if extension.lower() in ["json"]:
            with open(prompt_file, "r") as f:
                user_prompts = json.load(f)
                assert isinstance(user_prompts, list), "JSON content is not a list"
        elif extension.lower()=="txt":
            with open(prompt_file, "r") as f:
                user_prompt = "\n".join(f.readlines())
            user_prompts = [user_prompt]
        else:
            assert False, f"Error: unrecognized Prompt file extension '{extension}'!"

    elif not sys.stdin.isatty():
        user_prompt = "\n".join(sys.stdin.readlines())
        user_prompts = [user_prompt]
    else:
        print("No user prompt provided. Exiting.")
        sys.exit(1)
    
    questions=[]
    answers=[]
    answers_cleaned=[]
    answers_binary=[]
    # Process prompts in batches
    for i in trange(len(user_prompts)):
        if isinstance(user_prompts[i], list):
            messages = user_prompts[i]
        else:
            messages=[
                {"role": "user", "content": user_prompts[i]}
            ]

        #print(messages)
        try:
            response = client.chat.completions.create(
                model = model_name,
                messages = messages,
                stream=False,
            )
            answer = response.choices[0].message.content
        except Exception as error:
            print(error)
            print("Inference failed, will try again in a minute...")
            time.sleep(60)
            try:
                response = client.chat.completions.create(
                    model = model_name,
                    messages = messages,
                    stream=False,
                )
                answer = response.choices[0].message.content
            except Exception as error:
                print(error)
                print("Inference failed, will try again in a 10 minutes...")
                time.sleep(600)
                try:
                    response = client.chat.completions.create(
                        model = model_name,
                        messages = messages,
                        stream=False,
                    )
                    answer = response.choices[0].message.content
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
        assert os.path.exists(
            target_file
        ), f"Provided target file does not exist {target_file}"
        extension = os.path.splitext(target_file)[1].strip(".")
        if extension.lower() in ["json"]:
            with open(target_file, "r") as f:
                targets = json.load(f)
                assert isinstance(targets, list), "JSON content is not a list"
        else:
            assert False, f"Error: unrecognized target file extension '{extension}'!"
        
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
    with my_open_w(output_fpath) as file:
        json.dump(output_data, file, indent=4)
             

if __name__ == "__main__":
    fire.Fire(main)