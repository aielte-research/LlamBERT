import os, sys
import json
import fire
from typing import Union, List
import torch.nn as nn
from tqdm import trange
import numpy as np
import random
import torch
from openai import OpenAI
client = OpenAI()
import sys
sys.path.append("..")
from utils import my_open

use_cuda = torch.cuda.is_available()
def to_cuda(var):
    if use_cuda:
        return var.cuda()
    return var

def json_pretty_print(text, indent=4):
    level = 0
    list_level = 0
    inside_apostrophe = 0
    last_backslash_idx = -2
    ret = ""
    for i,c in enumerate(text):
        if c=="}" and inside_apostrophe % 2 == 0:
            level -= 1
            ret += "\n" + " "*(level*indent)
        ret += c
        if c=="{" and inside_apostrophe % 2 == 0:
            level += 1
            ret += "\n" + " "*(level*indent)
        elif c=="[" and inside_apostrophe % 2 == 0:
            list_level += 1
        elif c=="]" and inside_apostrophe % 2 == 0:
            list_level -= 1
        elif c=='"' and last_backslash_idx != i-1:
            inside_apostrophe += 1
        elif c=="\\":
            last_backslash_idx=i
        elif c=="," and inside_apostrophe % 2 == 0 and list_level<2:
            ret += "\n" + " "*(level*indent)
    return ret

def load_file(fpath):
    assert os.path.exists(fpath), f"Provided file does not exist '{fpath}'"
    
    extension = os.path.splitext(fpath)[1].strip(".")
    if extension.lower() in ["json"]:
        with open(fpath, "r") as f:
            ret = json.load(f)
        assert isinstance(ret, list), "JSON content is not a list"
    else:
        assert False, f"Error: unrecognized Prompt file extension '{extension}'!"

    return ret

class MLP(nn.Sequential):
    def __init__(
        self,
        channels: List[int]=[64,2],
        batch_norm: bool=False,
        dropout: Union[int, float]=0,
        bias: bool= True
    ):
        seq = []
        for i in range(len(channels)-2):
            seq.append(nn.Linear(*channels[i:i+2], bias = bias))
            seq.append(nn.ReLU())
            if dropout>0:
                seq.append(nn.Dropout(p = dropout))
            if batch_norm:
                seq.append(nn.BatchNorm1d(num_features = channels[i+1]))

        seq.append(nn.Linear(*channels[-2:], bias = bias))

        super().__init__(*seq)

def batch_iterator(data, batch_size, shuffle=True, cuda=True):
    if batch_size==0:
        starts=[0]
        batch_size=len(data)
    else:
        starts=np.arange(0, len(data), batch_size)
        if shuffle:
            random.shuffle(data)
            #np.random.shuffle(starts)
    
    for start in starts:
        batch = data[start:start+batch_size]
                    
        input_tensor = torch.Tensor([entry["embedding"] for entry in batch])
        #goal_tensor = torch.nn.functional.one_hot(torch.LongTensor([entry["label"] for entry in batch]))
        goal_tensor = torch.LongTensor([entry["label"] for entry in batch])
        
        if cuda:
            yield to_cuda(input_tensor), to_cuda(goal_tensor)
        else:
            yield input_tensor, goal_tensor

def classify(output):
    res=[]
    for seq in output:
        res.append(np.argmax(seq))
    return res 

def get_results(model, data, batch_size):
    answers_binary=[]
    correct=0
    model.eval()
    with torch.no_grad():
        for input_batch, goal_batch in batch_iterator(data, batch_size, cuda=False, shuffle=False):
            output_batch = model(to_cuda(input_batch)).detach().cpu().numpy()
            classified = classify(output_batch)
            answers_binary += classified
            correct+=sum(1 for a, b in zip(classified, goal_batch.detach().cpu().numpy()) if a == b)
    return {
        "output_stats": {
            "negative": len([x for x in answers_binary if x==0]),
            "positive": len([x for x in answers_binary if x==1]),
        },
        "accuracy": int(100000*correct/len(data))/1000
    }

class RunningAverage():
    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        if self.steps==0:
            return 0
        else:
            return self.total / float(self.steps)
        

def main(
    train_file: str,
    test_file: Union[None, str]=None,
    output_path: str="model_outputs/IMDB",
    MLP_layers: List[int]=[512,128,32,2],
    batch_size: int=250,
    epochs: int=10,
    learning_rate: float=0.0005,
    dropout: float=0.75,
):
    if train_file is None:
        print("No file with train embeddings provided (option --promt-file). Exiting.")
        sys.exit(1)

    train_data = load_file(train_file)
    embedding_dims = set([len(entry["embedding"]) for entry in train_data])

    if test_file is not None:
        test_data = load_file(test_file)
        embedding_dims |= set([len(entry["embedding"]) for entry in test_data])
    
    assert len(embedding_dims)==1, f"Error: Not all embedding vetors are the same size! Sizes: '{embedding_dims}'!"

    embedding_dim = list(embedding_dims)[0]

    model = to_cuda(MLP([embedding_dim]+MLP_layers,dropout=dropout,batch_norm=False))
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    lossfn = torch.nn.CrossEntropyLoss()
    
    pbar = trange(epochs)
    train_accuracy=0
    if test_file is not None:
        test_accuracy=0
        max_test_accuracy=0
    for i in pbar:
        model.train()
        loss_avg = RunningAverage()
        for input_batch, goal_batch in batch_iterator(train_data, batch_size, shuffle=True):
            output_batch = model(input_batch)
            loss = lossfn(output_batch, goal_batch)
            loss_avg.update(float(loss.mean().item()))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if test_file is not None:
                pbar.set_postfix(
                    train_loss = '{:.6f}'.format(loss_avg()),
                    train_accuracy = '{:05.3f}%'.format(train_accuracy),
                    test_accuracy = '{:05.3f}%'.format(test_accuracy)
                )
            else:
                pbar.set_postfix(
                    train_loss = '{:.6f}'.format(loss_avg()),
                    train_accuracy = '{:05.3f}%'.format(train_accuracy)
                )
            

        train_eval_results = get_results(model, train_data, batch_size*100)
        train_accuracy = train_eval_results["accuracy"]
        if test_file is not None:
            test_eval_results = get_results(model, test_data, batch_size*100)
            test_accuracy = test_eval_results["accuracy"]
            max_test_accuracy = max(max_test_accuracy, test_accuracy)

            pbar.set_postfix(
                train_loss = '{:.6f}'.format(loss_avg()),
                train_accuracy = '{:05.3f}%'.format(train_accuracy),
                test_accuracy = '{:05.3f}%'.format(test_accuracy),
                max_test_accuracy = '{:05.3f}%'.format(max_test_accuracy)
            )
        else:
            pbar.set_postfix(
                train_loss = '{:.6f}'.format(loss_avg()),
                train_accuracy = '{:05.3f}%'.format(train_accuracy)
            )
        
    # output_fpath = os.path.join(output_path,f'{os.path.splitext(os.path.basename(test_file))[0].strip(".")}_{os.path.basename(test_file.rstrip("/"))}.json')
    # with my_open(output_fpath) as outfile:
    #     outfile.write(json_pretty_print(json.dumps(user_prompts,separators=(',', ': '))))
    #     json.dump(user_prompts, file, indent=4)

if __name__ == "__main__":
    fire.Fire(main)