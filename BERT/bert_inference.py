from transformers import BertTokenizer, BertModel
import torch
import json
from pprint import pprint
import re
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import argparse


def load_data_from_json(path):
    with open(path, "r") as file:
        data = json.load(file)["outputs_full"]

    formatted_data = []
    for line in data:
        line = line.replace("\n", "").split("following meanings")[-1]
        concepts = re.findall(r"'(.*?)'", line)
        if len(concepts) == 0:
            print(line)
        if "yes" in line.lower()[-5:]:
            for concept in concepts:
                formatted_data.append([concept, 1])
        elif "no" in line.lower()[-5:]:
            for concept in concepts:
                formatted_data.append([concept, 0])
        else:
            print(f"No answer in: {line}")

    return formatted_data
        
    
class BertDataset(Dataset):
    def __init__(self, tokenizer, max_length, data):
        super(BertDataset, self).__init__()
        self.data=data
        self.tokenizer=tokenizer
        self.max_length=max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        
        text = self.data[index][0]
        
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            padding='max_length',
            add_special_tokens=True,
            return_attention_mask=True,
            max_length=self.max_length,
            truncation=True,
        )
        ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        mask = inputs["attention_mask"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'target': self.data[index][1]
            }


class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        self.out = nn.Linear(768, 1)
        
    def forward(self,ids,mask,token_type_ids):
        _,o2= self.bert_model(ids,attention_mask=mask,token_type_ids=token_type_ids, return_dict=False)
        
        out= self.out(o2)
        
        return out


def finetune(epochs, dataloader, model, loss_fn, optimizer):
    model.train()
    for epoch in range(epochs):

        train_correct = 0
        train_samples = 0
        train_batch_accuracies = []
        
        loop=tqdm(enumerate(dataloader), leave=False, total=len(dataloader))
        for batch, dl in loop:
            ids = dl['ids']
            token_type_ids = dl['token_type_ids']
            mask = dl['mask']
            label =dl['target']
            label = label.unsqueeze(1)
            
            optimizer.zero_grad()
            
            output=model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids)
            label = label.type_as(output)

            loss=loss_fn(output,label)
            loss.backward()
            
            optimizer.step()
            
            pred = np.where(output >= 0, 1, 0)

            num_correct = sum(1 for a, b in zip(pred, label) if a[0] == b[0])
            train_correct += num_correct
            num_samples = pred.shape[0]
            train_samples += num_samples
            accuracy = num_correct/num_samples
            train_batch_accuracies.append(accuracy)
            
            # Show progress while training
            loop.set_description(f'Epoch={epoch}/{epochs}')
            loop.set_postfix(loss=loss.item(),acc=accuracy)
            
        print(f'Got {train_correct} / {train_samples} with accuracy {float(train_correct)/float(train_samples)*100:.2f}')
        # plot train_batch_accuracies
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='Bert finetuning',
                    description='This program finetunes a BERT model')
    parser.add_argument('-d', '--data_path', default="/home/mlajos/DeepNeurOntology/LLM/model_outputs/cui_formatted_llama.json_meta-llama_Llama-2-70b-chat-hf")
    args = parser.parse_args()

    data = load_data_from_json(path=args.data_path)
    print(f"Numver of labeled concepts: {len(data)}")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    dataset= BertDataset(tokenizer, max_length=100, data=data)

    dataloader=DataLoader(dataset=dataset, batch_size=32, shuffle=True)

    model=BERT()

    loss_fn = nn.BCEWithLogitsLoss()

    optimizer= optim.Adam(model.parameters(),lr= 0.001)

    
    for param in model.bert_model.parameters():
        param.requires_grad = False

    model = finetune(5, dataloader, model, loss_fn, optimizer)
