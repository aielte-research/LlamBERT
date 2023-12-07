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
import neptune
from dotenv import load_dotenv
import os
        
    
class BertDataset(Dataset):
    def __init__(self, tokenizer, max_length, data, device):
        super(BertDataset, self).__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        
        text = self.data[index]["txt"]
        
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
            'ids': torch.tensor(ids, dtype=torch.long).to(self.device),
            'mask': torch.tensor(mask, dtype=torch.long).to(self.device),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long).to(self.device),
            'target': self.data[index]["label"]
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


def finetune(num_epochs, train_dataloader, val_dataloader, model, loss_fn, optimizer, run):
    
    for epoch in range(num_epochs):
        
        # train loop
        model.train()
        train_correct = 0
        train_samples = 0
        train_batch_accuracies = []
        
        loop=tqdm(enumerate(train_dataloader), leave=False, total=len(train_dataloader))
        for _, data in loop:
            ids = data['ids']
            token_type_ids = data['token_type_ids']
            mask = data['mask']
            label = data['target']
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
            
            pred = np.where(output.to("cpu") >= 0, 1, 0)

            num_correct = sum(1 for a, b in zip(pred, label) if a[0] == b[0])
            train_correct += num_correct
            num_samples = pred.shape[0]
            train_samples += num_samples
            train_accuracy = num_correct/num_samples
            train_batch_accuracies.append(train_accuracy)
            
            # Show progress while training
            loop.set_description(f'Epoch={epoch}/{num_epochs}')
            loop.set_postfix(loss=loss.item(),acc=train_accuracy)
            if run is not None:
                run["train/loss"].append(loss.item())
                run["train/accuracy"].append(train_accuracy)
            
        print(f'Epoch={epoch}/{num_epochs}, Train:\t got {train_correct} / {train_samples} with accuracy {float(train_correct)/float(train_samples)*100:.2f}%')

        with torch.no_grad():
            model.eval()
            val_correct = 0
            val_samples = 0
            
            loop=tqdm(enumerate(val_dataloader), leave=False, total=len(val_dataloader))
            for _, data in loop:
                ids = data['ids']
                token_type_ids = data['token_type_ids']
                mask = data['mask']
                label = data['target']
                label = label.unsqueeze(1)
                
                
                output=model(
                    ids=ids,
                    mask=mask,
                    token_type_ids=token_type_ids)
                label = label.type_as(output)

                val_loss=loss_fn(output,label)
                
                pred = np.where(output.to("cpu") >= 0, 1, 0)

                num_correct = sum(1 for a, b in zip(pred, label) if a[0] == b[0])
                val_correct += num_correct
                num_samples = pred.shape[0]
                val_samples += num_samples
                val_accuracy = num_correct/num_samples
                
                # Show progress while training
                loop.set_description(f'Epoch={epoch}/{num_epochs}')
                loop.set_postfix(loss=val_loss.item(),acc=val_accuracy)
                if run is not None:
                    run["val/loss"].append(val_loss.item())
                    run["val/accuracy"].append(val_accuracy)
            
        print(f'Epoch={epoch}/{num_epochs}, Validation:\t got {val_correct} / {val_samples} with accuracy {float(val_correct)/float(val_samples)*100:.2f}%')
    
    # plot train_batch_accuracies

    return model


def test(test_dataloader, model, run):
    with torch.no_grad():
        model.eval()
        test_correct = 0
        test_samples = 0
        
        loop=tqdm(enumerate(test_dataloader), leave=False, total=len(test_dataloader))
        for _, data in loop:
            ids = data['ids']
            token_type_ids = data['token_type_ids']
            mask = data['mask']
            label = data['target']
            label = label.unsqueeze(1)
            
            
            output=model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids)
            label = label.type_as(output)

            test_loss=loss_fn(output,label)
            
            pred = np.where(output.to("cpu") >= 0, 1, 0)

            num_correct = sum(1 for a, b in zip(pred, label) if a[0] == b[0])
            test_correct += num_correct
            num_samples = pred.shape[0]
            test_samples += num_samples
            test_accuracy = num_correct/num_samples
            
            if run is not None:
                run["test/loss"].append(test_loss.item())
                run["test/accuracy"].append(test_accuracy)
        
    print(f'Test:\t got {test_correct} / {test_samples} with accuracy {float(test_correct)/float(test_samples)*100:.2f}%')


if __name__ == '__main__':
    
    load_dotenv()
    run = neptune.init_run(
        project="aielte/DNeurOn",
        api_token=os.getenv("NEPTUNE_API_TOKEN"),
    )
    parser = argparse.ArgumentParser(
                    prog='Bert finetuning',
                    description='This program finetunes a BERT model')
    parser.add_argument('-d', '--data_path', required=True)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device={device}")

    with open(os.path.join(args.data_path, "train.json"), "r") as file:
        train_data = json.load(file)
    with open(os.path.join(args.data_path, "test.json"), "r") as file:
        test_data = json.load(file)
    print(f"Number of train examples loaded: {len(train_data)}")
    print(f"Number of test examples loaded: {len(test_data)}")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    MAX_LEN = 128
    BATCH_SIZE = 512

    train_dataset = BertDataset(tokenizer, device=device, max_length=MAX_LEN, data=train_data[:20000])
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_dataset = BertDataset(tokenizer, device=device, max_length=MAX_LEN, data=train_data[20000:])
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    test_dataset = BertDataset(tokenizer, device=device, max_length=MAX_LEN, data=test_data)
    test_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model=BERT().to(device)

    loss_fn = nn.BCEWithLogitsLoss()

    optimizer= optim.Adam(model.parameters(),lr= 0.001)

    
    for param in model.bert_model.parameters():
        param.requires_grad = False

    model = finetune(
        num_epochs=10, 
        train_dataloader=train_dataloader, 
        val_dataloader=val_dataloader, 
        model=model, 
        loss_fn=loss_fn, 
        optimizer=optimizer,
        run=run
    )

    test(
        model=model,
        run=run,
        test_dataloader=test_dataloader
    )
