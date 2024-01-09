from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForSequenceClassification
# from transformers import BertForSequenceClassification, BertTokenizerFast
from transformers.integrations import NeptuneCallback
import torch
import json
import argparse
import neptune
from dotenv import load_dotenv
import os
from cfg_parser import parse
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import random
import numpy as np
from neptune.types import File

if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
print(f"Using device={DEVICE}")

def to_cuda(var):
    if DEVICE=="cuda":
        return var.cuda()
    return var

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, zero_division=0.0, average="binary", pos_label=1)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)


def mislabel_data(data, percent):
    print(f"WARNING: Mislabeling {percent}% of data ({int(len(data)*(percent/100))} lines)!")
    indices_to_change = random.sample(range(len(data)), int(len(data)*(percent/100)))
    #print(indices_to_change)
    for idx in indices_to_change:
        if data[idx]["label"] == 0:
            data[idx]["label"] = 1  
        else:
            data[idx]["label"] = 0


def log_mispredicted(data, prediction_results, run, log_name="mislabeled/test"):
    print(f"Calculating and logging mispredicted datapoint into {log_name} neptune path...")
    prediction_results = prediction_results._asdict()
    mislabeled = []
    for i in range(len(data)):
        if np.argmax(prediction_results["predictions"][i]) != int(data[i]["label"]):
            mislabeled.append(f'{data[i]["txt"]} | True: {int(data[i]["label"])}, Predicted: {np.argmax(prediction_results["predictions"][i])} ({prediction_results["predictions"][i]})' )
    run[log_name].upload(File.from_content("\n".join(mislabeled)))


def main(cfg):
    if cfg["neptune_logging"]:
        run = neptune.init_run(
            project="aielte/DNeurOn",
            api_token=os.getenv("NEPTUNE_API_TOKEN"),
        )
        run_id = run["sys/id"].fetch()
        print(run_id)
        run["cfg"] = neptune.utils.stringify_unsupported(cfg)

    default_params={
        "num_epochs": 5,
        "batch_size": 16,
        "max_len": 512,
        "model_save_location": None,
        "freeze_encoder": False,
        "log_test_outputs": False,
        "train_ratio": 0.9
    }
    for key in default_params:
        cfg[key] = cfg.get(key, default_params[key])

    if run is not None:
        run["device"] = DEVICE

    with open(os.path.join(cfg['data_path'], "train.json"), "r") as file:
        data = json.load(file)
        train_data = data[:int(cfg["train_ratio"]*len(data))]
        val_data = data[int(cfg["train_ratio"]*len(data)):]
    with open(os.path.join(cfg['data_path'], "test.json"), "r") as file:
        test_data = json.load(file)
    
    if cfg["reduce_lines_for_testing"]:
        print("WARNING: Keeping only 100 sentences for test and train for tesing!")
        train_data = train_data[:100]
        val_data = val_data[:100]
        test_data = test_data[:100]

    if int(cfg["mislabel_percent"]) != 0:
        mislabel_data(train_data, cfg["mislabel_percent"])

    print(f"Number of train samples loaded: {len(train_data)}")
    print(f"Number of validation samples loaded: {len(val_data)}")
    print(f"Number of test samples loaded: {len(test_data)}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    #tokenizer = BertTokenizerFast.from_pretrained(cfg["model_name"])

    train_encodings = tokenizer([d["txt"] for d in train_data], truncation=True, padding=True, max_length=cfg["max_len"])
    train_dataset = SentimentDataset(encodings=train_encodings, labels=[d["label"] for d in train_data])

    if cfg["train_ratio"] < 1:
        evaluation_strategy = "steps"
        val_encodings = tokenizer([d["txt"] for d in val_data], truncation=True, padding=True, max_length=cfg["max_len"])
        val_dataset = SentimentDataset(encodings=val_encodings, labels=[d["label"] for d in val_data])
    else:
        evaluation_strategy = "no"
        val_dataset = None

    test_encodings = tokenizer([d["txt"] for d in test_data], truncation=True, padding=True, max_length=cfg["max_len"])
    test_dataset = SentimentDataset(encodings=test_encodings, labels=[d["label"] for d in test_data])

    model = to_cuda(AutoModelForSequenceClassification.from_pretrained(cfg["model_name"], num_labels=2))

    if cfg["freeze_encoder"]:
        for name, param in model.named_parameters():
            if "embedding" in name or "encoder" in name:
                param.requires_grad = False
    
    training_args = TrainingArguments(
        output_dir='./results',         
        num_train_epochs=cfg["num_epochs"],             
        per_device_train_batch_size=cfg["batch_size"], 
        per_device_eval_batch_size=cfg["batch_size"],   # 1024 70 GB uses VRAM 
        warmup_steps=int(1000/cfg["batch_size"]),                
        weight_decay=0.01,            
        logging_dir='./logs',           
        load_best_model_at_end=False,  # Too muxh space  
        learning_rate=cfg["lr"],
        logging_steps=int(min(5000, len(train_dataset))/cfg["batch_size"]),   #  This is complicated for the testing with 100 samples    
        save_strategy='no',
        evaluation_strategy=evaluation_strategy,    
        report_to="none",
    )

    neptune_callback = NeptuneCallback(run=run)

    trainer = Trainer(
        model=model,                         # the instantiated Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset,          # evaluation dataset
        compute_metrics=compute_metrics,     # the callback that computes metrics of interest
        callbacks=[neptune_callback],
    )

    trainer.train()

    test_results = trainer.evaluate(
        eval_dataset=test_dataset,
    )

    #the trainer stops the neptune run, so we reopen it
    run = neptune.Run(with_id=run_id)
    run["test"] = test_results

    test_predict_results = trainer.predict(
        test_dataset=test_dataset,
    )
    log_mispredicted(data=test_data, prediction_results=test_predict_results, log_name="mislabeled/test", run=run)

    if cfg["train_ratio"] < 1:
        val_predict_results = trainer.predict(
            test_dataset=val_dataset,
        )
        log_mispredicted(data=val_data, prediction_results=val_predict_results, log_name="mislabeled/validation", run=run)

    if cfg["log_test_outputs"]:
        print(f"Logging all test outputs...")
        prediction_results = test_predict_results._asdict()
        run["test/test_outputs"].upload(File.from_content("\n".join([str(np.argmax(x)) for x in prediction_results["predictions"]])))

    if "model_save_location" in cfg:
        dirname = cfg["model_save_location"]
        if dirname is not None:
            print(f"Saving model to {dirname}")
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            trainer.save_model(dirname)

    run.stop()

if __name__ == '__main__':    
    load_dotenv()
    parser = argparse.ArgumentParser(
                    prog='Bert finetuning',
                    description='This program finetunes a BERT model')
    parser.add_argument('-c', '--config_path', required=True)
    args = parser.parse_args()

    cfg, _ = parse(args.config_path)
    for c in cfg:
        main(c)