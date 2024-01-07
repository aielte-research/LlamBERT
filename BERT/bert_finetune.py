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

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    if run is not None:
        run["device"] = device
    print(f"Using device={device}")

    MAX_LEN = cfg["max_len"]
    BATCH_SIZE = cfg["batch_size"]
    NUM_EPOCHS = cfg["num_epochs"]
    TRAIN_RATIO = cfg["train_ratio"]
    LEARNING_RATE = cfg["lr"]
    if "log_test_outputs" in cfg:
        LOG_TEST_OUTPUTS = cfg["log_test_outputs"]
    else:
        LOG_TEST_OUTPUTS = False

    with open(os.path.join(cfg['data_path'], "train.json"), "r") as file:
        data = json.load(file)
        train_data = data[:int(TRAIN_RATIO*len(data))]
        val_data = data[int(TRAIN_RATIO*len(data)):]
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

    train_encodings = tokenizer([d["txt"] for d in train_data], truncation=True, padding=True, max_length=MAX_LEN)
    train_dataset = SentimentDataset(encodings=train_encodings, labels=[d["label"] for d in train_data])

    if TRAIN_RATIO < 1:
        evaluation_strategy = "steps"
        val_encodings = tokenizer([d["txt"] for d in val_data], truncation=True, padding=True, max_length=MAX_LEN)
        val_dataset = SentimentDataset(encodings=val_encodings, labels=[d["label"] for d in val_data])
    else:
        evaluation_strategy = "no"
        val_dataset = None

    test_encodings = tokenizer([d["txt"] for d in test_data], truncation=True, padding=True, max_length=MAX_LEN)
    test_dataset = SentimentDataset(encodings=test_encodings, labels=[d["label"] for d in test_data])

    model = AutoModelForSequenceClassification.from_pretrained(cfg["model_name"], num_labels=2).to(device)
    #model = BertForSequenceClassification.from_pretrained(cfg["model_name"], num_labels=2).to(device)
    
    training_args = TrainingArguments(
        output_dir='./results',         
        num_train_epochs=NUM_EPOCHS,             
        per_device_train_batch_size=BATCH_SIZE, 
        per_device_eval_batch_size=BATCH_SIZE,   # 1024 70 GB uses VRAM 
        warmup_steps=int(1000/BATCH_SIZE),                
        weight_decay=0.01,            
        logging_dir='./logs',           
        load_best_model_at_end=False,  # Too muxh space  
        learning_rate=LEARNING_RATE,
        logging_steps=int(min(5000, len(train_dataset))/BATCH_SIZE),   #  This is complicated for the testing with 100 samples    
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

    if TRAIN_RATIO < 1:
        val_predict_results = trainer.predict(
            test_dataset=val_dataset,
        )
        log_mispredicted(data=val_data, prediction_results=val_predict_results, log_name="mislabeled/validation", run=run)

    if LOG_TEST_OUTPUTS:
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