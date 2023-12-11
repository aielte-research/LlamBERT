from transformers import BertForSequenceClassification, TrainingArguments, Trainer, BertTokenizerFast
from transformers.integrations import NeptuneCallback
import torch
import json
import argparse
import neptune
from dotenv import load_dotenv
import os
from cfg_parser import parse
from sklearn.metrics import accuracy_score


def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  # calculate accuracy using sklearn's function
  acc = accuracy_score(labels, preds)
  return {
      'accuracy': acc,
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


def main(cfg):
    if cfg["neptune_logging"]:
        run = neptune.init_run(
            project="aielte/DNeurOn",
            api_token=os.getenv("NEPTUNE_API_TOKEN"),
        )
        run_id = run["sys/id"].fetch()
        print(run_id)
        run["cfg"] = cfg

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    if run is not None:
        run["device"] = device
    print(f"Using device={device}")

    with open(os.path.join(cfg['data_path'], "train.json"), "r") as file:
        train_data = json.load(file)
    with open(os.path.join(cfg['data_path'], "test.json"), "r") as file:
        test_data = json.load(file)
    print(f"Number of train examples loaded: {len(train_data)}")
    print(f"Number of test examples loaded: {len(test_data)}")

    tokenizer = BertTokenizerFast.from_pretrained(cfg["model_name"])

    MAX_LEN = cfg["max_len"]
    BATCH_SIZE = cfg["batch_size"]
    NUM_EPOCHS = cfg["num_epochs"]
    TRAIN_LEN = cfg["train_lines"]

    train_encodings = tokenizer([d["txt"] for d in train_data[:TRAIN_LEN]], truncation=True, padding=True, max_length=MAX_LEN)
    train_dataset = SentimentDataset(encodings=train_encodings, labels=[d["label"] for d in train_data[:TRAIN_LEN]])

    val_encodings = tokenizer([d["txt"] for d in train_data[TRAIN_LEN:]], truncation=True, padding=True, max_length=MAX_LEN)
    val_dataset = SentimentDataset(encodings=val_encodings, labels=[d["label"] for d in train_data[TRAIN_LEN:]])

    test_encodings = tokenizer([d["txt"] for d in test_data], truncation=True, padding=True, max_length=MAX_LEN)
    test_dataset = SentimentDataset(encodings=test_encodings, labels=[d["label"] for d in test_data])

    model = BertForSequenceClassification.from_pretrained(cfg["model_name"], num_labels=2).to(device)

    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=NUM_EPOCHS,              # total number of training epochs
        per_device_train_batch_size=BATCH_SIZE,  # batch size per device during training
        per_device_eval_batch_size=1024,   # batch size for evaluation
        warmup_steps=int(1000/BATCH_SIZE),                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        load_best_model_at_end=True,     # load the best model when finished training (default metric is loss)
        # but you can specify `metric_for_best_model` argument to change to accuracy or other metric
        logging_steps=int(5000/BATCH_SIZE),               # log & save weights each logging_steps
        save_steps=int(100000/BATCH_SIZE),
        evaluation_strategy="steps",     # evaluate each `logging_steps`
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

    
