DATASET_FILE, HUMAN_ANNOTATIONS = 'sampled_fails_500.csv', 'analyzed_fails_984.tsv'
RUNS = ['bert_large:squad', 'bert_large:nq_closed']

# load models and datasets
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch
import pandas as pd

from subprocess import run

def load_models():
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForSequenceClassification.from_pretrained('roberta-base')
    print('models loaded')
    return model, tokenizer

def load_dataset():
    def load_raw_data(dataset_file, annotation_file, use_runs):
        data = pd.read_csv(dataset_file)
        data = data[data['run'].isin(use_runs)]
        anno = pd.read_csv(annotation_file, sep='	')

        anno.rename(columns={ 'worker_score': 'ground_truth' }, inplace=True)
        run_names = [data[data['id'] == id]['run'].array[0] for id in anno['id']]
        anno['run'] = run_names

        data = data.set_index(['run', 'id'])
        anno = anno.set_index(['run', 'id'])

        data = data.join(anno, how='inner')

        return data[['context', 'question', 'model_answer', 'answers']].to_records(), [int(label) for label in data['ground_truth'].array]

    def split_dataset(data, train_ratio = 0.75, val_ratio = 0.15, test_ratio = 0.10, labels=None):
        from sklearn.model_selection import train_test_split
        # train/test/val algorithm from https://datascience.stackexchange.com/a/53161

        # train is now 75% of the entire data set
        x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=1-train_ratio, stratify=labels)

        # test is now 10% of the initial data set
        # validation is now 15% of the initial data set
        x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + val_ratio), stratify=y_test)

        return (x_train, y_train), (x_val, y_val), (x_test, y_test)

    data, labels = load_raw_data(DATASET_FILE, HUMAN_ANNOTATIONS, RUNS)
    return split_dataset(data, 0.8, 0.1, 0.1, labels)

class QAValidationDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, data, labels):
        data = [ f"<s>{q}</s>{a}</s>{g}</s>" for _, _, _, q, a, g in data ]
        self.encodings = tokenizer(data, truncation=True, padding=True)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# TRAIN STUFF
model, tokenizer = load_models()
train, val, test = load_dataset()
train, val, test = [QAValidationDataset(tokenizer, *ds) for ds in load_dataset()]

training_args = TrainingArguments(
    output_dir=run(['witty-phrase-generator', '-a2'], capture_output=True, text=True).stdout.strip(),
    num_train_epochs=30,            # total number of training epochs
    per_device_train_batch_size=32, # batch size per device during training
    per_device_eval_batch_size=64,  # batch size for evaluation
    warmup_steps=500,               # number of warmup steps for learning rate scheduler
    weight_decay=0.01,              # strength of weight decay
    logging_dir='./logs',           # directory for storing logs
    logging_steps=1,
    report_to="wandb"               # log to wandb
)

trainer = Trainer(
    model=model,                    # the instantiated 🤗 Transformers model to be trained
    args=training_args,             # training arguments, defined above
    train_dataset=train,            # training dataset
    eval_dataset=val                # evaluation dataset
)

trainer.train()

# TEST STUFF
model.to('cpu')
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
outputs = model(**inputs, labels=labels)
loss = outputs.loss
logits = outputs.logits

