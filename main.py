DATASET_FILE, HUMAN_ANNOTATIONS = 'sampled_fails_500.csv', 'analyzed_fails_984.tsv'
RUNS = ['bert_large:squad', 'bert_large:nq_closed']

# load models and datasets
from sentence_transformers.cross_encoder import CrossEncoder
from datasets import load_dataset as hf_load_dataset

from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm, trange
import wandb

from operator import itemgetter
from subprocess import run

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

def load_models():
    model, tokenizer = RobertaForSequenceClassification.from_pretrained('roberta-base'), RobertaTokenizer.from_pretrained('roberta-base')
    # model, tokenizer = CrossEncoder('roberta')
    print('models loaded')
    return model, tokenizer

def load_dataset(tokenizer):
    dses = [ hf_load_dataset('stsb_multi_mt', name='en', split=split) for split in ['train', 'dev', 'test'] ]
    return [QAValidationDataset(tokenizer, [ [None, None, None, s1, s1, s2] for s1, s2 in zip(things['sentence1'], things['sentence2']) ], [ s/5 for s in things['similarity_score'] ]) for things in dses]

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
    # return split_dataset(data, 0.8, 0.1, 0.1, labels)
    return [QAValidationDataset(tokenizer, *dat) for dat in split_dataset(data, 0.8, 0.1, 0.1, labels)]


# TRAIN STUFF
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = int(1e5)
# dataloaders vs datasets https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
# finetuning a huggingface model using native pytorch https://huggingface.co/docs/transformers/training

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, tokenizer = load_models()
train_dataloader, val_dataloader, test_dataloader = [DataLoader(ds, BATCH_SIZE, shuffle=True) for ds in load_dataset(tokenizer)]
wandb.init(project='qaval_roberta_noaug')
wandb.watch(model)

# loss_fn = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

model.to(device)

def train(dataloader, model, optimizer):
    for num, batch in tqdm(enumerate(dataloader), total=len(dataloader), leave=False):
        X = batch['input_ids'].to(device)
        y = batch['labels']
        y = torch.stack((y, 1-y)).transpose(0, 1).to(device)

        pred = model(X, labels=y)

        optimizer.zero_grad()
        pred.loss.backward()
        optimizer.step()

        wandb.log({'loss': pred.loss.item()})

def test(dataloader, model):
    test_loss, correct = 0, 0

    with torch.no_grad():
        for batch in dataloader:
            X, y = itemgetter('input_ids', 'labels')(batch)
            X = X.to(device)
            y = batch['labels']
            accuracy = (y > 0.5).type(torch.int).to(device)
            y = torch.stack((y, 1-y)).transpose(0, 1).to(device)
            pred = model(X, labels=y)
            test_loss += pred.loss.item()

            # print('heres the preds', pred.logits.argmax(dim=1), 'heres labels', accuracy, 'heres the shapes', [pred.logits.argmax(dim=1).shape, accuracy.shape, 100])
            # test_loss += loss_fn(pred.logits, y).item()
            # print('adding', (pred.logits.argmax(dim=1) == y).type(torch.float).sum().item())
            correct += (pred.logits.argmax(dim=1) == accuracy).type(torch.float).sum().item()
            # print('    ', correct)

    # print('total correct:', correct)

    test_loss /= len(dataloader)
    correct /= len(dataloader.dataset)

    wandb.log({ 'test_loss': test_loss, 'accuracy': correct })

for t in trange(EPOCHS):
    train(train_dataloader, model, optimizer)
    test(val_dataloader, model)

# TEST STUFF
model.to('cpu')
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
outputs = model(**inputs, labels=labels)
loss = outputs.loss
logits = outputs.logits

