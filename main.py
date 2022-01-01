DATASET_FILE, HUMAN_ANNOTATIONS = 'sampled_fails_500.csv', 'analyzed_fails_984.tsv'
RUNS = ['bert_large:squad', 'bert_large:nq_closed']

# load models and datasets
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers import InputExample
from datasets import load_dataset as hf_load_dataset

from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
import pandas as pd
from tqdm import tqdm, trange
import wandb

from normal_crossencoder.main import get_data

from operator import itemgetter as ig
from itertools import islice
# from subprocess import run

def qaval_adaptor(dsnames, dataset):
    assert isinstance(dataset[0], InputExample)
    data = [ (','.join(dsnames), ex.guid, None, None, ex.texts[0], ex.texts[1]) for ex in dataset ]
    labels = [ int(ex.label) for ex in dataset ]
    return data, labels

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

def load_dataset(tokenizer, dsname='mine'):
    if isinstance(dsname, list):
        return [QAValidationDataset(tokenizer, *qaval_adaptor(dsname, sp)) for sp in get_data(dsname)]

    assert dsname in ['mine', 'quora', 'stsb']

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

    if dsname == 'quora':
        ds = hf_load_dataset('quora', split='train')
        questions, labels = ig('questions', 'is_duplicate')(ds)
        questions = [ ['quora', x['id'][0]//2, None, None, *x['text']] for x in questions ]
        labels = [ int(x) for x in labels ]
        # print(questions)
        # import time; time.sleep(1)
        return [QAValidationDataset(tokenizer, *dat) for dat in split_dataset(questions, 0.8, 0.1, 0.1, labels)]

    if dsname == 'augmented':
        data = pd.read_csv('data/squad_augmented.tsv');

    if dsname == 'stsb':
        dses = [ hf_load_dataset('stsb_multi_mt', name='en', split=split) for split in ['train', 'dev', 'test'] ]
        return [QAValidationDataset(tokenizer, [ [None, None, None, s1, s1, s2] for s1, s2 in zip(things['sentence1'], things['sentence2']) ], [ s/5 for s in things['similarity_score'] ]) for things in dses]

    data, labels = load_raw_data(DATASET_FILE, HUMAN_ANNOTATIONS, RUNS)
    # return split_dataset(data, 0.8, 0.1, 0.1, labels)
    return [QAValidationDataset(tokenizer, *dat) for dat in split_dataset(data, 0.8, 0.1, 0.1, labels)]

def train(dataloader, model, optimizer, validate):
    for num, batch in tqdm(enumerate(dataloader), total=len(dataloader), leave=False):
        X = batch['input_ids'].to(device)
        # y = batch['labels'].to(device)

        # print('\n\ny shape', y, '\nx shape', X.shape, '\n\n')
        # return

        y = batch['labels'].to(device)
        # # print('\n\ny dtype', y.dtype, type(y.dtype))
        # if y.dtype == torch.float32:
        #     y = torch.stack((y, 1-y)).transpose(0, 1)
        # # print(type(y), y.shape, y.dtype, y[:3], type(X), X.shape)
        # # y = y.to(device)

        optimizer.zero_grad()

        pred = model(X, labels=y)

        pred.loss.backward()
        optimizer.step()

        if (wandb.run.step % 3000 == 0):
            validate(model)
            model.save_pretrained(f"saved_models/{wandb.run.name}/{wandb.run.step // 1000}k")
        wandb.log({'loss': pred.loss.item()}, commit=True)

def test(dataloader, model, name='validation'):
    test_loss, correct = 0, 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f'val {name}...', leave=False):
            X, y = ig('input_ids', 'labels')(batch)
            X = X.to(device)
            y = batch['labels'].to(device)

            accuracy = (y > 0.5).type(torch.int).to(device)
            # if y.dtype == torch.float32:
            #     y = torch.stack((y, 1-y)).transpose(0, 1)

            # y = y.to(device)
            pred = model(X, labels=y)
            test_loss += pred.loss.item()

            correct += (pred.logits.argmax(dim=1) == accuracy).type(torch.float).sum().item()
            # correct += (pred.logits.argmax(dim=1) == y).type(torch.float).sum().item()

    # print('total correct:', correct)

    test_loss /= len(dataloader)
    correct /= len(dataloader.dataset)

    wandb.log({ f'{name}/test_loss': test_loss, f'{name}/accuracy': correct })


def test_multiple(dataloaders):
    def run(model):
        for name, test_set in dataloaders.items():
            test(test_set, model, name)
    return run

if __name__ == '__main__':
# TRAIN STUFF
# dataloaders vs datasets https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
# finetuning a huggingface model using native pytorch https://huggingface.co/docs/transformers/training

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('loading models...')
    model, tokenizer = load_models()
    print('loading datasets...')

    sas_dataloader = DataLoader(QAValidationDataset(tokenizer, *qaval_adaptor('sas', get_data(['sas'])[0])), batch_size=64)

    train_config = {
        'epochs': int(1e5),
        'lr': 1e-3,
        'bs': 32,
    }
    train_phases = [
        { 'epochs': 1,  'dataset': 'quora' },
        { 'epochs': 2,  'dataset': ['quora', 'wes'] },
        { 'epochs': 100, 'dataset': ['wes'] }
    ]

# print(f"\ntrain: {len(train_dataloader.dataset)}, val: {len(val_dataloader.dataset)}, test: {len(test_dataloader.dataset)}")
# print(f"train batches: {len(train_dataloader)}, val batches: {len(val_dataloader)}, test batches: {len(test_dataloader)}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config['lr'])

    wandb.init()
    wandb.watch(model)
    print('model', model)
    print('config:', train_config, train_phases)
    print('optimizer', optimizer)

# loss_fn = nn.NLLLoss()
    model.to(device)



    print('beginning train loop')
    for phase in train_phases:
        conf = { **train_config, **phase }

        train_dataloader, val_dataloader, test_dataloader = [DataLoader(ds, conf['bs'], shuffle=True) for ds in load_dataset(tokenizer, conf['dataset'])]

        validate = test_multiple({
            'SAS_eval': sas_dataloader,
            ','.join(conf['dataset']) if isinstance(conf['dataset'], list) else conf['dataset']: val_dataloader,
        })

        for t in trange(conf['epochs'], desc=str(conf['dataset'])):
            train(train_dataloader, model, optimizer, validate)
            # test(val_dataloader, model)
            # train(train_dataloader_quora, model, optimizer)
        wandb.log({ 'loss': -0.1 }, commit=True)

# TEST STUFF
# model.to('cpu')
# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
# outputs = model(**inputs, labels=labels)
# loss = outputs.loss
# logits = outputs.logits
