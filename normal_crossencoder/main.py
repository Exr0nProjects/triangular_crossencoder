# using sentence_transformers instead of just torch and datasets (WOW!!)
from datasets import load_dataset
from sentence_transformers import CrossEncoder, InputExample
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator, CECorrelationEvaluator

import torch
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import pandas as pd

from operator import itemgetter as ig

RUN_CONFIG = {
    'batch_size': 16,
    'epochs': int(1e5),
    'warmup': int(1e2),
    'eval_steps': int(1e3),
    'base_model': 'cross-encoder/stsb-roberta-large',
    'training': [
        {
            'train_data': ['stsb'],
            'epochs': 5,
        },
        {
            'train_data': ['quora'],
            'epochs': 100
        }
    ]
}

def split_dataset(data, train_ratio = 0.75, val_ratio = 0.15, test_ratio = 0.10, labels=None):
    from sklearn.model_selection import train_test_split
    # train/test/val algorithm from https://datascience.stackexchange.com/a/53161

    # train is now 75% of the entire data set
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=1-train_ratio, stratify=labels)

    # test is now 10% of the initial data set
    # validation is now 15% of the initial data set
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + val_ratio), stratify=y_test)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def get_model(model, num_labels=1):
    if model == 'roberta_stsb':
        return CrossEncoder('cross-encoder/stsb-roberta-large', num_labels=num_labels)
    if model == 'roberta_quora':
        return CrossEncoder('cross-encoder/quora-roberta-large', num_labels=num_labels)
    else:
        return CrossEncoder(model, num_labels=num_labels)

# def flat(iter):
#     for x in iter:
#         try:
#             for y in x:
#                 yield y
#         except TypeError:
#             yield x

def flat(iter):
    return [ x for y in iter for x in y ]

def get_data(dsnames):
    def gen_symmetric_ex(i, a, b, s):
        return (InputExample(i + 'f', [a, b], s), InputExample(i + 'r', [b, a], s))

    def get_data_single(dsname):
        if dsname == 'quora':
            # raise NotImplementedError('need to return splits')
            ds = load_dataset('quora', split='train')
            data = [ (f"quora-{x['id'][0]//2}", *x['text']) for x in ds['questions'] ]
            labels = [ float(x) for x in ds['is_duplicate'] ]

            dses = split_dataset(data, 0.75, 0.15, 0.1, labels)

            return [ flat(gen_symmetric_ex(i, a, b, s) for (i, a, b), s in zip(dat, lab)) for (dat, lab) in dses ]
        elif dsname == 'stsb':
            dses = (load_dataset('stsb_multi_mt', name='en', split=split) for split in ['train', 'dev', 'test'])
            ret = [ flat(gen_symmetric_ex(f"stsb-{i}", x['sentence1'], x['sentence2'], x['similarity_score']/5) for i, x in enumerate(ds)) for ds in dses ]
            return ret
        elif dsname == 'sas':
            sas_squad = pd.read_csv('../data/semantic_answer_similarity_data/data/SQuAD_SAS.csv')
            sas_nqopen = pd.read_csv('../data/semantic_answer_similarity_data/data/NQ-open_SAS.csv')
            # sas_squad['run'] = ['sas_squad'] * len(sas_squad)
            # sas_squad['id'] = ['sas_squad' + int(x) for x in sas_squad.index]
            # sas_nqopen['run'] = ['sas_nqopen'] * len(sas_nqopen)
            # sas_squad['id'] = ['sas_nqopen' + int(x) for x in sas_nqopen.index]
            sas_data = pd.concat([sas_squad, sas_nqopen])
            return [ flat(gen_symmetric_ex(f"sas-{i}", a, b, l >= 1) for i, (a, b, l) in sas_data.iterrows()), [], [] ]
        else:
            raise ValueError(f"unknown dataset '{dsname}'")

    return [ flat(sp) for sp in zip(*[get_data_single(ds) for ds in dsnames]) ]

def eval_callback(score, epoch, steps):
    wandb.log({ 'eval_score': score })

class LoggingLoss:
    def __init__(self, loss_fn, wandb):
        self.loss_fn = loss_fn
        self.wandb = wandb

    def __call__(self, logits, labels):
        loss = self.loss_fn(logits, labels)
        wandb.log({ 'train_loss': loss })
        return loss

if __name__ == '__main__':
    wandb.init(config=RUN_CONFIG)

    model = get_model(wandb.config.base_model)
    wandb.watch(model.model)

    eval_data, _, _ = get_data(['sas'])

    for i, train_phase in enumerate(wandb.config.training):
        print(train_phase)
        conf = { **RUN_CONFIG, **train_phase }
        # wandb.config.update(train_phase)
        # print(wandb.config)
        train, dev, test = get_data(conf['train_data'])
        # evaluator = CECorrelationEvaluator.from_input_examples(dev, name=f"{wandb.run.id}-phase-{i}")
        evaluator = CEBinaryClassificationEvaluator.from_input_examples(eval_data, name='final_metric')

        model.fit(train_dataloader=DataLoader(train, shuffle=True, batch_size=conf['batch_size']),
                  evaluator=evaluator,
                  epochs=conf['epochs'],
                  loss_fct=LoggingLoss(BCEWithLogitsLoss(), wandb),
                  evaluation_steps=conf['eval_steps'],
                  callback=eval_callback,
                  warmup_steps=conf['warmup'],
                  output_path="saved_models/"
        )
