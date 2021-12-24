OUTPUT_PATH = '../data/squad_augmented.tsv'

from datasets import load_dataset

import pandas as pd

from operator import itemgetter

def generate_examples(id, context, question, answers):
    ret = []
    print(context, question, answers)

def augment_squad():
    for split in ['train', 'validation']:
        ds = load_dataset('squad', split=split);
        for row in ds:
            id, context, question, answers = itemgetter('id', 'context', 'question', 'answers')(row)
            print(id, context, question, answers)
            for start, text in zip(answers['answer_start'], answers['text']):
                print(f"    '{context[start:start + len(text)]}'")

if __name__ == '__main__':
    augment_squad()
