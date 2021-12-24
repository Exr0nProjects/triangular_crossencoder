OUTPUT_PATH = '../data/squad_augmented.tsv'

from datasets import load_dataset

import pandas as pd

from re import findall
from operator import itemgetter
from random import randint, random, seed, choice

def seek_words(context, start, end, num):   # inc l exc r
    # print('seeking', start, end, num)
    direction = num > 0
    num = abs(num)
    pos = end + 1 if direction else start - 2
    while num != 0:
        pos += (1 if direction else -1)
        if (pos == -1 or pos == len(context)) and num == 1:
            pos = max(min(pos, len(context)), 0)
            break
        if pos < 0 or pos >= len(context):
            raise ValueError(f"Position {pos} is out of range of {len(context)}!")
        if context[pos] == ' ':
            num -= 1
    return context[pos+1:end] if pos < start else context[start:pos]


def generate_examples(id, context, question, answer, space_before):
    ret = []

    num_words = len(context.split())
    # print(context, question, answer, space_before)

    # four words forwards
    try:
        ret.append(seek_words(context, space_before, space_before + len(answer), 4))
    except ValueError:
        pass

    # four words backwards
    try:
        ret.append(seek_words(context, space_before, space_before + len(answer), 4))
    except ValueError:
        pass

    # random string from context

    for i in range(1000):
        space_before = choice([i for i, c in enumerate(context) if c == ' '][2:-2])
        length = 1
        while context[space_before + length] != ' ':
            length += 1
        # print('selected center', space_before, space_before + length, f"'{context[space_before+1:space_before+length]}'")
        seek = randint(3, 6) * (-1 if space_before > len(context)/2 else 1)
        # print(center, seek, context[center:])
        ret.append(seek_words(context, space_before+1, space_before+length, seek))
    print('\n'.join(ret))

def augment_squad():
    for split in ['train', 'validation']:
        ds = load_dataset('squad', split=split)
        for row in ds:
            id, context, question, answers = itemgetter('id', 'context', 'question', 'answers')(row)
            # print(id, context, question, answers)
            for start, text in zip(answers['answer_start'], answers['text']):
                # print(f"    '{context[start:start + len(text)]}'")
                generate_examples(id, context, question, text, start)
        break


if __name__ == '__main__':
    seed(1336)
    augment_squad()
    # print(f"'{seek_words('one two three four five', 4, 11, 2)}'")
