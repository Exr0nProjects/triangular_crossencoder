OUTPUT_PATH = '../data/squad_augmented.tsv'

from datasets import load_dataset

import pandas as pd
from tqdm import tqdm

from re import findall
from operator import itemgetter
from json import dumps
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
    try: ret.append(('_4fwd', seek_words(context, space_before, space_before + len(answer), 4), 0))
    except ValueError: pass

    # four words backwards
    try: ret.append(('_4bkw', seek_words(context, space_before, space_before + len(answer), -4), 0))
    except ValueError: pass

    # random string from context
    start = randint(0, num_words - 8)
    ret.append(('_rand', ' '.join(context.split()[start:start+randint(2, 6)]), 0))

    # random substring from answer
    # TODO: probably high false negative rate
    num_words = len(answer.split())
    beg = randint(0, num_words-1)
    end = randint(beg+1, num_words if beg > 0 else num_words-1)
    ret.append(('_subs', ' '.join(answer.split()[beg:end]), 0))

    return ret
    # return [['squad_augmented', id + f'_aug{i}', context, question, f_ans, answer] for i, f_ans in enumerate(ret)]

def augment_squad():
    ret = pd.DataFrame(columns=['id', 'label', 'context', 'question', 'model_answer', 'answers'])
    for split in ['train', 'validation']:
        ds = load_dataset('squad', split=split)
        for row in tqdm(ds):
            id, context, question, answers = itemgetter('id', 'context', 'question', 'answers')(row)
            # print(id, context, question, answers)
            for start, text in dict.fromkeys(zip(answers['answer_start'], answers['text'])):
                # print(f"    '{context[start:start + len(text)]}'")
                for id_suf, f_ans, label in generate_examples(id, context, question, text, start):
                    ret = ret.append({ 'id': id + id_suf, 'label': int(label), 'context': context, 'question': question, 'model_answer': f_ans, 'answers': dumps(answers['text']) }, ignore_index=True)
            break
    print(ret)

    # TODO: ideas for positive examples: paraphrase, generative answering model, sentence structures templating
    # SOURCE: sts benchmark data from here: http://ixa2.si.ehu.eus/stswiki/index.php/STSbenchmark

if __name__ == '__main__':
    # seed(1336)
    augment_squad()
    # print(f"'{seek_words('one two three four five', 4, 11, 2)}'")
