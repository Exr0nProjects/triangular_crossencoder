from datasets import load_dataset
import pandas as pd
import numpy as np
from tqdm import tqdm

from collections import defaultdict
from operator import itemgetter as ig
from itertools import islice, chain, repeat
from random import sample, choice, shuffle
from gc import collect

# SUBSET = '2018thresh20'

def generate_splits(subset, split=[0.75, 0.15, 0.1]):
    assert abs(sum(split) - 1.0) < 0.0001
    # get the data in dictionary form
    groups = defaultdict(list)
    ds = load_dataset('Exr0n/wiki-entity-similarity', subset, split='train')
    ds = list(tqdm(ds, total=len(ds)))
    for article, link in tqdm(map(ig('article', 'link_text'), ds), total=len(ds)):
        groups[article].append(link)
    del ds

    # greedily allocate splits
    order = sorted(groups.keys(), reverse=True, key=lambda e: groups[e])
    splits = [[] for _ in split]
    sizes = [0.001] * len(split)    # avoid div zero error
    for group in order:
        impoverished = np.argmax([ s - (x/sum(sizes)) for x, s in zip(sizes, split) ])
        splits[impoverished].append(group)
        sizes[impoverished] += len(groups[group])

    sizes = [ int(x) for x in sizes ]
    print('final sizes', sizes, [x/sum(sizes) for x in sizes])

    # generate positive examples
    ret = [ [[(k, t) for t in groups[k]] for k in keys] for keys in splits ]

    # generate negative examples randomly (TODO: probably a more elegant swapping soln)
    for i, keys in enumerate(splits):
        for key in keys:
            try:
                got = sample(keys, len(groups[key])+1)
                ret[i].append(
                    [(key, choice(groups[k])) for k in got if k != key]
                    [:len(groups[key])]
                )
            except ValueError:
                raise ValueError("well frick one group is bigger than all the others combined. try sampling one at a time")

    collect()
    return [(chain(*s), chain(repeat(1, z), repeat(0, z))) for z, s in zip(sizes, ret)]


if __name__ == '__main__':
    for size in [5, 10, 20]:
        x = generate_splits(subset='2018thresh' + str(size))

        for (data, labels), split in zip(x, ['train', 'dev', 'test']):
            articles, lts = list(zip(*data))
            df = pd.DataFrame({ 'article': articles, 'link_text': lts, 'is_same': list(labels) })
            df = df.sample(frac=1).reset_index(drop=True)
            df.to_csv('2018thresh' + str(size) + split + '.csv', index=False)
            # print(df.head(30), df.tail(30))

    # tests
    # for data, labels in x[2:]:
    #     data = list(data)
    #     labels = list(labels)
    #
    #     assert sum(labels) * 2 == len(labels)
    #     num = sum(labels)
    #
    #     before = [ a for a, _ in data[:num] ]
    #     after  = [ a for a, _ in data[num:] ]
    #     assert before == after
    #
    #     print(data[num:])
