RUNS = ['bert_large:nq_closed', 'bert_large:squad']
# SAMPLE_FILE = 'out/sampled_fails_500.csv'
# SAMPLES_FILE, MANUAL_ANALYSIS_FILE, EM_CORRECT_FILES = 'out/sampled_fails_100.csv', './analyzed_fails_100.tsv', [ '../data/model_correct_outputs/bert_large:squad.csv.out', '../data/model_correct_outputs/bert_large:nq_closed.csv.out' ]
SAMPLES_FILE, MANUAL_ANALYSIS_FILE, EM_CORRECT_FILES = 'out/sampled_fails_500.csv', './analyzed_fails_984.tsv', [ '../data/model_correct_outputs/bert_large:squad.csv.out', '../data/model_correct_outputs/bert_large:nq_closed.csv.out' ]

contrast_colors = [ (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255), (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195), (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128) ] # https://sashamaps.net/docs/resources/20-colors/
contrast_colors = [(r/256, g/256, b/256) for r, g, b in contrast_colors]

import pandas as pd
from tqdm import tqdm
from sentence_transformers import CrossEncoder
from nltk.translate import meteor_score
from matplotlib import pyplot as plt

from sklearn.metrics import roc_curve, auc as calc_auc

from validation import f1 as f1_single

from itertools import islice
from ast import literal_eval
from operator import itemgetter

print('loading models....')
crossencoder_model = CrossEncoder('cross-encoder/stsb-roberta-large')


def exact_match(_q, golds, out, ctx):
    return out in golds

def f1(_q, golds, out, ctx):
    # for g in golds: print(f"{f1_single(g, out):.4f} {out} {g}")
    return max([f1_single(g, out) for g in golds])

def SAS_crossencoder(_q, golds, out, ctx):
    return max(crossencoder_model.predict([(g, out) for g in golds]))

def meteor(_q, golds, out, ctx):
    return meteor_score.meteor_score(golds, out)

METRICS = {
    'EM': exact_match,
    'F1': f1,
    'SAS': SAS_crossencoder,
    'METEOR': meteor
}

def evaluate_fails(outputs, total=None):
    print('evaluating...')
    df = pd.DataFrame(columns=['id']+list(METRICS.keys()))
    # for (run, qid), (title, ctx, q, out, gold) in tqdm(outputs, total=total):
    for (run, qid), dat in tqdm(outputs, total=total):
        try:
            title, ctx, q, out, gold = itemgetter('title', 'context', 'question', 'model_answer', 'answers')(dat)
            cur = { 'run': run, 'id': qid }
            for met_name, fn in METRICS.items():
                cur[met_name] = fn(q, literal_eval(gold), str(out), ctx)
            df = df.append(cur, ignore_index=True)
        except TypeError:
            print('TypeError:', run, qid, title, q, out, gold)
        except AttributeError:
            print(gold, literal_eval(gold), out, type(out))

    df = df.set_index(['run', 'id'])
    return df

def plot_metrics_histogram(scores):
    plt.hist(scores['F1'], bins=100, label='F1', fc=(1, 0, 0, 0.5))
    plt.hist(scores['SAS'], bins=100, label='SAS', fc=(0, 0, 1, 0.5))
    plt.hist(scores['METEOR'], bins=100, label='METEOR', fc=(0, 1, 0, 0.5))
    plt.xlabel('correctness')
    plt.ylabel('frequency')
    plt.legend()
    plt.savefig('metrics_histogram.png', dpi=300)

def plot_roc(df, filename='out/metrics_ROC.png'):
    print("plotting ROC")
    fig, ax = plt.subplots()
    tpr, fpr, thresh, auc = dict(), dict(), dict(), dict()
    for i, metric in enumerate(df.columns):
        if metric not in ['ground_truth']:
            # print(df['ground_truth'], df[metric], metric)
            fpr[metric], tpr[metric], thresh[metric] = roc_curve(df['ground_truth'], df[metric])
            auc[metric] = calc_auc(fpr[metric], tpr[metric])
            plt.plot(fpr[metric], tpr[metric], color=contrast_colors[i], label=f"{metric} ROC (area = {auc[metric]:.3f})")
    ax.plot([0, 1], [0, 1], color='black', linestyle='--')
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic example')
    ax.legend(loc="lower right")
    fig.savefig(filename)


if __name__ == '__main__':
    print('loading data...')
    fails_df = pd.read_csv(SAMPLES_FILE)
    # fails_df = pd.read_csv("out/sampled_500_filtered.csv")
    fails_df = fails_df[fails_df['run'].isin(RUNS)]
    fails_df['model_answer'].astype(str)

    analyzed_df = pd.read_csv(MANUAL_ANALYSIS_FILE, sep='	')
    run_names = [fails_df[fails_df['id'] == id]['run'].array[0] for id in analyzed_df['id']]
    # analyzed_df['ground_truth'] = analyzed_df['worker_score']
    analyzed_df.rename(columns={ 'worker_score': 'ground_truth' }, inplace=True)
    analyzed_df['run'] = run_names

    fails_df = fails_df.set_index(['run', 'id'])
    analyzed_df = analyzed_df.set_index(['run', 'id'])

    correct_dfs = [ pd.read_csv(fn) for fn in EM_CORRECT_FILES ]
    correct_df = pd.concat(correct_dfs)
    correct_df = correct_df.set_index(['run', 'id'])
    correct_df['ground_truth'] = [True] * len(correct_df)

    # outputs_df = fails_df
    # outputs_df = outputs_df.append(correct_df)
    # print(outputs_df)

    scores = evaluate_fails(fails_df.iterrows(), total=len(fails_df))
    fails_evaled_df = pd.merge(scores, analyzed_df['ground_truth'], left_index=True, right_index=True)

    scores = evaluate_fails(correct_df.iterrows(), total=len(correct_df))
    # TODO: check if these merges are working properly
    # TODO: check inter-annotator agreement using kappa values
    # TODO: figure out whether using continuous output actually makes sense
    # notes on existing papers:
    #   evaluating question answering evaluatilanthology.org/D19-5817.pdf)
        # they just say that METEOR is the best, but it's not much better than F1 here
    #   semantic answer similarity (https://arxiv.org/abs/2108.06130)
        # SAS does a little better, but it's not perfect (also mostly for open domain)
    # papers tend to measure correlation using Pearson's r and Kendall's tau-b
    correct_df = pd.merge(scores, correct_df['ground_truth'], left_index=True, right_index=True)
    all_eval_df = correct_df.append(fails_evaled_df)

    print("all eval df:")
    print(all_eval_df)

    # plot_metrics_histogram(scores)
    plot_roc(all_eval_df)

    grouped_by_run = all_eval_df.groupby("run")
    for group in grouped_by_run:
        print(grouped_by_run.get_group(group[0]))
        plot_roc(grouped_by_run.get_group(group[0]), filename='out/'+group[0]+'_metricsROC.png')
