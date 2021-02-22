from collections import Counter
import csv

import torch.nn as nn
import scipy.stats

BENCHMARKS = {
    'mc-30': 'word-benchmarks/word-similarity/monolingual/en/mc-30.csv',
    'rg-65': 'word-benchmarks/word-similarity/monolingual/en/rg-65.csv',
    'wordsim353-sim': 'word-benchmarks/word-similarity/monolingual/en/wordsim353-sim.csv',
    'men': 'word-benchmarks/word-similarity/monolingual/en/men.csv'
}

SAMPLES = ['CAR', 'FRUIT', 'HAPPY', 'JUMP']

def evaluate_embed(embed):
    cos = nn.CosineSimilarity(dim=0)

    for query in SAMPLES:
        v_query = embed[query]
        sims = Counter()
        for target, v_target in embed.items():
            if query == target:
                continue

            sims[target] = cos(v_query, v_target).item()

        print(query, ' - ', ' '.join(f'{target}:{sim:0.2f}' for target, sim in sims.most_common(10)))

    for benchmark, filepath in BENCHMARKS.items():
        golds = []
        preds = []

        total_rows = 0
        with open(filepath) as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                _, w1, w2, gold = row
                if not gold:
                    continue
                if '-' in w1:
                    w1 = w1.split('-')[0]
                if '-' in w2:
                    w2 = w2.split('-')[0]
                total_rows += 1

                w1 = w1.upper()
                w2 = w2.upper()
                gold = float(gold)
                if w1 in embed and w2 in embed:
                    pred = cos(embed[w1], embed[w2]).item()

                    golds.append(gold)
                    preds.append(pred)

        corr, _ = scipy.stats.pearsonr(golds, preds)
        print(f'benchmark = {benchmark}, corr = {corr}, # valid = {len(preds)} / {total_rows}')


def save_embed(embed, path):
    with open(path, mode='w') as f:
        for word, vec in embed.items():
            values = [f'{v:.4f}' for v in vec]
            f.write(word)
            f.write(' ')
            f.write(' '.join(values))
            f.write('\n')
