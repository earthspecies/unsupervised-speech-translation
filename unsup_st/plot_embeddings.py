import json

import numpy as np
import plotly
import plotly.graph_objects as go
from sklearn.manifold import TSNE

from unsup_st.dataset import build_vocabulary


def read_embeds(path):
    embeds = {}
    with open(path) as f:
        for line in f:
            fields = line.strip().split(' ')
            word, *vec = fields
            vec = np.array([float(v) for v in vec])

            embeds[word] = vec
    return embeds


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str)
    parser.add_argument('--output-file', type=str)
    parser.add_argument('--train-dataset', type=str)
    args = parser.parse_args()

    # read metadata
    metadata = {}
    for path in args.train_dataset.split(','):
        with open(path) as f:
            for line in f:
                data = json.loads(line)
                metadata[data['prefix']] = data['offsets']

    vocab = build_vocabulary(metadata, min_freq=100)

    embeds = read_embeds(args.input_file)

    words = []
    vecs = []

    for word, vec in embeds.items():
        if word in vocab.word2id:
            words.append(word)
            vecs.append(vec)

    model = TSNE(n_components=2, init='pca', random_state=0)
    coordinates = model.fit_transform(vecs)

    xs, ys = zip(*coordinates)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            text=words,
            textposition='bottom right',
            mode='markers+text'))
    # fig.show()
    plotly.offline.plot(fig, filename=args.output_file, auto_open=False)


if __name__ == '__main__':
    main()
