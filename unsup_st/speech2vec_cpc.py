from collections import defaultdict
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from unsup_st.evaluate_embed import evaluate_embed, save_embed
from unsup_st.dataset import LibriMorseDataset, build_vocabulary
from unsup_st.models import Speech2VecRNN, Speech2VecXF


def pad_collate(batch):
    xs, ys, src_words, tgt_words = zip(*batch)

    xs = [torch.tensor(x) for x in xs]
    ys = [torch.tensor(y) for y in ys]
    xs_len = [len(x) for x in xs]
    ys_len = [len(y) for y in ys]

    xs = nn.utils.rnn.pad_sequence(xs, batch_first=True)
    ys = nn.utils.rnn.pad_sequence(ys, batch_first=True)

    return (xs, xs_len, ys, ys_len, src_words, tgt_words)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['rnn', 'transformer'])
    parser.add_argument('--datadir', type=str)
    parser.add_argument('--outdir', type=str)
    parser.add_argument('--train-dataset', type=str)
    parser.add_argument('--valid-dataset', type=str)
    parser.add_argument('--hidden-size', type=int)
    parser.add_argument('--layers', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--additive-margin', type=float)
    args = parser.parse_args()

    random.seed(42)

    device = torch.device('cuda:0')

    train_metadata = {}
    for path in args.train_dataset.split(','):
        with open(path) as f:
            for line in f:
                data = json.loads(line)
                train_metadata[data['prefix']] = data['offsets']
    
    valid_metadata = {}
    with open(args.valid_dataset) as f:
        for line in f:
            data = json.loads(line)
            valid_metadata[data['prefix']] = data['offsets']

    print(f'train_dataset: {len(train_metadata)} files')
    print(f'valid_dataset: {len(valid_metadata)} files')

    all_metadata = {}
    all_metadata.update(train_metadata)
    all_metadata.update(valid_metadata)
    vocab = build_vocabulary(all_metadata, min_freq=3)

    train_dataset = LibriMorseDataset(train_metadata, datadir=args.datadir, vocab=vocab, window=3, subsample=True, load_audio=True)
    valid_dataset = LibriMorseDataset(valid_metadata, datadir=args.datadir, vocab=vocab, window=3, subsample=True, load_audio=True)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate)

    xys = []
    for i in range(10):
        xs, ys, _, _ = train_dataset[i]
        xys.append(xs)
        xys.append(ys)
    xys = np.concatenate(xys, axis=0)
    mean = torch.tensor(xys.mean(axis=0)).to(device)
    std = torch.tensor(xys.std(axis=0)).to(device)

    if args.model == 'rnn':
        model = Speech2VecRNN(
            hidden_size=args.hidden_size,
            hidden_channels=(32, 48),
            scale_factor=.5,
            mean=mean,
            std=std,
            additive_margin=args.additive_margin)
    elif args.model == 'transformer':
        model = Speech2VecXF(
            hidden_size=args.hidden_size,
            layers=args.layers,
            scale_factor=.5,
            mean=mean,
            std=std,
            additive_margin=args.additive_margin)
    else:
        raise ValueError(f'Invalid model type: {args.model}')
    model = model.to(device)

    optimizer = optim.Adam(params=model.parameters(), lr=args.lr)

    for epoch in range(151):

        # train
        model.train()

        train_loss = 0.
        train_steps = 0

        for step, batch in enumerate(train_dataloader):
            xs, xs_len, ys, ys_len, src_words, tgt_words = batch
            xs = xs.to(device)
            ys = ys.to(device)
            optimizer.zero_grad()

            loss, embed = model(xs, xs_len, ys, ys_len)
            loss.backward()

            train_loss += loss.cpu()
            train_steps += 1
            optimizer.step()

        print(f'epoch = {epoch}, train_loss = {train_loss / train_steps}')

        if epoch > 0 and epoch % 5 == 0:
            # eval
            model.eval()

            embeds = defaultdict(list)
            for batch in train_dataloader:
                xs, xs_len, _, _, src_words, _ = batch
                xs = xs.to(device)

                with torch.no_grad():
                    _, embed = model(xs, xs_len)
                    embed = embed.detach().cpu()

                for wid, vec in zip(src_words, embed):
                    word = vocab.id2word[wid]
                    embeds[word].append(vec.detach())

            embeds_mean = {word: torch.vstack(vecs).mean(dim=0) for word, vecs in embeds.items()}

            evaluate_embed(embeds_mean)
            save_embed(embeds_mean, f'{args.outdir}/epoch{epoch:03}.vec')

if __name__ == '__main__':
    main()
