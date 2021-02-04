from collections import Counter, defaultdict
import json
import math
import random
import os
import sys

import cv2
import librosa
import librosa.display
import numpy as np
import scipy.stats
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


SR = 16000


def get_mfcc(audio):
    n_fft = 512
    hop_length = 160   # sr * .01
    mfcc = librosa.feature.mfcc(audio, sr=SR, n_mfcc=13, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window='hann')

    mean = mfcc.mean(axis=1, keepdims=True)
    std = mfcc.std(axis=1, keepdims=True)
    mfcc = (mfcc - mean) / std
    return mfcc


class LibriMorseDataset(Dataset):
    def __init__(self, metadata, window=3, subsample=True):

        counts = Counter([word for _, offsets in metadata.items() for word, _, _ in offsets])
        total_counts = sum(counts.values())
        freqs = {word: count / total_counts for word, count in counts.items()}
        th = 1.5e-5
        p_drop = Counter({word: 1 - math.sqrt(th / freq) for word, freq in freqs.items()})

        self.xs = []
        self.ys = []
        self.src_words = []
        self.tgt_words = []
        for prefix, offsets in metadata.items():
            print(prefix)

            offsets = [(i, w, st, ed) for i, (w, st, ed) in enumerate(offsets)]
            if subsample:
                offsets_sampled = []
                for i, word, st, ed in offsets:
                    if random.random() > p_drop[word]:
                        offsets_sampled.append((i, word, st, ed))
                offsets = offsets_sampled

            if len(offsets) <= 1:
                print('Skipping short segments')
                continue

            cache_path = f'data/LibriMorse.cache/{prefix}.mfcc.npy'
            if os.path.exists(cache_path):
                mfcc = np.load(cache_path)
            else:
                audio, _ = librosa.load(f'data/LibriMorse/{prefix}.wav', sr=SR)
                mfcc = get_mfcc(audio)
                mfcc = mfcc.transpose()  # (L, F)
                np.save(cache_path, mfcc)

            for i, (src_offset, src_word, src_st, src_ed) in enumerate(offsets):
                x = mfcc[int(src_st*100):int(src_ed*100), :]
                for j in range(i-window, i+window+1):
                    if i == j or j < 0 or j >= len(offsets):
                        continue
                    tgt_offset, tgt_word, tgt_st, tgt_ed = offsets[j]
                    if abs(tgt_offset - src_offset) > 3:
                        continue
                    y = mfcc[int(tgt_st*100):int(tgt_ed*100), :]

                    self.xs.append(x)
                    self.ys.append(y)
                    self.src_words.append(src_word)
                    self.tgt_words.append(tgt_word)

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return (self.xs[idx], self.ys[idx], self.src_words[idx], self.tgt_words[idx])


class Speech2Vec(nn.Module):
    def __init__(self, input_size=13, hidden_size=256, scale_factor=None):
        super(Speech2Vec, self).__init__()

        self.scale_factor = scale_factor

        self.hidden_size = hidden_size

        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0,
            bidirectional=True
        )

        self.projection = nn.Linear(2*hidden_size, hidden_size)

        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0,
            bidirectional=False
        )

        self.head = nn.Linear(hidden_size, input_size)

        self.loss_func = nn.MSELoss(reduction='none')

    def forward(self, xs, xs_len, ys=None, ys_len=None, ys_mask=None):
        if self.scale_factor is not None:
            xs = nn.functional.interpolate(xs.unsqueeze(1), scale_factor=(self.scale_factor, 1), mode='bilinear').squeeze(1)
            if ys is not None:
                ys = nn.functional.interpolate(ys.unsqueeze(1), scale_factor=(self.scale_factor, 1), mode='bilinear').squeeze(1)
            if ys_mask is not None:
                ys_mask = nn.functional.interpolate(ys_mask.unsqueeze(1), scale_factor=(self.scale_factor, 1), mode='bilinear').squeeze(1)

            xs_len = [l * self.scale_factor for l in xs_len]
            if ys_len is not None:
                ys_len = [l * self.scale_factor for l in ys_len]

        xs = nn.utils.rnn.pack_padded_sequence(xs, xs_len, batch_first=True, enforce_sorted=False)

        _, (embed, _) = self.encoder(xs)
        embed = torch.cat((embed[0], embed[1]), dim=1)
        embed = self.projection(embed)

        loss = None
        out = None
        if ys is not None:
            ys_len_max = ys.shape[1]
            decoder_input = embed.view(-1, 1, self.hidden_size).repeat(1, ys_len_max, 1)
            decoder_input = nn.utils.rnn.pack_padded_sequence(decoder_input, ys_len, batch_first=True, enforce_sorted=False)
            decoder_h0 = embed.unsqueeze(0)
            decoder_c0 = torch.zeros_like(decoder_h0)

            out, _ = self.decoder(decoder_input, (decoder_h0, decoder_c0))
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
            out = self.head(out)

            loss = self.loss_func(out, ys)
            loss = (loss * ys_mask).sum(dim=1) / ys_mask.sum(dim=1)    # mean time-wise
            loss = loss.mean()                     # mean along batch & feature dims

        return loss, embed, out


def pad_collate(batch):
    xs, ys, src_words, tgt_words = zip(*batch)

    xs = [torch.tensor(x) for x in xs]
    ys = [torch.tensor(y) for y in ys]
    xs_len = [len(x) for x in xs]
    ys_len = [len(y) for y in ys]
    ys_mask = [torch.ones((len(y), 1)) for y in ys]

    xs = nn.utils.rnn.pad_sequence(xs, batch_first=True)
    ys = nn.utils.rnn.pad_sequence(ys, batch_first=True)
    ys_mask = nn.utils.rnn.pad_sequence(ys_mask, batch_first=True)

    return (xs, xs_len, ys, ys_len, ys_mask, src_words, tgt_words)


def _dump_vec(vec):
    print(''.join(('1' if x > 0 else '0') for x in vec[:, 0]))


def main():
    random.seed(42)

    device = torch.device('cuda:0')

    metadata = {}
    for line in sys.stdin:
        data = json.loads(line)
        metadata[data['prefix']] = data['offsets']

    print(f'total: {len(metadata)} files')

    dataset = LibriMorseDataset(metadata, window=3, subsample=True)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=2048,
        shuffle=True,
        collate_fn=pad_collate)

    model = Speech2Vec(hidden_size=256, scale_factor=.2)
    model = model.to(device)
    optimizer = optim.Adam(params=model.parameters(), lr=1e-3)

    for epoch in range(500):

        # train
        model.train()

        train_loss = 0.
        train_steps = 0

        for step, batch in enumerate(dataloader):
            xs, xs_len, ys, ys_len, ys_mask, src_words, tgt_words = batch
            xs = xs.to(device)
            ys = ys.to(device)
            ys_mask = ys_mask.to(device)
            optimizer.zero_grad()

            loss, embed, out = model(xs, xs_len, ys, ys_len, ys_mask)
            loss.backward()

            train_loss += loss.cpu()
            train_steps += 1
            optimizer.step()

            if step % 100 == 0:
                print(f'epoch = {epoch}, step = {step}, train_loss = {train_loss / train_steps}')
                out = out[0].transpose(0, 1).unsqueeze(-1).repeat(1, 1, 3).detach().cpu().numpy()
                out = (out + .1) * 255.
                cv2.imwrite(f'log/out-{epoch:03d}-{step:03d}.png', out)
                ys = nn.functional.interpolate(ys.unsqueeze(1), scale_factor=(model.scale_factor, 1), mode='bilinear').squeeze(1)
                ys = ys[0].transpose(0, 1).unsqueeze(-1).repeat(1, 1, 3).detach().cpu().numpy()
                ys = (ys + .1) * 255.
                cv2.imwrite(f'log/ys-{epoch:03d}-{step:03d}.png', ys)

        print(f'epoch = {epoch}, train_loss = {train_loss / train_steps}')

        if epoch % 5 == 0:
            # eval
            model.eval()

            embeds = defaultdict(list)
            for batch in dataloader:
                xs, xs_len, _, _, _, src_words, _ = batch
                xs = xs.to(device)

                with torch.no_grad():
                    _, embed, _ = model(xs, xs_len)

                for word, vec in zip(src_words, embed):
                    embeds[word].append(vec.detach())

            embeds_mean = {word: torch.vstack(vecs).mean(dim=0) for word, vecs in embeds.items()}

            cos = nn.CosineSimilarity(dim=0)
            golds = []
            preds = []

            with open('data/SimLex-999/SimLex-999.txt') as f:
                next(f)
                for line in f:
                    w1, w2, _, gold = line.split('\t')[:4]
                    w1 = w1.upper()
                    w2 = w2.upper()
                    gold = float(gold)
                    if w1 in embeds_mean and w2 in embeds_mean:
                        pred = cos(embeds_mean[w1], embeds_mean[w2]).item()

                        golds.append(gold)
                        preds.append(pred)

            corr, _ = scipy.stats.pearsonr(golds, preds)
            print(corr)

if __name__ == '__main__':
    main()
