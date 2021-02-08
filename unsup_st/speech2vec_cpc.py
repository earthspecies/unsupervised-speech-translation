from collections import defaultdict
import json
import random

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from unsup_st.evaluate_embed import evaluate_embed
from unsup_st.dataset import LibriMorseDataset, build_vocabulary


class Speech2Vec(nn.Module):
    def __init__(self, input_size=13, hidden_size=100, scale_factor=None, use_bidirectional_loss=False):
        super(Speech2Vec, self).__init__()

        self.scale_factor = scale_factor

        self.hidden_size = hidden_size

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=32, out_channels=48, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(48),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=48, out_channels=hidden_size, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True)
        )

        self.encoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0,
            bidirectional=True
        )

        self.projection = nn.Linear(hidden_size, hidden_size)

        self.loss_func = nn.CrossEntropyLoss()

        self.use_bidirectional_loss = use_bidirectional_loss

    def forward(self, xs, xs_len, ys=None, ys_len=None):
        if self.scale_factor is not None:
            xs = nn.functional.interpolate(xs.unsqueeze(1), scale_factor=(self.scale_factor, 1), mode='bilinear').squeeze(1)
            if ys is not None:
                ys = nn.functional.interpolate(ys.unsqueeze(1), scale_factor=(self.scale_factor, 1), mode='bilinear').squeeze(1)

            xs_len = [l * self.scale_factor for l in xs_len]
            if ys_len is not None:
                ys_len = [l * self.scale_factor for l in ys_len]

        xs = self.cnn(xs.transpose(1, 2)).transpose(1, 2)
        xs_len = [l // 8 for l in xs_len]
        xs = nn.utils.rnn.pack_padded_sequence(xs, xs_len, batch_first=True, enforce_sorted=False)

        _, (xs_embed, _) = self.encoder(xs)
        # xs_embed = torch.cat((xs_embed[0], xs_embed[1]), dim=1)
        xs_embed = xs_embed[0] + xs_embed[1]

        loss = None
        if ys is not None:
            ys = self.cnn(ys.transpose(1, 2)).transpose(1, 2)
            ys_len = [l // 8 for l in ys_len]
            ys = nn.utils.rnn.pack_padded_sequence(ys, ys_len, batch_first=True, enforce_sorted=False)
            _, (ys_embed, _) = self.encoder(ys)
            # ys_embed = torch.cat((ys_embed[0], ys_embed[1]), dim=1)
            ys_embed = ys_embed[0] + ys_embed[1]
            ys_embed = self.projection(ys_embed)

            batch_size = xs_embed.shape[0]
            pred = torch.mm(xs_embed, ys_embed.transpose(0, 1))
            gold = torch.arange(start=0, end=batch_size, device=pred.device)
            loss = self.loss_func(pred, gold)

            if self.use_bidirectional_loss:
                loss_backward = self.loss_func(pred.transpose(0, 1), gold)
                loss = (loss + loss_backward) / 2.

        return loss, xs_embed


def pad_collate(batch):
    xs, ys, src_words, tgt_words = zip(*batch)

    xs = [torch.tensor(x) for x in xs]
    ys = [torch.tensor(y) for y in ys]
    xs_len = [len(x) for x in xs]
    ys_len = [len(y) for y in ys]

    xs = nn.utils.rnn.pad_sequence(xs, batch_first=True)
    ys = nn.utils.rnn.pad_sequence(ys, batch_first=True)

    return (xs, xs_len, ys, ys_len, src_words, tgt_words)


def _dump_vec(vec):
    return ''.join(('1' if x > 0 else '0') for x in vec)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train-dataset', type=str)
    parser.add_argument('--valid-dataset', type=str)
    parser.add_argument('--hidden-size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--batch-size', type=int)
    args = parser.parse_args()

    random.seed(42)

    device = torch.device('cuda:0')

    train_metadata = {}
    with open(args.train_dataset) as f:
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

    train_dataset = LibriMorseDataset(train_metadata, vocab=vocab, window=3, subsample=True, load_audio=True)
    valid_dataset = LibriMorseDataset(valid_metadata, vocab=vocab, window=3, subsample=True, load_audio=True)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate)

    model = Speech2Vec(
        hidden_size=args.hidden_size,
        scale_factor=.5,
        use_bidirectional_loss=False)
    model = model.to(device)

    optimizer = optim.Adam(params=model.parameters(), lr=args.lr)

    for epoch in range(500):

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

            # if step % 100 == 0:
            #     print(f'epoch = {epoch}, step = {step}, train_loss = {train_loss / train_steps}, word={tgt_words[0]}')
            #     ys = nn.functional.interpolate(ys.unsqueeze(1), scale_factor=(model.scale_factor, 1), mode='bilinear').squeeze(1)
            #     ys = ys[0].transpose(0, 1).unsqueeze(-1).repeat(1, 1, 3).detach().cpu().numpy()
            #     ys = (ys + .1) * 255.
            #     cv2.imwrite(f'log/ys-{epoch:03d}-{step:03d}.png', ys)

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

                for wid, vec in zip(src_words, embed):
                    word = vocab.id2word[wid]
                    embeds[word].append(vec.detach())

            embeds_mean = {word: torch.vstack(vecs).mean(dim=0) for word, vecs in embeds.items()}

            evaluate_embed(embeds_mean)

if __name__ == '__main__':
    main()
