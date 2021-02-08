from collections import defaultdict
import json
import random
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from unsup_st.dataset import LibriMorseDataset, build_vocabulary
from unsup_st.evaluate_embed import evaluate_embed


class Word2Vec(nn.Module):
    def __init__(self, num_embeddings, hidden_size=100, use_bidirectional_loss=False):
        super(Word2Vec, self).__init__()

        self.hidden_size = hidden_size
        self.src_embeddings = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=hidden_size)
        self.tgt_embeddings = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=hidden_size)

        self.loss_func = nn.CrossEntropyLoss()

        self.use_bidirectional_loss = use_bidirectional_loss

    def forward(self, src_words, tgt_words=None):

        xs_embed = self.src_embeddings(src_words)

        loss = None
        if tgt_words is not None:
            ys_embed = self.tgt_embeddings(tgt_words)

            batch_size = xs_embed.shape[0]
            pred = torch.mm(xs_embed, ys_embed.transpose(0, 1))
            gold = torch.arange(start=0, end=batch_size, device=pred.device)
            loss = self.loss_func(pred, gold)

            if self.use_bidirectional_loss:
                loss_backward = self.loss_func(pred.transpose(0, 1), gold)
                loss = (loss + loss_backward) / 2.

        return loss, xs_embed


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

    train_dataset = LibriMorseDataset(train_metadata, vocab=vocab, window=3, subsample=True, load_audio=False)
    valid_dataset = LibriMorseDataset(valid_metadata, vocab=vocab, window=3, subsample=True, load_audio=False)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, shuffle=False)

    model = Word2Vec(
        num_embeddings=len(vocab.word2id),
        hidden_size=args.hidden_size,
        use_bidirectional_loss=False)
    model = model.to(device)

    optimizer = optim.Adam(params=model.parameters(), lr=args.lr)

    for epoch in range(500):

        # train
        model.train()

        train_loss = 0.
        train_steps = 0

        for batch in train_dataloader:
            src_words, tgt_words = batch
            src_words = src_words.to(device)
            tgt_words = tgt_words.to(device)
            optimizer.zero_grad()

            loss, _ = model(src_words, tgt_words)
            loss.backward()

            train_loss += loss.cpu()
            train_steps += 1
            optimizer.step()

        print(f'epoch = {epoch}, train_loss = {train_loss / train_steps}')

        if epoch > 0 and epoch % 5 == 0:
            # eval
            model.eval()

            # compute loss
            valid_loss = 0.
            valid_steps = 0
            for batch in valid_dataloader:
                src_words, tgt_words = batch
                src_words = src_words.to(device)
                tgt_words = tgt_words.to(device)

                with torch.no_grad():
                    loss, _ = model(src_words, tgt_words)

                valid_loss += loss.cpu()
                valid_steps += 1

            print(f'epoch = {epoch}, valid_loss = {valid_loss / valid_steps}')

            # compute embeddings
            embeds = defaultdict(list)
            for batch in train_dataloader:
                src_words, _ = batch
                src_words = src_words.to(device)

                with torch.no_grad():
                    _, embed = model(src_words)

                for wid, vec in zip(src_words, embed):
                    wid = wid.detach().item()
                    word = vocab.id2word[wid]
                    if len(embeds[word]) < 100:
                        embeds[word].append(vec.detach())

            embeds_mean = {}
            for word, vecs in embeds.items():
                embeds_mean[word] = torch.vstack(vecs).mean(dim=0)
            evaluate_embed(embeds_mean)

if __name__ == '__main__':
    main()
