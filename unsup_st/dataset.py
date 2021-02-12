from collections import Counter, namedtuple
import math
import os
import random


import librosa
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

SR = 16000
FRAMES_PER_SEC = 50
Vocabulary = namedtuple('Vocabulary', 'p_drop word2id id2word')

def get_mfcc(audio, normalize=False):
    n_fft = 512
    hop_length = SR // FRAMES_PER_SEC
    mfcc = librosa.feature.mfcc(
        audio,
        sr=SR,
        n_mfcc=13,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window='hann')

    if normalize:
        mean = mfcc.mean(axis=1, keepdims=True)
        std = mfcc.std(axis=1, keepdims=True)
        mfcc = (mfcc - mean) / std
    return mfcc


def build_vocabulary(metadata, min_freq=3):
    counts = Counter([word for _, offsets in metadata.items() for word, _, _ in offsets])
    total_counts = sum(counts.values())
    freqs = {word: count / total_counts for word, count in counts.items()}
    th = 1.5e-5
    p_drop = Counter({word: 1 - math.sqrt(th / freq) for word, freq in freqs.items()})

    word2id = {}
    id2word = {}
    wid = 0
    for word, count in counts.items():
        if count >= min_freq:
            word2id[word] = wid
            id2word[wid] = word
            wid += 1

    return Vocabulary(p_drop, word2id, id2word)


class LibriMorseDataset(Dataset):
    def __init__(self, metadata, vocab, window=3, subsample=True, load_audio=True):
        self.load_audio = load_audio

        self.xs = []
        self.ys = []
        self.src_words = []
        self.tgt_words = []
        for prefix, offsets in tqdm(metadata.items()):

            offsets = [(i, w, st, ed) for i, (w, st, ed) in enumerate(offsets)]
            if subsample:
                offsets_sampled = []
                for i, word, st, ed in offsets:
                    if random.random() > vocab.p_drop[word]:
                        offsets_sampled.append((i, word, st, ed))
                offsets = offsets_sampled

            if len(offsets) <= 1:
                continue

            chosen_pairs = []
            for i, (src_offset, src_word, src_st, src_ed) in enumerate(offsets):
                for j in range(i-window, i+window+1):
                    if i == j or j < 0 or j >= len(offsets):
                        continue
                    tgt_offset, tgt_word, tgt_st, tgt_ed = offsets[j]
                    if abs(tgt_offset - src_offset) > 3:
                        continue
                    if src_word in vocab.word2id and tgt_word in vocab.word2id:
                        chosen_pairs.append((
                            (src_st, src_ed),
                            (tgt_st, tgt_ed)
                        ))
                        self.src_words.append(vocab.word2id[src_word])
                        self.tgt_words.append(vocab.word2id[tgt_word])

            if not chosen_pairs:
                continue

            if load_audio:
                cache_path = f'data/LibriMorse.cache/{prefix}.mfcc.npy'
                if os.path.exists(cache_path):
                    mfcc = np.load(cache_path)
                else:
                    audio, _ = librosa.load(f'data/LibriMorse/{prefix}.wav', sr=SR)
                    mfcc = get_mfcc(audio, normalize=False)
                    mfcc = mfcc.transpose()  # (L, F)
                    np.save(cache_path, mfcc)

                for (src_st, src_ed), (tgt_st, tgt_ed) in chosen_pairs:
                    x = mfcc[int(src_st*FRAMES_PER_SEC):int(src_ed*FRAMES_PER_SEC), :]
                    y = mfcc[int(tgt_st*FRAMES_PER_SEC):int(tgt_ed*FRAMES_PER_SEC), :]

                    self.xs.append(x)
                    self.ys.append(y)

                assert len(self.xs) == len(self.ys) == len(self.src_words) == len(self.tgt_words)

    def __len__(self):
        return len(self.src_words)

    def __getitem__(self, idx):
        if self.load_audio:
            return (self.xs[idx], self.ys[idx], self.src_words[idx], self.tgt_words[idx])
        
        return (self.src_words[idx], self.tgt_words[idx])
