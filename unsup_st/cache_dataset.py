import json
from pathlib import Path

import librosa
import numpy as np
from tqdm import tqdm

from unsup_st.dataset import get_mfcc, SR

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str)
    parser.add_argument('--dest', type=str)
    parser.add_argument('--dataset', type=str)
    args = parser.parse_args()

    Path(f'{args.dest}/{args.dataset}').mkdir(parents=True, exist_ok=True)

    with open(Path(args.src) / args.dataset / 'metadata.jsonl') as f:
        for line in tqdm(f):
            data = json.loads(line)
            prefix = data['prefix']
            dest_path = Path(args.dest) / f'{prefix}.mfcc.npy'

            audio, _ = librosa.load(Path(args.src) / f'{prefix}.wav', sr=SR)
            mfcc = get_mfcc(audio, normalize=False)
            mfcc = mfcc.transpose()  # (L, F)
            np.save(dest_path, mfcc)


if __name__ == '__main__':
    main()
