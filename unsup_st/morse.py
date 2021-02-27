import glob
import json

import numpy as np
from pathlib import Path
import soundfile as sf
from tqdm import tqdm

CHAR2CODE = {
    'A':'. -',
    'B':'- . . .',
    'C':'- . - .',
    'D':'- . .',
    'E':'.',
    'F':'. . - .',
    'G':'- - .',
    'H':'. . . .',
    'I':'. .',
    'J':'. - - -',
    'K':'- . -',
    'L':'. - . .',
    'M':'- -',
    'N':'- .',
    'O':'- - -',
    'P':'. - - .',
    'Q':'- - . -',
    'R':'. - .',
    'S':'. . .',
    'T':'-',
    'U':'. . -',
    'V':'. . . -',
    'W':'. - -',
    'X':'- . . -',
    'Y':'- . - -',
    'Z':'- - . .',
    ' ':'       ',
}
# dot=1
# dash=3
# space=1
# between letters = 3
# between words = 7

SR = 16000

unit = 0.05
u = np.linspace(0, unit, int(unit*SR))
u3 = np.linspace(0, unit*3, int(unit*SR*3))
dot = np.sin(2 * np.pi * 880 * u)
dash = np.sin(2 * np.pi * 880 * u3)
space = np.zeros_like(u)

CODE2AUDIO = {
    '.': dot,
    '-': dash,
    ' ': space,
}

def text2code(text):
    words = text.upper().split(' ')
    codes = ['   '.join(CHAR2CODE[c] for c in word) for word in words]
    margin = 3
    prev_end = -7 + margin
    offsets = []
    for word, code in zip(words, codes):
        st, ed = (prev_end + 7, prev_end + 7 + len(code.replace('-', '...')))
        offsets.append((word, st - margin, ed + margin))
        prev_end = ed
    code = '       '.join(codes)
    code = ' '*margin + code + ' '*margin
    return code, offsets

def code2audio(code):
    audio = [CODE2AUDIO[c] for c in code]
    audio = np.concatenate(audio)
    return audio
    
def text2audio(text):
    code, offsets = text2code(text)
    audio = code2audio(code)
    offsets = [(word, st * unit, ed * unit) for word, st, ed in offsets]
    return audio, offsets


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str)
    parser.add_argument('--dest', type=str)
    parser.add_argument('--dataset', type=str)
    args = parser.parse_args()

    Path(f'{args.dest}/{args.dataset}').mkdir(parents=True, exist_ok=True)

    metadata = {}
    for filename in tqdm(glob.glob(f'{args.src}/{args.dataset}/**/*.trans.txt', recursive=True)):
        with open(filename) as f:
            for line in f:
                prefix, trans = line.strip().split(' ', maxsplit=1)
                trans = trans.lower()
                trans = trans.replace("'", '')     # TODO: remove apostrophes .. is this good?
                audio, offsets = text2audio(trans)
                sf.write(f'{args.dest}/{args.dataset}/{prefix}.wav', audio, SR)
                metadata[f'{args.dataset}/{prefix}'] = offsets

    with open(f'{args.dest}/{args.dataset}/metadata.jsonl', mode='w') as f:
        for prefix, offsets in metadata.items():
            f.write(json.dumps({'prefix': prefix, 'offsets': offsets}))
            f.write('\n')


if __name__ == '__main__':
    main()
