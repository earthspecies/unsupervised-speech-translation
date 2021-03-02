# Unsupervised Speech Translation

## Prerequisites

```
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

## Datasets

```
mkdir data; cd data
wget https://www.openslr.org/resources/12/dev-clean.tar.gz
tar zxvf dev-clean.tar.gz
wget https://www.openslr.org/resources/12/train-clean-100.tar.gz
tar zxvf train-clean-100.tar.gz
wget https://www.openslr.org/resources/12/train-clean-360.tar.gz
tar zxvf train-clean-360.tar.gz
```

## Generate Morse code datasets

```
python unsup_st/morse.py --src data/LibriSpeech --dest data/LibriMorse --dataset dev-clean
python unsup_st/morse.py --src data/LibriSpeech --dest data/LibriMorse --dataset train-clean-100
python unsup_st/morse.py --src data/LibriSpeech --dest data/LibriMorse --dataset train-clean-360
```

## Convert datasets into features

```
python unsup_st/cache_dataset.py --src data/LibriMorse --dest data/LibriMorse.cache --dataset dev-clean
python unsup_st/cache_dataset.py --src data/LibriMorse --dest data/LibriMorse.cache --dataset train-clean-100
python unsup_st/cache_dataset.py --src data/LibriMorse --dest data/LibriMorse.cache --dataset train-clean-360
```
