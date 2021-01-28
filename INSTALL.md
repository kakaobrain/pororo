# Installation Guide

This document deals with how to install libraries required for Pororo installation.

<br>

## Common modules

- The libraries that should be installed in common for the use of Pororo are:
    - These libraries are installed when Pororo is installed through the `pip install` command, so you do not have to take additional action

```python
requirements = [
    "torch==1.6.0",
    "torchvision==0.7.0",
    "pillow>=4.1.1",
    "fairseq==0.10.2",
    "transformers>=4.0.0",
    "sentence_transformers==0.4.1.2",
    "nltk==3.5",
    "word2word",
    "wget",
    "joblib",
    "lxml",
    "g2p_en",
    "whoosh",
    "marisa-trie",
    "kss",
    "python-mecab-ko",
]
```

<br>

## Korean

- You may need to install additional libraries to perform specific tasks in Korean.

- `kollocate` is a library needed for the **Korean Collocation** task.

```console
pip install kollocate
```

- `koparadigm` is a library needed for the **Korean Morphological Inflection** task.

```console
pip install koparadigm
```

- `g2pk` is a library needed for the **Korean Grapheme-to-Phoneme** task.

```console
pip install g2pk
```

<br>

## Japanese

- You may need to install additional libraries to perform specific tasks in Japanese.

- `fugashi` and `ipadic` are the libraries needed for the **Japanese RoBERTa** model and the **Japanese PoS Tagging**.

```console
pip install fugashi ipadic
```

- `romkan` is a library needed for the **Japanese Grapheme-to-Phoneme** task.

```console
pip install romkan
```

<br>

## Chinese

- You may need to install additional libraries to perform specific tasks in Chinese.

- `g2pM` is a library needed for the **Chinese Grapheme-to-Phoneme** task.

```console
pip install g2pM
```

- `jieba` is a library needed for the **Chinese PoS Tagging** task.

```console
pip install jieba
```

<br>

## Etc.

### Linux/MacOS Supported Tasks

- Automatic Speech Recognition
- Speech Translation
- Optical Character Recognition
- Image Captioning

<br>

### Automatic Speech Recognition
  
- To utilize the **Automatic Speech Recognition** module, [wav2letter](https://github.com/facebookresearch/wav2letter) is required. `asr-install.sh` can be used for installation of th `wav2letter`

```console
bash asr-install.sh
```

- If you use the **YouTube** link as an input to the **ASR** module, you need to install [youtube-dl](https://github.com/ytdl-org/youtube-dl)

<br>

### Optical Character Recognition

- To utilize the **OCR** module, you need to install the following libraries

```console
apt-get install -y libgl1-mesa-glx
```

```console
pip install opencv-python scikit-image
```
