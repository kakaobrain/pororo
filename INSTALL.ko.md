# 설치 가이드

본 문서에서는 Pororo 설치를 위해 필요한 라이브러리에 대한 설명과 설치 방법을 다룹니다.

<br>

## 공통 모듈

- Pororo 사용을 위해 공통적으로 설치되어야 할 라이브러리는 다음과 같습니다.
- 해당 라이브러리들은 `pip install` 명령어를 통해 Pororo가 설치될 때 공통적으로 설치되므로, 추가적인 조치를 취해주지 않으셔도 됩니다.

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

## 한국어

- 한국어의 특정 태스크를 수행하기 위해서는 추가적인 라이브러리를 설치할 필요가 있을 수 있습니다.

- `kollocate`는 **한국어 Collocation** 태스크의 수행을 위해 필요한 라이브러리입니다.

```console
pip install kollocate
```

- `koparadigm`는 **한국어 Morphological Inflection** 태스크의 수행을 위해 필요한 라이브러리입니다.

```console
pip install koparadigm
```

- `g2pk`는 **한국어 Grapheme-to-Phoneme** 태스크의 수행을 위해 필요한 라이브러리입니다.

```console
pip install g2pk
```

<br>

## 일본어

- 일본어의 특정 태스크를 수행하기 위해서는 추가적인 라이브러리를 설치할 필요가 있을 수 있습니다.

- `fugashi`와 `ipadic`은 **일본어 RoBERTa** 모델의 토크나이즈와 **일본어 PoS Tagging**을 위해 필요한 라이브러리입니다.

```console
pip install fugashi ipadic
```

- `romkan`은 **일본어 Grapheme-to-Phoneme** 태스크의 수행을 위해 필요한 라이브러리입니다.

```console
pip install romkan
```

<br>

## 중국어

- 중국어의 특정 태스크를 수행하기 위해서는 추가적인 라이브러리를 설치할 필요가 있을 수 있습니다.

- `g2pM`은 **중국어 Grapheme-to-Phoneme** 태스크의 수행을 위해 필요한 라이브러리입니다.

```console
pip install g2pM
```

- `jieba`는 **중국어 PoS Tagging** 태스크의 수행을 위해 필요한 라이브러리입니다.

```console
pip install jieba
```

<br>

## 기타

### Linux/MacOS 지원 태스크

- Automatic Speech Recognition
- Speech Translation
- Optical Character Recognition
- Image Captioning

<br>

### Automatic Speech Recognition
  
- 음성인식 모듈을 활용하기 위해서는 [wav2letter](https://github.com/facebookresearch/wav2letter) 설치가 필요합니다. 레포지토리의 `asr-install.sh`를 실행함으로써 `wav2letter` 설치가 가능합니다.

```console
bash asr-install.sh
```

- 음성인식 모듈의 입력으로 **YouTube** 링크를 사용할 경우 [youtube-dl](https://github.com/ytdl-org/youtube-dl) 라이브러리 설치가 필요합니다. 

<br>

### Optical Character Recognition

- OCR 모듈을 활용하기 위해서는 아래 라이브러리들을 설치해주셔야 합니다.

```console
apt-get install -y libgl1-mesa-glx
```

```console
pip install opencv-python scikit-image
```