# PORORO: Platform Of neuRal mOdels for natuRal language prOcessing

<p align="center">
  <a href="https://github.com/kakaobrain/pororo/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/kakaobrain/pororo.svg" /></a>
  <a href="https://github.com/kakaobrain/pororo/blob/master/LICENSE"><img alt="Apache 2.0" src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" /></a>
  <a href="https://kakaobrain.github.io/pororo/"><img alt="Docs" src="https://img.shields.io/badge/docs-passing-success.svg" /></a>
  <a href="https://github.com/kakaobrain/pororo/issues"><img alt="Issues" src="https://img.shields.io/github/issues/kakaobrain/pororo" /></a>
</p>

<br>

![](assets/usage.gif)

자연어 처리와 음성 관련 태스크를 쉽게 수행할 수 있도록 도와주는 라이브러리 `pororo` 입니다.

자연어 처리 및 음성 분야 내 다양한 서브 태스크들을 태스크명을 입력하는 것만으로 쉽게 해결할 수 있습니다.

<br>

## Installation

- `torch==1.6(cuda 10.1)`과 `python>=3.6` 환경에서 정상적으로 동작합니다.

- 아래 커맨드를 통해 패키지를 설치하실 수 있습니다.

```console
pip install pororo
```

- 혹은 아래와 같이 **로컬 환경**에서 설치를 하실 수도 있습니다.

```console
git clone https://github.com/kakaobrain/pororo.git
cd pororo
pip install -e .
```

- **공통 모듈** 외 특정 태스크를 위한 라이브러리 설치를 위해서는 [INSTALL.md](INSTALL.ko.md)를 참조해주세요.

- **Automatic Speech Recognition**의 활용을 위해서는 [_wav2letter_](https://github.com/facebookresearch/wav2letter)를 별도로 설치해주어야 합니다. 해당 라이브러리 설치를 위해서는 [asr-install.sh](asr-install.sh) 파일을 실행시켜주세요.

```console
bash asr-install.sh
```

- **Speech Synthesis**의 활용을 위해서는 [tts-install.sh](tts-install.sh) 파일을 실행시켜주세요.

```console
bash tts-install.sh
```

<br>

## Usage

- 다음과 같은 명령어로 `Pororo` 를 사용할 수 있습니다.
- 먼저, `Pororo` 를 임포트하기 위해서는 다음과 같은 명령어를 실행하셔야 합니다:

```python
>>> from pororo import Pororo
```

- 임포트 이후에는, 다음 명령어를 통해 현재 `Pororo` 에서 지원하고 있는 태스크를 확인하실 수 있습니다.

```python
>>> from pororo import Pororo
>>> Pororo.available_tasks()
"Available tasks are ['mrc', 'rc', 'qa', 'question_answering', 'machine_reading_comprehension', 'reading_comprehension', 'sentiment', 'sentiment_analysis', 'nli', 'natural_language_inference', 'inference', 'fill', 'fill_in_blank', 'fib', 'para', 'pi', 'cse', 'contextual_subword_embedding', 'similarity', 'sts', 'semantic_textual_similarity', 'sentence_similarity', 'sentvec', 'sentence_embedding', 'sentence_vector', 'se', 'inflection', 'morphological_inflection', 'g2p', 'grapheme_to_phoneme', 'grapheme_to_phoneme_conversion', 'w2v', 'wordvec', 'word2vec', 'word_vector', 'word_embedding', 'tokenize', 'tokenise', 'tokenization', 'tokenisation', 'tok', 'segmentation', 'seg', 'mt', 'machine_translation', 'translation', 'pos', 'tag', 'pos_tagging', 'tagging', 'const', 'constituency', 'constituency_parsing', 'cp', 'pg', 'collocation', 'collocate', 'col', 'word_translation', 'wt', 'summarization', 'summarisation', 'text_summarization', 'text_summarisation', 'summary', 'gec', 'review', 'review_scoring', 'lemmatization', 'lemmatisation', 'lemma', 'ner', 'named_entity_recognition', 'entity_recognition', 'zero-topic', 'dp', 'dep_parse', 'caption', 'captioning', 'asr', 'speech_recognition', 'st', 'speech_translation', 'ocr', 'srl', 'semantic_role_labeling', 'p2g', 'aes', 'essay', 'qg', 'question_generation', 'age_suitability']"
```

- 태스크 별로 어떠한 모델이 지원되는지 확인하기 위해서는 아래 과정을 거치시면 됩니다.

```python
>>> from pororo import Pororo
>>> Pororo.available_models("collocation")
'Available models for collocation are ([lang]: ko, [model]: kollocate), ([lang]: en, [model]: collocate.en), ([lang]: ja, [model]: collocate.ja), ([lang]: zh, [model]: collocate.zh)'
```

- 특정 태스크를 수행하고자 하실 때에는, `task` 인자에 앞서 살펴본 태스크명과 `lang` 인자에 언어명을 넣어주시면 됩니다.

```python
>>> from pororo import Pororo
>>> ner = Pororo(task="ner", lang="en")
```

- 객체 생성 이후에는, 다음과 같이 입력 값을 넘겨주는 방식으로 사용이 가능합니다.

```python
>>> ner("Michael Jeffrey Jordan (born February 17, 1963) is an American businessman and former professional basketball player.")
[('Michael Jeffrey Jordan', 'PERSON'), ('(', 'O'), ('born', 'O'), ('February 17, 1963)', 'DATE'), ('is', 'O'), ('an', 'O'), ('American', 'NORP'), ('businessman', 'O'), ('and', 'O'), ('former', 'O'), ('professional', 'O'), ('basketball', 'O'), ('player', 'O'), ('.', 'O')]
```

- 여러 언어를 지원하는 태스크라면, `lang` 인자를 바꾸어 서로 다른 언어로 훈련된 모델을 활용할 수 있습니다.

```python
>>> ner = Pororo(task="ner", lang="ko")
>>> ner("마이클 제프리 조던(영어: Michael Jeffrey Jordan, 1963년 2월 17일 ~ )은 미국의 은퇴한 농구 선수이다.")
[('마이클 제프리 조던', 'PERSON'), ('(', 'O'), ('영어', 'CIVILIZATION'), (':', 'O'), (' ', 'O'), ('Michael Jeffrey Jordan', 'PERSON'), (',', 'O'), (' ', 'O'), ('1963년 2월 17일 ~', 'DATE'), (' ', 'O'), (')은', 'O'), (' ', 'O'), ('미국', 'LOCATION'), ('의', 'O'), (' ', 'O'), ('은퇴한', 'O'), (' ', 'O'), ('농구 선수', 'CIVILIZATION'), ('이다.', 'O')]
>>> ner = Pororo(task="ner", lang="ja")
>>> ner("マイケル・ジェフリー・ジョーダンは、アメリカ合衆国の元バスケットボール選手")
[('マイケル・ジェフリー・ジョーダン', 'PERSON'), ('は', 'O'), ('、アメリカ合衆国', 'O'), ('の', 'O'), ('元', 'O'), ('バスケットボール', 'O'), ('選手', 'O')]
>>> ner = Pororo(task="ner", lang="zh")
>>> ner("麥可·傑佛瑞·喬丹是美國退役NBA職業籃球運動員，也是一名商人，現任夏洛特黃蜂董事長及主要股東")
[('麥可·傑佛瑞·喬丹', 'PERSON'), ('是', 'O'), ('美國', 'GPE'), ('退', 'O'), ('役', 'O'), ('nba', 'ORG'), ('職', 'O'), ('業', 'O'), ('籃', 'O'), ('球', 'O'), ('運', 'O'), ('動', 'O'), ('員', 'O'), ('，', 'O'), ('也', 'O'), ('是', 'O'), ('一', 'O'), ('名', 'O'), ('商', 'O'), ('人', 'O'), ('，', 'O'), ('現', 'O'), ('任', 'O'), ('夏洛特黃蜂', 'ORG'), ('董', 'O'), ('事', 'O'), ('長', 'O'), ('及', 'O'), ('主', 'O'), ('要', 'O'), ('股', 'O'), ('東', 'O')]
```

- 하나의 태스크가 여러 모델을 지원하는 경우, `model` 인자에 넘겨주는 값을 변경해 다른 모델을 사용하실 수 있습니다.

```python
>>> from pororo import Pororo
>>> mt = Pororo(task="mt", lang="multi", model="transformer.large.multi.mtpg")
>>> fast_mt = Pororo(task="mt", lang="multi", model="transformer.large.multi.fast.mtpg")
```

<br>

## Documentation

보다 자세한 정보는 [full documentation](https://kakaobrain.github.io/pororo/)을 참조하세요.

궁금한 사항이나 의견 등이 있으시다면 [이슈](https://github.com/kakaobrain/pororo/issues)를 남겨주세요.

<br>

## Citation

PORORO 라이브러리를 프로젝트 혹은 연구에 활용하신다면 아래 정보를 통해 인용을 해주시기 바랍니다:

```
@misc{pororo,
  author       = {Heo, Hoon and Ko, Hyunwoong and Kim, Soohwan and
                  Han, Gunsoo and Park, Jiwoo and Park, Kyubyong},
  title        = {PORORO: Platform Of neuRal mOdels for natuRal language prOcessing},
  howpublished = {\url{https://github.com/kakaobrain/pororo}},
  year         = {2021},
}
```

<br>

## Contributors

[허훈](https://github.com/huffon), [고현웅](https://github.com/hyunwoongko), [김수환](https://github.com/sooftware), [한건수](https://github.com/robinsongh381), [박지우](https://github.com/bernardscumm) 그리고 [박규병](https://github.com/Kyubyong)

<br>

## License

`PORORO` 프로젝트는 **Apache License 2.0 라이센스**를 따릅니다.

Copyright 2021 Kakao Brain Corp. <https://www.kakaobrain.com> All Rights Reserved.
