# Pororo: A Deep Learning based Multilingual Natural Language Processing Library

![](assets/usage.gif)

자연어 처리와 음성 관련 태스크를 쉽게 수행할 수 있도록 도와주는 라이브러리 `pororo` 입니다.

자연어 처리 및 음성 분야 내 다양한 서브 태스크들을 태스크명을 입력하는 것만으로 쉽게 해결할 수 있습니다.

#signals-pororo

<br>

## Installation

- `torch==1.6(cuda 10.1)` 환경에서 정상적으로 동작합니다.

- 아래 커맨드를 통해 패키지를 설치하실 수 있습니다.

```console
pip install pororo
```

- **공통 모듈** 외 특정 태스크를 위한 라이브러리 설치를 위해서는 [INSTALL.md](INSTALL.ko.md)를 참조해주세요.

- **Automatic Speech Recognition**의 활용을 위해서는 [_wav2letter_](https://github.com/facebookresearch/wav2letter)를 별도로 설치해주어야 합니다. 해당 라이브러리 설치를 위해서는 [asr-install.sh](asr-install.sh) 파일을 실행시켜주시면 됩니다.

```console
bash asr-install.sh
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
'Available models for collocation are ([lang]: ko, [model]: kollocate), ([lang]: en, [model]: collocate), ([lang]: ja, [model]: collocate), ([lang]: zh, [model]: collocate)'
```

- 특정 태스크를 수행하고자 하실 때에는, `task` 인자에 앞서 살펴본 태스크명과 `lang` 인자에 언어명을 넣어주시면 됩니다.

```python
>>> from pororo import Pororo
>>> mrc = Pororo(task="mrc", lang="ko")
```

- 객체 생성 이후에는, 다음과 같이 입력 값을 넘겨주는 방식으로 사용이 가능합니다.

```python
>>> mrc(
  "카카오브레인이 공개한 것은?",
  """카카오 인공지능(AI) 연구개발 자회사 카카오브레인이 AI 솔루션을 첫 상품화했다. 카카오는 카카오브레인 '포즈(pose·자세분석) API'를 유료 공개한다고 24일 밝혔다. 카카오브레인이 AI 기술을 유료 API를 공개하는 것은 처음이다. 공개하자마자 외부 문의가 쇄도한다. 포즈는 AI 비전(VISION, 영상·화면분석) 분야 중 하나다. 카카오브레인 포즈 API는 이미지나 영상을 분석해 사람 자세를 추출하는 기능을 제공한다."""
)
('포즈(pose·자세분석) API', (33, 44))
```

<br>

## References

Pororo 라이브러리를 프로젝트 혹은 연구에 활용하신다면 아래 정보를 통해 인용을 해주시기 바랍니다:

```
@misc{pororo,
  author       = {Heo, Hoon and Park, Kyubyong and Ko, Hyunwoong and
                  Kim, Soohwan and Han, Gunsoo and Park, Jiwoo},
  title        = {Pororo: A Deep Learning based Multilingual Natural Language Processing Library},
  howpublished = {\url{https://github.com/kakaobrain/pororo}},
  year         = {2021},
}
```

<br>

## License

Pororo 프로젝트는 Apache License 2.0 라이센스를 따릅니다.

Copyright 2021 Kakao Brain Corp. <https://www.kakaobrain.com> All Rights Reserved.
