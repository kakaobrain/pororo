# Pororo: A Deep Learning based Multilingual Natural Language Processing Library

![](assets/usage.gif)

`Pororo` performs Natural Language Processing and Speech-related tasks.

It is easy to solve various subtasks in the natural language processing field by simply entering the task name.

<br>

## Installation

- It operates in the environment of `torch=1.6(cuda 10.1)`

- You can install a package through the command below:

```console
pip install pororo
```

- For library installation for specific tasks other than the **common module**, please refer to [INSTALL.md](INSTALL.md)

- For the utilization of **Automatic Speech Recognition**, [_wav2letter_](https://github.com/facebookresearch/wav2letter) should be installed separately. For the installation, please run the [asr-install.sh](asr-install.sh) file

```console
bash asr-install.sh
```

<br>

## Usage

- `Pororo` can be used as follows:
- First, in order to import `Pororo`, you must execute the following snippet

```python
>>> from pororo import Pororo
```

- After the import, you can check the tasks currently supported by the `Pororo` through the following commands

```python
>>> from pororo import Pororo
>>> Pororo.available_tasks()
"Available tasks are ['mrc', 'rc', 'qa', 'question_answering', 'machine_reading_comprehension', 'reading_comprehension', 'sentiment', 'sentiment_analysis', 'nli', 'natural_language_inference', 'inference', 'fill', 'fill_in_blank', 'fib', 'para', 'pi', 'cse', 'contextual_subword_embedding', 'similarity', 'sts', 'semantic_textual_similarity', 'sentence_similarity', 'sentvec', 'sentence_embedding', 'sentence_vector', 'se', 'inflection', 'morphological_inflection', 'g2p', 'grapheme_to_phoneme', 'grapheme_to_phoneme_conversion', 'w2v', 'wordvec', 'word2vec', 'word_vector', 'word_embedding', 'tokenize', 'tokenise', 'tokenization', 'tokenisation', 'tok', 'segmentation', 'seg', 'mt', 'machine_translation', 'translation', 'pos', 'tag', 'pos_tagging', 'tagging', 'const', 'constituency', 'constituency_parsing', 'cp', 'pg', 'collocation', 'collocate', 'col', 'word_translation', 'wt', 'summarization', 'summarisation', 'text_summarization', 'text_summarisation', 'summary', 'gec', 'review', 'review_scoring', 'lemmatization', 'lemmatisation', 'lemma', 'ner', 'named_entity_recognition', 'entity_recognition', 'zero-topic', 'dp', 'dep_parse', 'caption', 'captioning', 'asr', 'speech_recognition', 'st', 'speech_translation', 'ocr', 'srl', 'semantic_role_labeling', 'p2g', 'aes', 'essay', 'qg', 'question_generation', 'age_suitability']"
```

- To check which models are supported by each task, you can go through the following process

```python
>>> from pororo import Pororo
>>> Pororo.available_models("collocation")
'Available models for collocation are ([lang]: ko, [model]: kollocate), ([lang]: en, [model]: collocate), ([lang]: ja, [model]: collocate), ([lang]: zh, [model]: collocate)'
```

- If you want to perform a specific task, you can put the task name in the `task` argument and the language name in the `lang` argument

```python
>>> from pororo import Pororo
>>> mrc = Pororo(task="mrc", lang="ko")
```

- After object construction, it can be used in a way that passes the input value as follows

```python
>>> mrc(
  "카카오브레인이 공개한 것은?",
  """카카오 인공지능(AI) 연구개발 자회사 카카오브레인이 AI 솔루션을 첫 상품화했다. 카카오는 카카오브레인 '포즈(pose·자세분석) API'를 유료 공개한다고 24일 밝혔다. 카카오브레인이 AI 기술을 유료 API를 공개하는 것은 처음이다. 공개하자마자 외부 문의가 쇄도한다. 포즈는 AI 비전(VISION, 영상·화면분석) 분야 중 하나다. 카카오브레인 포즈 API는 이미지나 영상을 분석해 사람 자세를 추출하는 기능을 제공한다."""
)
('포즈(pose·자세분석) API', (33, 44))
```

<br>

## References

If you apply this library to any project and research, please cite our code:

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

## Contributors

[Hoon Heo](https://github.com/huffon), [Kyubyong Park](https://github.com/Kyubyong), [Hyunwoong Ko](https://github.com/hyunwoongko), [Soohwan Kim](https://github.com/sooftware), [Gunsoo Han](https://github.com/robinsongh381) and [Jiwoo Park](https://github.com/bernardscumm)

<br>

## License

`Pororo` project is licensed under the terms of **the Apache License 2.0**.

Copyright 2021 Kakao Brain Corp. <https://www.kakaobrain.com> All Rights Reserved.
