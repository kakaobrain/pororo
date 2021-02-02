# Installation and Usage

## Installation

- We highly recommend **conda** environment to prevent dependecy problem

```
# pororo only supports python>=3.6
conda create -n pororo python=3.6
conda activate pororo
```

- To install `pororo` run following command:

```
pip install pororo
```

- Or you can install it **locally**:

```console
git clone https://github.com/kakaobrain/pororo.git
cd pororo
pip install -e .
```

- Since `Pororo` sets **English** as a default language option, you should follow [INSTALL](https://github.kakaocorp.com/kakaobrain/pororo/blob/master/INSTALL.md) guide to install other dependency libraries

<br>

## Usage

- To see what's available in `pororo`, run the following command:

```python
>>> from pororo import Pororo
>>> Pororo.available_tasks()
"Available tasks are ['mrc', 'rc', 'qa', 'question_answering', 'machine_reading_comprehension', 'reading_comprehension', 'sentiment', 'sentiment_analysis', 'nli', 'natural_language_inference', 'inference', 'fill', 'fill_in_blank', 'fib', 'para', 'pi', 'cse', 'contextual_subword_embedding', 'similarity', 'sts', 'semantic_textual_similarity', 'sentence_similarity', 'sentvec', 'sentence_embedding', 'sentence_vector', 'se', 'inflection', 'morphological_inflection', 'g2p', 'grapheme_to_phoneme', 'grapheme_to_phoneme_conversion', 'w2v', 'wordvec', 'word2vec', 'word_vector', 'word_embedding', 'tokenize', 'tokenise', 'tokenization', 'tokenisation', 'tok', 'segmentation', 'seg', 'mt', 'machine_translation', 'translation', 'pos', 'tag', 'pos_tagging', 'tagging', 'const', 'constituency', 'constituency_parsing', 'cp', 'pg', 'collocation', 'collocate', 'col', 'word_translation', 'wt', 'summarization', 'summarisation', 'text_summarization', 'text_summarisation', 'summary', 'gec', 'review', 'review_scoring', 'lemmatization', 'lemmatisation', 'lemma', 'ner', 'named_entity_recognition', 'entity_recognition', 'zero-topic', 'dp', 'dep_parse', 'caption', 'captioning', 'asr', 'speech_recognition', 'st', 'speech_translation', 'ocr', 'srl', 'semantic_role_labeling', 'p2g', 'aes', 'essay', 'qg', 'question_generation', 'age_suitability']"
```

- To see which models are available with specific task, run the following command:

```python
>>> from pororo import Pororo
>>> Pororo.available_models("collocation")
'Available models for collocation are ([lang]: ko, [model]: kollocate), ([lang]: en, [model]: collocate), ([lang]: ja, [model]: collocate), ([lang]: zh, [model]: collocate)'
```

- `pororo` takes the concept of HuggingFace Transformers's [**Pipeline**](https://huggingface.co/transformers/main_classes/pipelines.html)
- Therefore, we use factory class `Pororo` to load task-specific model from our hub

```python 
>>> from pororo import Pororo
>>> mrc = Pororo(task="mrc")  # define task model as PororoBertMrc
```

<br>

- You can check the information of the task-specific module by printing object

```python
>>> summary = Pororo(task="summary")
>>> summary
[TASK]: summary
[LANG]: ko
[MODEL]: transformer.base.ko.summary
```