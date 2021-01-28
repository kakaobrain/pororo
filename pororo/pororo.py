"""
Pororo task-specific factory class

    isort:skip_file

"""

import logging
from typing import Optional
from pororo.tasks.utils.base import PororoTaskBase

import torch

from pororo.tasks import (
    PororoAgeSuitabilityFactory,
    PororoAesFactory,
    PororoAsrFactory,
    PororoBlankFactory,
    PororoCaptionFactory,
    PororoCollocationFactory,
    PororoConstFactory,
    PororoDpFactory,
    PororoGecFactory,
    PororoP2gFactory,
    PororoInflectionFactory,
    PororoLemmatizationFactory,
    PororoMrcFactory,
    PororoNerFactory,
    PororoNliFactory,
    PororoOcrFactory,
    PororoParaIdFactory,
    PororoParaphraseFactory,
    PororoG2pFactory,
    PororoPosFactory,
    PororoQuestionGenerationFactory,
    PororoReviewFactory,
    PororoSentenceFactory,
    PororoSentimentFactory,
    PororoSrlFactory,
    PororoStsFactory,
    PororoContextualFactory,
    PororoSummarizationFactory,
    PororoTokenizationFactory,
    PororoTranslationFactory,
    PororoWordFactory,
    PororoWordTranslationFactory,
    PororoZeroShotFactory,
    PororoSpeechTranslationFactory,
)

SUPPORTED_TASKS = {
    "mrc": PororoMrcFactory,
    "rc": PororoMrcFactory,
    "qa": PororoMrcFactory,
    "question_answering": PororoMrcFactory,
    "machine_reading_comprehension": PororoMrcFactory,
    "reading_comprehension": PororoMrcFactory,
    "sentiment": PororoSentimentFactory,
    "sentiment_analysis": PororoSentimentFactory,
    "nli": PororoNliFactory,
    "natural_language_inference": PororoNliFactory,
    "inference": PororoNliFactory,
    "fill": PororoBlankFactory,
    "fill_in_blank": PororoBlankFactory,
    "fib": PororoBlankFactory,
    "para": PororoParaIdFactory,
    "pi": PororoParaIdFactory,
    "cse": PororoContextualFactory,
    "contextual_subword_embedding": PororoContextualFactory,
    "similarity": PororoStsFactory,
    "sts": PororoStsFactory,
    "semantic_textual_similarity": PororoStsFactory,
    "sentence_similarity": PororoStsFactory,
    "sentvec": PororoSentenceFactory,
    "sentence_embedding": PororoSentenceFactory,
    "sentence_vector": PororoSentenceFactory,
    "se": PororoSentenceFactory,
    "inflection": PororoInflectionFactory,
    "morphological_inflection": PororoInflectionFactory,
    "g2p": PororoG2pFactory,
    "grapheme_to_phoneme": PororoG2pFactory,
    "grapheme_to_phoneme_conversion": PororoG2pFactory,
    "w2v": PororoWordFactory,
    "wordvec": PororoWordFactory,
    "word2vec": PororoWordFactory,
    "word_vector": PororoWordFactory,
    "word_embedding": PororoWordFactory,
    "tokenize": PororoTokenizationFactory,
    "tokenise": PororoTokenizationFactory,
    "tokenization": PororoTokenizationFactory,
    "tokenisation": PororoTokenizationFactory,
    "tok": PororoTokenizationFactory,
    "segmentation": PororoTokenizationFactory,
    "seg": PororoTokenizationFactory,
    "mt": PororoTranslationFactory,
    "machine_translation": PororoTranslationFactory,
    "translation": PororoTranslationFactory,
    "pos": PororoPosFactory,
    "tag": PororoPosFactory,
    "pos_tagging": PororoPosFactory,
    "tagging": PororoPosFactory,
    "const": PororoConstFactory,
    "constituency": PororoConstFactory,
    "constituency_parsing": PororoConstFactory,
    "cp": PororoConstFactory,
    "pg": PororoParaphraseFactory,
    "collocation": PororoCollocationFactory,
    "collocate": PororoCollocationFactory,
    "col": PororoCollocationFactory,
    "word_translation": PororoWordTranslationFactory,
    "wt": PororoWordTranslationFactory,
    "summarization": PororoSummarizationFactory,
    "summarisation": PororoSummarizationFactory,
    "text_summarization": PororoSummarizationFactory,
    "text_summarisation": PororoSummarizationFactory,
    "summary": PororoSummarizationFactory,
    "gec": PororoGecFactory,
    "review": PororoReviewFactory,
    "review_scoring": PororoReviewFactory,
    "lemmatization": PororoLemmatizationFactory,
    "lemmatisation": PororoLemmatizationFactory,
    "lemma": PororoLemmatizationFactory,
    "ner": PororoNerFactory,
    "named_entity_recognition": PororoNerFactory,
    "entity_recognition": PororoNerFactory,
    "zero-topic": PororoZeroShotFactory,
    "dp": PororoDpFactory,
    "dep_parse": PororoDpFactory,
    "caption": PororoCaptionFactory,
    "captioning": PororoCaptionFactory,
    "asr": PororoAsrFactory,
    "speech_recognition": PororoAsrFactory,
    "st": PororoSpeechTranslationFactory,
    "speech_translation": PororoSpeechTranslationFactory,
    "ocr": PororoOcrFactory,
    "srl": PororoSrlFactory,
    "semantic_role_labeling": PororoSrlFactory,
    "p2g": PororoP2gFactory,
    "aes": PororoAesFactory,
    "essay": PororoAesFactory,
    "qg": PororoQuestionGenerationFactory,
    "question_generation": PororoQuestionGenerationFactory,
    "age_suitability": PororoAgeSuitabilityFactory,
}

LANG_ALIASES = {
    "english": "en",
    "eng": "en",
    "korean": "ko",
    "kor": "ko",
    "kr": "ko",
    "chinese": "zh",
    "chn": "zh",
    "cn": "zh",
    "japanese": "ja",
    "jap": "ja",
    "jp": "ja",
    "jejueo": "je",
    "jje": "je",
}

logging.getLogger("transformers").setLevel(logging.WARN)
logging.getLogger("fairseq").setLevel(logging.WARN)
logging.getLogger("sentence_transformers").setLevel(logging.WARN)
logging.getLogger("youtube_dl").setLevel(logging.WARN)
logging.getLogger("pydub").setLevel(logging.WARN)
logging.getLogger("librosa").setLevel(logging.WARN)


class Pororo:
    r"""
    This is a generic class that will return one of the task-specific model classes of the library
    when created with the `__new__()` method
    """

    def __new__(
        cls,
        task: str,
        lang: str = "en",
        model: Optional[str] = None,
        **kwargs,
    ) -> PororoTaskBase:
        if task not in SUPPORTED_TASKS:
            raise KeyError("Unknown task {}, available tasks are {}".format(
                task,
                list(SUPPORTED_TASKS.keys()),
            ))

        lang = lang.lower()
        lang = LANG_ALIASES[lang] if lang in LANG_ALIASES else lang

        # Get device information from torch API
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Instantiate task-specific pipeline module, if possible
        task_module = SUPPORTED_TASKS[task](
            task,
            lang,
            model,
            **kwargs,
        ).load(device)

        return task_module

    @staticmethod
    def available_tasks() -> str:
        """
        Returns available tasks in Pororo project

        Returns:
            str: Supported task names

        """
        return "Available tasks are {}".format(list(SUPPORTED_TASKS.keys()))

    @staticmethod
    def available_models(task: str) -> str:
        """
        Returns available model names correponding to the user-input task

        Args:
            task (str): user-input task name

        Returns:
            str: Supported model names corresponding to the user-input task

        Raises:
            KeyError: When user-input task is not supported

        """
        if task not in SUPPORTED_TASKS:
            raise KeyError(
                "Unknown task {} ! Please check available models via `available_tasks()`"
                .format(task))

        langs = SUPPORTED_TASKS[task].get_available_models()
        output = f"Available models for {task} are "
        for lang in langs:
            output += f"([lang]: {lang}, [model]: {', '.join(langs[lang])}), "
        return output[:-2]
