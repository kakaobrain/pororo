# flake8: noqa
"""
__init__.py for import child .py files

    isort:skip_file
"""

# Utility classes & functions
import pororo.tasks.utils
from pororo.tasks.utils.download_utils import download_or_load
from pororo.tasks.utils.base import (
    PororoBiencoderBase,
    PororoFactoryBase,
    PororoGenerationBase,
    PororoSimpleBase,
    PororoTaskGenerationBase,
)

# Factory classes
from pororo.tasks.age_suitability import PororoAgeSuitabilityFactory
from pororo.tasks.automated_essay_scoring import PororoAesFactory
from pororo.tasks.automatic_speech_recognition import PororoAsrFactory
from pororo.tasks.collocation import PororoCollocationFactory
from pororo.tasks.constituency_parsing import PororoConstFactory
from pororo.tasks.dependency_parsing import PororoDpFactory
from pororo.tasks.fill_in_the_blank import PororoBlankFactory
from pororo.tasks.grammatical_error_correction import PororoGecFactory
from pororo.tasks.grapheme_conversion import PororoP2gFactory
from pororo.tasks.image_captioning import PororoCaptionFactory
from pororo.tasks.morph_inflection import PororoInflectionFactory
from pororo.tasks.lemmatization import PororoLemmatizationFactory
from pororo.tasks.named_entity_recognition import PororoNerFactory
from pororo.tasks.natural_language_inference import PororoNliFactory
from pororo.tasks.optical_character_recognition import PororoOcrFactory
from pororo.tasks.paraphrase_generation import PororoParaphraseFactory
from pororo.tasks.paraphrase_identification import PororoParaIdFactory
from pororo.tasks.phoneme_conversion import PororoG2pFactory
from pororo.tasks.pos_tagging import PororoPosFactory
from pororo.tasks.question_generation import PororoQuestionGenerationFactory
from pororo.tasks.machine_reading_comprehension import PororoMrcFactory
from pororo.tasks.semantic_role_labeling import PororoSrlFactory
from pororo.tasks.semantic_textual_similarity import PororoStsFactory
from pororo.tasks.sentence_embedding import PororoSentenceFactory
from pororo.tasks.sentiment_analysis import PororoSentimentFactory
from pororo.tasks.contextualized_embedding import PororoContextualFactory
from pororo.tasks.text_summarization import PororoSummarizationFactory
from pororo.tasks.tokenization import PororoTokenizationFactory
from pororo.tasks.machine_translation import PororoTranslationFactory
from pororo.tasks.word_embedding import PororoWordFactory
from pororo.tasks.word_translation import PororoWordTranslationFactory
from pororo.tasks.zero_shot_classification import PororoZeroShotFactory
from pororo.tasks.review_scoring import PororoReviewFactory
from pororo.tasks.speech_translation import PororoSpeechTranslationFactory
