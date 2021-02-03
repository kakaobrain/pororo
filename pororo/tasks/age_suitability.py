"""Movie scirpt analysis modeling class"""

import math
from typing import Optional

import torch
import torch.nn as nn
from transformers import RobertaModel

from pororo.tasks.utils.base import PororoFactoryBase, PororoSimpleBase
from pororo.tasks.utils.download_utils import download_or_load


class PororoAgeSuitabilityFactory(PororoFactoryBase):
    """
    Conduct Age Suitability task

    English (`roberta.base.en.movie`)

        - dataset: Age Suitability Rating (Mahsa Shafaei et al. 2020)
        - metric for rating: F1 (82.0)
        - metric for emotion: Pearson & Spearman

            +--------------+---------+----------+
            | Emotion Type | Pearson | Spearman |
            +==============+=========+==========+
            | Nudiy        |  0.41   |   0.41   |
            +--------------+---------+----------+
            | Violence     |  0.46   |   0.44   |
            +--------------+---------+----------+
            | Profanity    |  0.52   |   0.53   |
            +--------------+---------+----------+
            | Alcohol      |  0.43   |   0.39   |
            +--------------+---------+----------+
            | Frightening  |  0.49   |   0.47   |
            +--------------+---------+----------+
            | AVERAGE      |  0.44   |   0.43   |
            +--------------+---------+----------+

    Args:
        script: (str) input script

    Returns:
        dict: predicted rating and emotion of a given (movie) script

    Examples:
        >>> age_suitability = Pororo(task="age_suitability", lang="en")
        >>> age_suitability("When I was a little girl ...")
        {
            'rating': {
                'class':'R',
                'description':'Restricted. Under 17 requires accompanying parent or adult guardian.'
            },
            'emotion': {
                'nudity':'Low',
                'violence': 'Moderate',
                'profanity': 'Moderate',
                'alcohol': 'Mild',
                'frightening': 'Severe'
            }
        }

    """

    def __init__(self, task: str, lang: str, model: Optional[str]):
        super().__init__(task, lang, model)

    @staticmethod
    def get_available_langs():
        return ["en"]

    @staticmethod
    def get_available_models():
        return {
            "en": ["roberta.base.en.movie"],
        }

    def load(self, device: str):
        """
        Load user-selected task-specific model

        Args:
            device (str): device information

        Returns:
            object: User-selected task-specific model

        """
        if "bert" in self.config.n_model:
            from transformers import RobertaTokenizer
            model_path = download_or_load(
                f"bert/{self.config.n_model}",
                self.config.lang,
            )

            config = torch.load(f"{model_path}/config.pt")
            tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

            model = RobertaEncoder.from_pretrained(
                device=device,
                model_path=model_path,
                tokenizer=tokenizer,
                config=config,
            ).eval().to(device)

            return PororoBertMovie(model, self.config)


class PororoBertMovie(PororoSimpleBase):

    def __init__(self, model, config):
        super().__init__(config)
        self._model = model

        self._rating_fn = {
            0: "G",
            1: "R",
            2: "PG-13",
            3: "PG",
            4: "NC-17",
        }

        self._rating_description = {
            "G":
                "General Audiences. All ages admitted.",
            "PG":
                "Parental Guidance Suggested. Some material may not be suitable for children.",
            "PG-13":
                "Parents Strongly Cautioned. Some material may be inappropriate for children under 13.",
            "R":
                "Restricted. Under 17 requires accompanying parent or adult guardian.",
            "NC-17":
                "NC-17: No Children. No one 17 and under admitted.",
        }

        self._emotion_fn = {
            0: "nudity",
            1: "violence",
            2: "profanity",
            3: "alcohol",
            4: "frightening",
        }

    def _get_rating(self, logit):
        _, max_val = torch.max(logit, dim=-1)
        rating = self._rating_fn[max_val.item()]
        return rating

    def _get_emotion(self, logit, max_score=3.0):

        def _score_to_label(score):

            if score <= 1.0:
                return "Low"
            elif 1.0 < score <= 2.0:
                return "Mild"
            elif 2.0 < score <= 3.0:
                return "Moderate"
            else:
                return "Severe"

        result = {}
        emotion_list = logit.tolist()[0]
        for k, v in self._emotion_fn.items():
            score = round(emotion_list[k] * max_score, 4)
            label = _score_to_label(score)
            result[v] = label

        return result

    def predict(self, script: str, **kwargs) -> dict:
        """
        Conduct (movie) script analysis

        Args:
            script: (str) input script

        Returns:
            dict: predict rating and emotion of a given script

        Notes:
            Description of Movie Rating:
                G: General Audiences. All ages admitted.
                PG: Parental Guidance Suggested. Some material may not be suitable for children.
                PG-13: Parents Strongly Cautioned. Some material may be inappropriate for children under 13.
                R: Restricted. Under 17 requires accompanying parent or adult guardian.
                NC-17: No Children. No one 17 and under admitted.

        """
        _, logit_rating, logit_emotion = self._model(script)
        rating = self._get_rating(logit_rating)
        rating_description = self._rating_description[rating]
        emotion = self._get_emotion(logit_emotion)

        result = {
            "rating": {
                "class": rating,
                "description": rating_description
            },
            "emotion": emotion,
        }

        return result


class RobertaEncoder(nn.Module):
    """RoBERTa encoder."""

    def __init__(
        self,
        device,
        model_path,
        tokenizer,
        config,
        num_genre=24,
        num_rating=5,
        num_emotion=5,
    ):
        super(RobertaEncoder, self).__init__()

        self._device = device
        self._model_path = model_path
        self._tokenizer = tokenizer
        self._config = config
        self._num_genre = num_genre
        self._num_rating = num_rating
        self._num_emotion = num_emotion
        self._max_position_embeddings = config.max_position_embeddings

        self.roberta = RobertaModel(config)
        self.linear_genre = nn.Linear(config.hidden_size, num_genre)
        self.linear_rating = nn.Linear(config.hidden_size, num_rating)
        self.linear_emotion = nn.Linear(config.hidden_size, num_emotion)
        self._load_weight(device)

    def _load_weight(self, device: str):
        checkpoint = torch.load(
            f"{self._model_path}/model.pt",
            map_location=device,
        )
        self.load_state_dict(checkpoint)

    def _check_num_labels(self):
        genre_dict = torch.load(f"{self._model_path}/vocab_genre.pt")
        rating_dict = torch.load(f"{self._model_path}/vocab_rating.pt")
        emotion_dict = torch.load(f"{self._model_path}/vocab_emotion.pt")

        assert len(genre_dict) == self._num_genre
        assert len(rating_dict) == self._num_rating
        assert len(emotion_dict) == self._num_emotion

    def _prepare_input(self, script):
        """
        prepare for inference

        Args:
            script (str): input script

        Returns:
            tokenized indexes

        """
        maxlen = self._max_position_embeddings
        bos_idx = 0  # self.tokenizer.bos_token_id
        pad_idx = 1  # self.tokenizer.pad_token_id
        eos_idx = 2  # self.tokenizer.eos_token_id

        def _filter(text):
            text = text.replace("\n", "")
            return text

        script = _filter(script)
        tokens = self._tokenizer(script)["input_ids"][1:-1]

        tokenized_idx = []

        for n in range(math.ceil(len(tokens) / (maxlen - 2))):
            chunk = ([bos_idx] + tokens[n * (maxlen - 2):(n + 1) *
                                        (maxlen - 2)] + [eos_idx])

            if len(chunk) < maxlen:
                chunk += [pad_idx] * (maxlen - len(chunk))

            tokenized_idx.append(chunk)

        tokenized_idx = torch.tensor(tokenized_idx).unsqueeze(0).to(
            self._device)
        return tokenized_idx

    @staticmethod
    def _get_attention_mask(src_tokens):
        attention_mask = torch.zeros_like(src_tokens)
        for i, v in enumerate(src_tokens):
            for j in range(len(src_tokens[i])):
                if src_tokens[i][j] != 1:
                    attention_mask[i][j] = 1
        return attention_mask.float()

    def forward(self, script: str) -> tuple:
        """
        Args:
            script (str): input script

        Returns:
            tuple of three logits

        """

        src_tokens = self._prepare_input(script)

        embed = []

        for i in range(src_tokens.size(0)):

            src_single = src_tokens[i, :]
            mask = self._get_attention_mask(src_single)
            output = self.roberta(
                src_single,
                mask,
            )
            pooler_output = output.pooler_output
            pooler_average = torch.mean(pooler_output, dim=0)
            embed.append(pooler_average)

        embed = torch.stack(embed)  # (batch, hidden_dim)

        # Logit for each target
        logit_genre = self.linear_genre(embed)
        logit_rating = self.linear_rating(embed)
        logit_emotion = self.linear_emotion(embed)

        return (logit_genre, logit_rating, logit_emotion)

    @classmethod
    def from_pretrained(cls, device, model_path, tokenizer, config):
        return cls(device, model_path, tokenizer, config)
