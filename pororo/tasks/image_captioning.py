"""Image Captioning related modeling class"""

import os
from typing import Optional

import torch

from pororo.tasks.utils.base import PororoFactoryBase, PororoSimpleBase
from pororo.tasks.utils.download_utils import download_or_load


class PororoCaptionFactory(PororoFactoryBase):
    """
    Generates textual description of an image

    English (`transformer.base.en.caption`)

        - dataset: MS-COCO 2017 (Tsung-Yi Lin et al. 2014)
        - metric: TBU

    Examples:
        >>> caption = Pororo(task="caption", lang="en")
        >>> caption("https://i.pinimg.com/originals/b9/de/80/b9de803706fb2f7365e06e688b7cc470.jpg")
        'Two men sitting at a table with plates of food.'

    """

    def __init__(self, task: str, lang: str, model: Optional[str]):
        super().__init__(task, lang, model)

    @staticmethod
    def get_available_langs():
        return ["en", "ko", "zh", "ja"]

    @staticmethod
    def get_available_models():
        return {
            "en": ["transformer.base.en.caption"],
            "ko": ["transformer.base.en.caption"],
            "zh": ["transformer.base.en.caption"],
            "ja": ["transformer.base.en.caption"],
        }

    def load(self, device: str):
        """
        Load user-selected task-specific model

        Args:
            device (str): device information

        Returns:
            object: User-selected task-specific model

        """
        translator = None

        if "transformer" in self.config.n_model:
            from transformers import BertTokenizer

            from pororo.models.caption import Caption, Detr

            load_dict = download_or_load(
                f"transformer/{self.config.n_model}",
                "en",
            )
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

            pad_token_id = tokenizer.pad_token_id
            vocab_size = tokenizer.vocab_size

            transformer = Caption(pad_token_id, vocab_size)
            transformer.load_state_dict(
                torch.load(
                    os.path.join(
                        load_dict.path,
                        f"{self.config.n_model}.pt",
                    ),
                    map_location=device,
                )["model"])
            transformer.eval().to(device)

            detr = Detr(device)

            if self.config.lang != "en":
                assert self.config.lang in [
                    "ko",
                    "ja",
                    "zh",
                ], "Unsupported language code is selected!"
                from pororo.tasks import PororoTranslationFactory

                translator = PororoTranslationFactory(
                    task="mt",
                    lang="multi",
                    model="transformer.large.multi.mtpg",
                )
                translator = translator.load(device)

            return PororoCaptionBrainCaption(
                detr,
                transformer,
                tokenizer,
                translator,
                device,
                self.config,
            )


class PororoCaptionBrainCaption(PororoSimpleBase):

    def __init__(
        self,
        extractor,
        generator,
        tokenizer,
        translator,
        device,
        config,
    ):
        super().__init__(config)
        self._extractor = extractor
        self._generator = generator
        self._tokenizer = tokenizer
        self._translator = translator

        self._start_token = tokenizer.convert_tokens_to_ids(
            tokenizer._cls_token)
        self._end_token = tokenizer.convert_tokens_to_ids(tokenizer._sep_token)
        self._device = device

        self._max_len = 128

    def _create_caption_and_mask(self):
        """
        Create dummy caption and mask templates

        Returns:
            torch.tensor : template tensors

        """
        caption_template = torch.zeros((1, self._max_len), dtype=torch.long)
        mask_template = torch.ones((1, self._max_len), dtype=torch.bool)

        caption_template[:, 0] = self._start_token
        mask_template[:, 0] = False

        return caption_template.to(self._device), mask_template.to(self._device)

    # TODO : Add beam search logic
    def _generate(self, features, boxes, caption, caption_mask):
        """
        Generate caption using decoding steps

        Args:
            features (torch.tensor): image feature tensor
            boxes (torch.tensor): bounding box features
            caption (torch.tensor): dummy caption template
            caption_mask (torch.tensor): mask template

        Returns:
            torch.tensor : generate token tensor

        """
        for i in range(self._max_len - 1):
            pred = self._generator(
                features,
                boxes,
                caption,
                caption_mask,
            )
            pred = pred[:, i, :]
            pred_id = torch.argmax(pred, axis=-1)

            if pred_id[0] == self._end_token:
                return caption

            caption[:, i + 1] = pred_id[0]
            caption_mask[:, i + 1] = False

        return caption

    def predict(self, image: str, **kwargs):
        """
        Predict caption using image features

        Args:
            image (str): image path

        Returns:
            str: generate captiong corresponding to input image

        """
        output = self._extractor.extract_feature(image)

        features = output["features"].unsqueeze(0).to(self._device)
        boxes = output["boxes"].unsqueeze(0).to(self._device)

        caption, caption_mask = self._create_caption_and_mask()

        caption = self._generate(
            features,
            boxes,
            caption,
            caption_mask,
        )
        caption = self._tokenizer.decode(
            caption[0].tolist(),
            skip_special_tokens=True,
        ).capitalize()

        # apply translation if needed
        if self._translator:
            caption = self._translator(caption, src="en", tgt=self.config.lang)
        return caption
