# Copyright (c) SKT and its affiliates and Kakao Brain.

from typing import Dict, List

import torch
from transformers import (
    BartConfig,
    BartForConditionalGeneration,
    PreTrainedTokenizerFast,
)


class KoBartModel(object):
    """KoBart Model from SKT"""

    def __init__(self, model: str, device: str):
        config = BartConfig.from_pretrained("hyunwoongko/kobart")
        self.model = BartForConditionalGeneration(config).eval().to(device)

        if "cuda" in device.type:
            self.model = self.model.half()

        self.model.model.load_state_dict(torch.load(
            model,
            map_location=device,
        ))
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(
            "hyunwoongko/kobart")
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        device: str,
        model_path: str = "path/to/model.pt",
    ):
        """
        load pretrained model from disk.
        this method is equivalent with constructor.

        Args:
            device (str): device
            model_path (str): full model path

        Returns:
            (KoBartModel): object of KoBartModel

        """
        return cls(model=model_path, device=device)

    def tokenize(
        self,
        texts: List[str],
        max_len: int = 1024,
    ) -> Dict:
        if isinstance(texts, str):
            texts = [texts]

        texts = [f"<s> {text}" for text in texts]
        eos = self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)
        eos_list = [eos for _ in range(len(texts))]

        tokens = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False,
            max_length=max_len - 1,
            # result + <eos>
        )

        return self.add_bos_eos_tokens(tokens, eos_list)

    def add_bos_eos_tokens(self, tokens, eos_list):
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        token_added_ids, token_added_masks = [], []

        for input_id, atn_mask, eos in zip(
                input_ids,
                attention_mask,
                eos_list,
        ):
            maximum_idx = [
                i for i, val in enumerate(input_id)
                if val != self.tokenizer.convert_tokens_to_ids("<pad>")
            ]

            if len(maximum_idx) == 0:
                idx_to_add = 0
            else:
                idx_to_add = max(maximum_idx) + 1

            eos = torch.tensor([eos], requires_grad=False)
            additional_atn_mask = torch.tensor([1], requires_grad=False)

            input_id = torch.cat([
                input_id[:idx_to_add],
                eos,
                input_id[idx_to_add:],
            ]).long()

            atn_mask = torch.cat([
                atn_mask[:idx_to_add],
                additional_atn_mask,
                atn_mask[idx_to_add:],
            ]).long()

            token_added_ids.append(input_id.unsqueeze(0))
            token_added_masks.append(atn_mask.unsqueeze(0))

        tokens["input_ids"] = torch.cat(token_added_ids, dim=0)
        tokens["attention_mask"] = torch.cat(token_added_masks, dim=0)
        return tokens

    @torch.no_grad()
    def translate(
        self,
        text: str,
        beam: int = 5,
        sampling: bool = False,
        temperature: float = 1.0,
        sampling_topk: int = -1,
        sampling_topp: float = -1,
        length_penalty: float = 1.0,
        max_len_a: int = 1,
        max_len_b: int = 50,
        no_repeat_ngram_size: int = 4,
        return_tokens: bool = False,
        bad_words_ids=None,
    ):
        """
        generate sentence from input sentence.

        See Also:
            1. method and argument names follow fairseq.models.transformer.TransformerModel
            >>> from fairseq.models.transformer import TransformerModel

            2. language codes follow farseq language codes
            >>> from transformers.tokenization_mbart import FAIRSEQ_LANGUAGE_CODES

        Args:
            text (str): input string
            beam (int): beam size
            sampling (bool): sampling or not
            temperature (float): temperature value
            sampling_topk (int): topk sampling
            sampling_topp (float): topp sampling probs
            return_tokens (bool): return tokens or not

        Returns:
            (str): generated sentence string (if return_tokens=False)
            (List[str]): list of generated tokens (if return_tokens=True)

        """

        if isinstance(text, str):
            texts = [text]
        else:
            texts = text

        tokenized = self.tokenize(texts)
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        generated = self.model.generate(
            input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            use_cache=True,
            early_stopping=False,
            decoder_start_token_id=self.tokenizer.bos_token_id,
            num_beams=beam,
            do_sample=sampling,
            temperature=temperature,
            top_k=sampling_topk if sampling_topk > 0 else None,
            top_p=sampling_topp if sampling_topk > 0 else None,
            no_repeat_ngram_size=no_repeat_ngram_size,
            bad_words_ids=[[self.tokenizer.convert_tokens_to_ids("<unk>")]]
            if not bad_words_ids else bad_words_ids +
            [[self.tokenizer.convert_tokens_to_ids("<unk>")]],
            length_penalty=length_penalty,
            max_length=max_len_a * len(input_ids[0]) + max_len_b,
        )

        if return_tokens:
            output = [
                self.tokenizer.convert_ids_to_tokens(_)
                for _ in generated.tolist()
            ]

            return (output[0] if isinstance(
                text,
                str,
            ) else output)

        else:
            output = self.tokenizer.batch_decode(
                generated.tolist(),
                skip_special_tokens=True,
            )

            return (output[0].strip() if isinstance(
                text,
                str,
            ) else [o.strip() for o in output])
