# Copyright (c) Studio Ousia, its affiliates and Kakao Brain. All Rights Reserved

from itertools import chain

import joblib
import numpy as np
import six
import torch
from marisa_trie import RecordTrie, Trie


class Wikipedia2VecItem(object):
    r"""Python wrapper class for wikipedia2vec item class"""

    def __init__(self, index, count, doc_count):
        self.index = index
        self.count = count
        self.doc_count = doc_count


class Wikipedia2VecWord(Wikipedia2VecItem):
    r"""Python wrapper class for wikipedia2vec word class"""

    def __init__(self, text, index, count, doc_count):
        super().__init__(index, count, doc_count)
        self.text = text

    def __repr__(self):
        return f"<Word {self.text}>"

    def __reduce__(self):
        return self.__class__, (
            self.text,
            self.index,
            self.count,
            self.doc_count,
        )


class Wikipedia2VecEntity(Wikipedia2VecItem):
    r"""Python wrapper class for wikipedia2vec entity class"""

    def __init__(self, title, index, count, doc_count):
        super().__init__(index, count, doc_count)
        self.title = title

    def __repr__(self):
        return f"<Entity {self.title}>"

    def __reduce__(self):
        return self.__class__, (
            self.title,
            self.index,
            self.count,
            self.doc_count,
        )


class Wikipedia2VecDict(object):
    r"""Python wrapper class for wikipedia2vec dictionary class"""

    def __init__(
        self,
        word_dict,
        entity_dict,
        redirect_dict,
        word_stats,
        entity_stats,
        language,
        lowercase,
        build_params,
        min_paragraph_len=0,
        uuid="",
        device="cuda",
    ):
        self._word_dict = word_dict
        self._word_dict = word_dict
        self._entity_dict = entity_dict
        self._redirect_dict = redirect_dict
        self._word_stats = word_stats[:len(self._word_dict)]
        self._entity_stats = entity_stats[:len(self._entity_dict)]
        self.min_paragraph_len = min_paragraph_len
        self.uuid = uuid
        self.language = language
        self.lowercase = lowercase
        self.build_params = build_params
        self._entity_offset = len(self._word_dict)
        self.device = device

    @property
    def entity_offset(self):
        return self._entity_offset

    @property
    def word_size(self):
        return len(self._word_dict)

    @property
    def entity_size(self):
        return len(self._entity_dict)

    def __len__(self):
        return len(self._word_dict) + len(self._entity_dict)

    def __iter__(self):
        return chain(self.words(), self.entities())

    def words(self):
        for (word, index) in six.iteritems(self._word_dict):
            yield Wikipedia2VecWord(word, index, *self._word_stats[index])

    def entities(self):
        for (title, index) in six.iteritems(self._entity_dict):
            yield Wikipedia2VecEntity(
                title,
                index + self._entity_offset,
                *self._entity_stats[index],
            )

    def get_word(self, word, default=None):
        index = self.get_word_index(word)

        if index == -1:
            return default
        return Wikipedia2VecWord(word, index, *self._word_stats[index])

    def get_entity(self, title, resolve_redirect=True, default=None):
        index = self.get_entity_index(title, resolve_redirect=resolve_redirect)

        if index == -1:
            return default

        dict_index = index - self._entity_offset
        title = self._entity_dict.restore_key(dict_index)
        return Wikipedia2VecEntity(
            title,
            index,
            *self._entity_stats[dict_index],
        )

    def get_word_index(self, word):
        try:
            return self._word_dict[word]
        except KeyError:
            return -1

    def get_entity_index(self, title, resolve_redirect=True):
        if resolve_redirect:
            try:
                index = self._redirect_dict[title][0][0]
                return index + self._entity_offset
            except KeyError:
                pass
        try:
            index = self._entity_dict[title]
            return index + self._entity_offset
        except KeyError:
            return -1

    def get_item_by_index(self, index):
        if index < self._entity_offset:
            return self.get_word_by_index(index)
        return self.get_entity_by_index(index)

    def get_word_by_index(self, index):
        word = self._word_dict.restore_key(index)
        return Wikipedia2VecWord(
            word,
            index,
            *self._word_stats[index],
        )

    def get_entity_by_index(self, index):
        dict_index = index - self._entity_offset
        title = self._entity_dict.restore_key(dict_index)
        return Wikipedia2VecEntity(
            title,
            index,
            *self._entity_stats[dict_index],
        )

    @staticmethod
    def load(target, device, mmap=True):
        word_dict = Trie()
        entity_dict = Trie()
        redirect_dict = RecordTrie("<I")

        if not isinstance(target, dict):
            if mmap:
                target = joblib.load(target, mmap_mode="r")
            else:
                target = joblib.load(target)

        word_dict.frombytes(target["word_dict"])
        entity_dict.frombytes(target["entity_dict"])
        redirect_dict.frombytes(target["redirect_dict"])

        word_stats = target["word_stats"]
        entity_stats = target["entity_stats"]
        if not isinstance(word_stats, np.ndarray):
            word_stats = np.frombuffer(
                word_stats,
                dtype=np.int32,
            ).reshape(-1, 2)
            word_stats = torch.tensor(
                word_stats,
                device=device,
                requires_grad=False,
            )
            entity_stats = np.frombuffer(
                entity_stats,
                dtype=np.int32,
            ).reshape(-1, 2)
            entity_stats = torch.tensor(
                entity_stats,
                device=device,
                requires_grad=False,
            )

        return Wikipedia2VecDict(
            word_dict,
            entity_dict,
            redirect_dict,
            word_stats,
            entity_stats,
            **target["meta"],
        )


class Wikipedia2Vec(object):

    def __init__(self, model_file, device):
        """
        Torch Wikipedia2Vec Wrapper class for word embedding task
        """

        model_object = joblib.load(model_file)

        if isinstance(model_object["dictionary"], dict):
            self.dictionary = Wikipedia2VecDict.load(
                model_object["dictionary"],
                device,
            )
        else:
            self.dictionary = model_object[
                "dictionary"]  # for backward compatibilit

        self.syn0 = torch.tensor(
            model_object["syn0"],
            device=device,
            requires_grad=False,
        )
        self.syn1 = torch.tensor(
            model_object["syn1"],
            device=device,
            requires_grad=False,
        )
        self.train_params = model_object.get("train_params")
        self.device = device

    def get_vector(self, item: Wikipedia2VecItem):
        return self.syn0[item.index]

    def get_word(self, word, default=None):
        return self.dictionary.get_word(word, default)

    def get_entity(self, title, resolve_redirect=True, default=None):
        return self.dictionary.get_entity(
            title,
            resolve_redirect,
            default,
        )

    def get_word_vector(self, word):
        obj = self.dictionary.get_word(word)

        if obj is None:
            return KeyError()
        return self.syn0[obj.index]

    def get_entity_vector(self, title, resolve_redirect=True):
        obj = self.dictionary.get_entity(
            title,
            resolve_redirect=resolve_redirect,
        )

        if obj is None:
            raise KeyError()
        return self.syn0[obj.index]

    def most_similar(self, item, count=100, min_count=None):
        vec = self.get_vector(item)

        return self.most_similar_by_vector(vec, count, min_count=min_count)

    def most_similar_by_vector(self, vec, count=100, min_count=None):
        if min_count is None:
            min_count = 0

        counts = torch.cat([
            torch.tensor(
                self.dictionary._word_stats[:, 0],
                device=self.device,
                requires_grad=False,
            ),
            torch.tensor(
                self.dictionary._entity_stats[:, 0],
                device=self.device,
                requires_grad=False,
            ),
        ])

        dst = self.syn0 @ vec / torch.norm(self.syn0, dim=1) / torch.norm(vec)
        dst[counts < min_count] = -100
        indexes = torch.argsort(-dst)

        return [(
            self.dictionary.get_item_by_index(ind),
            dst[ind],
        ) for ind in indexes[:count]]
