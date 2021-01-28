"""Word Embedding related modeling class"""

from collections import OrderedDict
from typing import Optional

from whoosh.qparser import QueryParser

from pororo.tasks.utils.base import PororoFactoryBase, PororoSimpleBase
from pororo.tasks.utils.download_utils import download_or_load


class PororoWordFactory(PororoFactoryBase):
    """
    Get vector or find similar word and entity from pretrained model using wikipedia

    See also:
        Wikipedia2Vec: An Efficient Toolkit for Learning and Visualizing the Embeddings of Words and Entities from Wikipedia (https://arxiv.org/abs/1812.06280)

    English (`wikipedia2vec.en`)

        - dataset: enwiki-20180420
        - metric: N/A

    Korean (`wikipedia2vec.ko`)

        - dataset: kowiki-20200720
        - metric: N/A

    Japanese (`wikipedia2vec.ja`)

        - dataset: jawiki-20180420
        - metric: N/A

    Chinese (`wikipedia2vec.zh`)

        - dataset: zhwiki-20180420
        - metric: N/A

    Args:
        query (str): input qeury
        top_n (int): number of result word or entity (need for `find_similar_words`)
        group (bool): return grouped dictionary or not (need for `find_similar_words`)

    Notes:
        PororoWikipedia2Vec has two diffrent kinds of output format following below.
        1. 'something' (word) : word2vec result (non-hyperlink in wikipedia documents)
        2. 'something' (other) : entity2vec result (hyperlink in wikipedia documents)

    Examples:
        >>> word2vec = Pororo("word2vec", lang="ko")
        >>> word2vec("사과")  # vector search
        OrderedDict([
            ('사과 (word)',
                tensor([-0.2660, -0.2157, -0.3058, -0.5231, ..., 0.0905, -0.0078,  0.6168,  0.6907], device='cuda:0')),
            ('사과 (pome;fruit of Maloideae;fruit)',
                tensor([ 0.6187, -0.9504, -1.5744,  0.1751, ..., 0.0470,  0.4685,  0.7006, -0.3036], device='cuda:0')),
            ('사향사과 (religious concept)',
                tensor([-0.0748, -0.5694, -1.3145, -1.8251, ..., -0.0657,  0.9534,  0.1697, -0.8623], device='cuda:0')),
            ('사과 (교육) (liberal arts education)',
                tensor([-3.6215e-02, -1.0046e-01, -5.8013e-01, -3.4734e-01, ..., -1.1415e-01,  6.7168e-02,  8.6065e-01, -7.3844e-01], device='cuda:0')),
            ('사과 (영화) (film)',
                tensor([-0.2731, -0.2932, -0.2658, -0.0709, ..., 0.0279,  0.4272, -0.0810, -0.1934], device='cuda:0')),
            ('사과 (행위) (intentional human action)',
                tensor([-0.2321, -0.4228, -0.2982, -0.6823, ..., -0.3684,  0.4122,  0.7825, -0.2925], device='cuda:0'))
        ])
        >>> word2vec.find_similar_words("카카오")  # word or entity search
        OrderedDict([
            ('카카오 (word)', ['몰랑이 (television series)', 'NHN벅스 (business)', '나뚜루 ()', '쿠키런: 오븐브레이크 (video game;mobile game)', '네이버 오디오 클립 ()']),
            ('카카오 (taxon)', ['커피나무 (taxon)', '코코아콩 (seed;intermediate good)', '커피콩 (seed;product)', '카카오 매스 (food ingredient;food;intermediate good)', '콜라나무속 (taxon)']),
            ('카카오 (2006~2014년 기업) (business)', ['줌 (포털 사이트) (website)', '넷츠고 ()', '줌인터넷 ()', 'SK커뮤니케이션즈 (1999~2007년 기업) ()', '드림위즈 (website)']),
            ('카카오 (기업) (enterprise;business)', ['분류:카카오 (Wikimedia category)', '카카오 (2006~2014년 기업) (business)', '줌인터넷 ()', '줌 (포털 사이트) (website)', '네이버 (기업) (enterprise;business)'])
        ])
        >>> word2vec.find_similar_words("카카오", group=True)  # word or entity search using grouping
        OrderedDict([
            ('카카오 (word)',
                OrderedDict([('television series', ['몰랑이']), ('business', ['NHN벅스']), ('', ['나뚜루', '네이버 오디오클립']), ('video game', ['쿠키런: 오븐브레이크']), ('mobile game', ['쿠키런: 오븐브레이크'])])),
            ('카카오 (taxon)',
                OrderedDict([('taxon', ['커피나무', '콜라나무속']), ('seed', ['코코아콩', '커피콩']), ('intermediate good', ['코코아콩', '카카오 매스']), ('product', ['커피콩']), ('food ingredient', ['카카오 매스']), ('food', ['카카오 매스'])])),
            ('카카오 (2006~2014년 기업) (business)',
                OrderedDict([('website', ['줌 (포털 사이트)', '드림위즈']), ('', ['넷츠고', '줌인터넷', 'SK커뮤니케이션즈 (1999~2007년 기업)'])])),
            ('카카오 (기업) (enterprise;business)',
                OrderedDict([('Wikimedia category', ['분류:카카오']), ('business', ['카카오 (2006~2014년 기업)', '네이버 (기업)']), ('', ['줌인터넷']), ('website', ['줌 (포털 사이트)']), ('enterprise', ['네이버 (기업)'])]))
        ])

    """

    def __init__(self, task: str, lang: str, model: Optional[str]):
        super().__init__(task, lang, model)

    @staticmethod
    def get_available_langs():
        return ["en", "ko", "ja", "zh"]

    @staticmethod
    def get_available_models():
        return {
            "en": ["wikipedia2vec.en"],
            "ko": ["wikipedia2vec.ko"],
            "ja": ["wikipedia2vec.ja"],
            "zh": ["wikipedia2vec.zh"],
        }

    def load(self, device: str):
        """
        Load user-selected task-specific model

        Args:
            device (str): device information

        Returns:
            object: User-selected task-specific model

        """
        if "wikipedia2vec" in self.config.n_model:
            import whoosh.index as index

            from pororo.models.wikipedia2vec import Wikipedia2Vec

            vec_map = {
                "ko": "kowiki_20200720_100d.pkl",
                "en": "enwiki_20180420_100d.pkl",
                "ja": "jawiki_20180420_100d.pkl",
                "zh": "zhwiki_20180420_100d.pkl",
            }

            f_wikipedia2vec = download_or_load(
                f"misc/{vec_map[self.config.lang]}",
                self.config.lang,
            )
            wikipedia2vec = Wikipedia2Vec(f_wikipedia2vec, device)

            f_index = download_or_load(
                f"misc/{self.config.lang}_indexdir.zip",
                self.config.lang,
            )
            index_dir = index.open_dir(f_index)
            return PororoWikipedia2Vec(wikipedia2vec, index_dir, self.config)


class PororoWikipedia2Vec(PororoSimpleBase):

    def __init__(self, model, index_dir, config):
        super().__init__(config)
        self._model = model
        self._ix = index_dir

    def _normalize(self, query):
        """
        normalize input query

        Args:
            query (str): input query

        Returns:
            str: normalized input qeury
        """
        searchterm = query.lower()
        searchterm = searchterm.replace(" ", "_")
        return searchterm

    def _get_word_vector(self, word: str):
        """
        get word vector from word string

        Args:
            word (str): word string

        Returns:
            OrderedDict: {word_string: word_vector}

        """
        headword2vec = OrderedDict()
        Word = self._model.get_word(word)

        if Word is not None:
            vec = self._model.get_word_vector(word)
            headword = f"{Word.text} (word)"
            headword2vec[headword] = vec

        return headword2vec

    def _get_entity_vectors(self, entity: str):
        """
        get entity vector from entity string

        Args:
            entity (str): entity string

        Returns:
            OrderedDict: {entity_string: entity_vector}

        """
        headword2vec = OrderedDict()
        with self._ix.searcher() as searcher:
            query = QueryParser("searchterms", self._ix.schema).parse(entity)
            hits = searcher.search(query)

            for hit in hits:
                if "wiki_title" in hit:
                    wiki_title = hit["wiki_title"]
                    category = hit["categories"]
                    headword = f"{wiki_title} ({category})"
                    Entity = self._model.get_entity(wiki_title)
                    if Entity is not None:
                        vec = self._model.get_entity_vector(wiki_title)
                        headword2vec[headword] = vec
        return headword2vec

    @staticmethod
    def _append(headword, relative, headword2relatives):
        """
        append relative to dictionary

        Args:
            headword: head word
            relative: relative word or entity dictionary
            headword2relatives: given result dictionary

        """

        if headword in headword2relatives:
            headword2relatives[headword].append(relative)
        else:
            headword2relatives[headword] = [relative]

    def _postprocess(self, headword2relatives):
        """
        postprocessing for better output format

        Args:
            headword2relatives (OrderedDict):

        Returns:
            OrderedDict: postprocessed output

        """
        new_headword2relatives = OrderedDict()
        for headword, relatives in headword2relatives.items():
            cat2words = OrderedDict()
            for relative in relatives:
                word, category = relative.rsplit(" (", 1)
                category = category[:-1]
                categories = category.split(";")
                for category in categories:
                    self._append(category, word, cat2words)
            new_headword2relatives[headword] = cat2words

        return new_headword2relatives

    def find_similar_words(self, query, top_n=5, group=False):
        """
        find similar words from input query

        Args:
            query (str): input query
            top_n (int): number of result
            group (bool): return grouped dictionary or not

        Returns:
            OrderedDict: word or entity search result

        """

        searchterm = self._normalize(query)

        # Final return
        headword2relatives = OrderedDict()

        with self._ix.searcher() as searcher:
            # Word
            Word = self._model.get_word(searchterm)
            if Word is not None:
                word = Word.text
                headword = f"{word} (word)"
                results = self._model.most_similar(Word, top_n + 1)
                if len(results
                      ) > 1:  # note that the first result is the word itself.
                    for result in results[1:]:  # returned by wikipedia2vec
                        if hasattr(result[0], "text"):  # word
                            relative = result[0].text
                            relative_ = f"{relative} (word)"
                            self._append(
                                headword,
                                relative_,
                                headword2relatives,
                            )
                        else:  # entity
                            relative = result[0].title
                            idx = result[0].index.item()

                            from_idx = QueryParser(
                                "entity_idx",
                                self._ix.schema,
                            ).parse(str(idx))
                            hits = searcher.search(from_idx)
                            if len(hits) > 0:
                                category = hits[0]["categories"]
                                relative_ = f"{relative} ({category})"
                                self._append(
                                    headword,
                                    relative_,
                                    headword2relatives,
                                )
                            else:
                                relative_ = f"{relative} (misc)"
                                self._append(
                                    headword,
                                    relative_,
                                    headword2relatives,
                                )

            # Entity
            from_searchterms = QueryParser(
                "searchterms",
                self._ix.schema,
            ).parse(searchterm)
            hits = searcher.search(from_searchterms)
            for (hit) in (
                    hits
            ):  # returned by indexer <Hit {'categories': 'human', 'display': 'Messi', 'wiki_title': 'Messi (2014 film)'}>
                wiki_title = hit["wiki_title"]
                Entity = self._model.get_entity(wiki_title)
                entity = Entity.title
                category = hit["categories"]
                headword = f"{entity} ({category})"

                results = self._model.most_similar(Entity, top_n + 1)
                if len(results
                      ) > 1:  # note that the first result is the word itself.
                    for result in results[1:]:
                        if hasattr(result[0], "text"):  # word
                            relative = result[0].text
                            relative_ = f"{relative} (word)"
                            self._append(
                                headword,
                                relative_,
                                headword2relatives,
                            )
                        else:  # entity
                            relative = result[0].title
                            idx = result[0].index.item()

                            from_idx = QueryParser(
                                "entity_idx",
                                self._ix.schema,
                            ).parse(str(idx))
                            hits = searcher.search(from_idx)
                            if len(hits) > 0:
                                category = hits[0]["categories"]
                                relative_ = f"{relative} ({category})"
                                self._append(
                                    headword,
                                    relative_,
                                    headword2relatives,
                                )
                            else:
                                relative_ = f"{relative} (misc)"
                                self._append(
                                    headword,
                                    relative_,
                                    headword2relatives,
                                )

        return self._postprocess(
            headword2relatives) if group else headword2relatives

    def predict(
        self,
        query: str,
    ):
        """
        predict to find similar words or entities

        Args:
            query (str): input qeury

        Returns:
            OrderedDict: vector search result

        """

        searchterm = self._normalize(query)
        word2vec = self._get_word_vector(searchterm)
        entity2vec = self._get_entity_vectors(searchterm)
        word2vec.update(entity2vec)

        if not word2vec:
            raise ValueError(f"Oops! {query} does NOT exist in our database.")
        return word2vec
