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
        >>> word2vec = Pororo("word2vec", lang="en")
        >>> word2vec("apple")  # vector search
        OrderedDict([
            ('apple (word)',
                tensor([-1.8115e-01,  1.1258e+00, -3.3197e-01,  1.6572e-01,  ..., -6.4689e-01,  6.3094e-02, -8.8036e-02, -2.1675e-01], device='cuda:0')),
            ('Apple (fruit;pome;fruit of Maloideae)',
                tensor([-3.2076e-02,  1.5557e+00,  7.0766e-01, -7.8812e-01, ..., -4.7607e-02,  3.4023e-01,  5.3378e-01, -2.7254e-01], device='cuda:0')),
            ('Muggsy Bogues (human)',
                tensor([-1.0721,  0.9283,  1.2894,  0.4695, ..., 0.1366,  0.5774,  0.0939,  0.9778], device='cuda:0')),
            ('Ariane Passenger Payload Experiment (communications satellite)',
                tensor([ 7.5558e-02, -6.4360e-01,  2.9888e-01,  1.8166e-02,  ..., -7.9919e-01,  2.8561e-01, -4.6676e-01,  2.1841e-01], device='cuda:0')),
            ('Apple Inc. (business;enterprise;NASDAQ-100;giants of the web;Dow Jones Industrial Average)',
                tensor([-0.6466,  1.1077, -0.5390,  0.5268, ..., 0.0375,  0.3269,  1.4260, -0.0849], device='cuda:0')),
            ('Apple Records (record label)',
                tensor([-0.2443,  1.3124,  0.4259,  0.8220,  ..., -0.0310,  0.6967, -1.7474,  0.4733], device='cuda:0')),
            ('Apple (album) (studio album)',
                tensor([ 0.9694,  0.7516,  0.9456, -0.2018, ..., -0.0952, -0.3208, -1.1855,  0.1000], device='cuda:0')),
            ('Apple (automobile) (motor car)',
                tensor([ 0.0273, -0.0827,  0.3302,  0.0199, ..., 0.1942,  0.2985, -0.6952, -0.2728], device='cuda:0')),
            ('Apple River (Illinois) (river)',
                tensor([-0.2683,  1.0154,  0.3947, -0.4488,  ..., 0.3037,  0.0535, -0.4189,  1.3587], device='cuda:0')),
            ('The Apple (Star Trek: The Original Series) (Star Trek episode;television series episode)',
                tensor([ 2.9253e-01,  6.0142e-01,  5.8198e-01,  1.5138e-01, ..., -4.2186e-01,  9.4759e-01, -6.0089e-02,  1.0352e+00], device='cuda:0')),
            ('The Apple (1980 film) (film)',
                tensor([ 1.0943,  0.3313,  1.5675, -1.4343,  ..., -0.2276,  0.5506, -1.5071,  1.0106], device='cuda:0'))
        ])
        >>> word2vec.find_similar_words("apple")
            OrderedDict([
                ('apple (word)', ['blackberry (word)', 'silentype (word)', 'Apple Inc. (business;enterprise;NASDAQ-100;giants of the web;Dow Jones Industrial Average)', 'paulared (word)', 'trueimage (word)']),
                ('Apple (fruit;pome;fruit of Maloideae)', ['Pear (taxon)', 'Apricot (fruit)', 'Plum (taxon)', 'Peach (taxon)', 'Cherry (fruit;drupe)']),
                ('Muggsy Bogues (human)', ['Tom Gugliotta (human)', 'Billy Owens (human)', 'David Wingate (basketball) (human)', '1995–96 Cleveland Cavaliers season (basketball team season)', ':1989–90 Denver Nuggets season (misc)']),
                ('Ariane Passenger Payload Experiment (communications satellite)', ['INSAT-3E (communications satellite)', 'INSAT-3B (communications satellite)', 'INSAT-4E (communications satellite)', 'Rohini (satellite) (artificial satellite)', 'Bhaskara (satellite) (Earth observation satellite)']),
                ('Apple Inc. (business;enterprise;NASDAQ-100;giants of the web;Dow Jones Industrial Average)', ['Steve Jobs (human)', 'IPhone (model series;smartphone)', 'apple (word)', 'IPad (model series;tablet computer)', 'IOS 7 (mobile operating system;iOS;version, edition, or translation)']),
                ('Apple Records (record label)', ['Apple Corps (business;enterprise)', 'Come and Get It: The Best of Apple Records (compilation album;Apple Records Box Set)', 'beatles (word)', 'Maybe Tomorrow (The Iveys album) (album)', 'Maybe Tomorrow (The Iveys song) (Maybe Tomorrow;single)']),
                ('Apple (album) (studio album)', ['Shine (Mother Love Bone EP) (extended play)', 'Mother Love Bone (musical group)', 'The Rockfords (album) (album)', 'Temple of the Dog (album) (album)', 'Chloe Dancer/Crown of Thorns (Shine;song;single)']),
                ('Apple (automobile) (motor car)', ['Dayton Electric (automobile manufacturer)', 'Courier Car Co (automobile manufacturer)', 'Binghamton Electric (automobile manufacturer)', 'Century (automobile) (automobile manufacturer)', 'Babcock Electric Carriage Company (business)']),
                ('Apple River (Illinois) (river)', ['Little Menominee River (stream;river)', 'Plum River (river)', 'Nl:Lijst van rivieren in Illinois (misc)', "Fr:Liste des fleuves de l'Illinois (misc)", 'Sinsinawa River (river)']),
                ('The Apple (Star Trek: The Original Series) (Star Trek episode;television series episode)', ["Mudd's Women (television film;Star Trek episode;television series episode)", 'That Which Survives (Star Trek episode;television series episode)', 'Return to Tomorrow (Star Trek episode;television series episode)', 'The Deadly Years (Star Trek episode;television series episode)', 'By Any Other Name (Star Trek episode;television series episode)']),
                ('The Apple (1980 film) (film)', ['EST and The Forum in popular culture (cultural depiction)', "The Devil's Rain (Wikimedia disambiguation page)", 'Jesus Christ Superstar (film) (film)', 'Shock Treatment (film)', 'Xanadu (film) (film)'])
            ])
        >>> word2vec.find_similar_words("apple", top_n=3, group=True)
        OrderedDict([
            ('apple (word)',
                OrderedDict([('word', ['blackberry', 'silentype']), ('business', ['Apple Inc.']), ('enterprise', ['Apple Inc.']), ('NASDAQ-100', ['Apple Inc.']), ('giants of the web', ['Apple Inc.']), ('Dow Jones Industrial Average', ['Apple Inc.'])])),
            ('Apple (fruit;pome;fruit of Maloideae)',
                OrderedDict([('taxon', ['Pear', 'Plum']), ('fruit', ['Apricot'])])),
            ('Muggsy Bogues (human)',
                OrderedDict([('human', ['Tom Gugliotta', 'Billy Owens', 'David Wingate (basketball)'])])),
            ('Ariane Passenger Payload Experiment (communications satellite)',
                OrderedDict([('communications satellite', ['INSAT-3E', 'INSAT-3B', 'INSAT-4E'])])),
            ('Apple Inc. (business;enterprise;NASDAQ-100;giants of the web;Dow Jones Industrial Average)',
                OrderedDict([('human', ['Steve Jobs']), ('model series', ['IPhone']), ('smartphone', ['IPhone']), ('word', ['apple'])])),
            ('Apple Records (record label)',
                OrderedDict([('business', ['Apple Corps']), ('enterprise', ['Apple Corps']), ('compilation album', ['Come and Get It: The Best of Apple Records']), ('Apple Records Box Set', ['Come and Get It: The Best of Apple Records']), ('word', ['beatles'])])),
            ('Apple (album) (studio album)',
                OrderedDict([('extended play', ['Shine (Mother Love Bone EP)']), ('musical group', ['Mother Love Bone']), ('album', ['The Rockfords (album)'])])),
            ('Apple (automobile) (motor car)',
                OrderedDict([('automobile manufacturer', ['Dayton Electric', 'Courier Car Co', 'Binghamton Electric'])])),
            ('Apple River (Illinois) (river)',
                OrderedDict([('stream', ['Little Menominee River']), ('river', ['Little Menominee River', 'Plum River']), ('misc', ['Nl:Lijst van rivieren in Illinois'])])),
            ('The Apple (Star Trek: The Original Series) (Star Trek episode;television series episode)',
                OrderedDict([('television film', ["Mudd's Women"]), ('Star Trek episode', ["Mudd's Women", 'That Which Survives', 'Return to Tomorrow']), ('television series episode', ["Mudd's Women", 'That Which Survives', 'Return to Tomorrow'])])),
            ('The Apple (1980 film) (film)',
                OrderedDict([('cultural depiction', ['EST and The Forum in popular culture']), ('Wikimedia disambiguation page', ["The Devil's Rain"]), ('film', ['Jesus Christ Superstar (film)'])]))
        ])
        >>> word2vec = Pororo("word2vec", lang="ja")
        >>> word2vec("リンゴ")
        OrderedDict([
            ('リンゴ (word)', tensor([ 0.1310, -0.1558,  0.8368,  0.3689,  ..., 0.0253, -0.0910,  0.1332,  0.0920], device='cuda:0')),
            ('リンゴ (fruit;fruit of Maloideae;pome)', tensor([ 0.4617, -0.3032,  1.5106,  0.7717,  ..., -0.2006,  0.2382, -0.1939,  0.2378], device='cuda:0')),
            ('リンゴ (アルバム) (album)', tensor([-0.7952,  0.3122, -0.1794,  0.5237,  ...,  -0.4918, -0.1221, -0.0287,  0.6898], device='cuda:0'))
        ])
        >>> word2vec.find_similar_words("リンゴ")
        OrderedDict([
            ('リンゴ (word)', ['サクランボ (word)', 'イチゴ (word)', 'スターキングデリシャス (word)', 'ジュース (word)', 'アスパラガス (word)']),
            ('リンゴ (fruit;fruit of Maloideae;pome)', ['イチゴ (taxon)', 'モモ (taxon)', 'ブドウ (grape juice;berry)', 'ナシ (taxon)', 'サクランボ (drupe;fruit)']),
            ('リンゴ (アルバム) (album)', ['グッドナイト・ウィーン (album;studio album)', '想い出のフォトグラフ (Ringo;single;song)', '明日への願い (single)', "オール・シングス・マスト・パス (George Harrison's albums in chronological order;triple album;studio album)", 'バック・オフ・ブーガルー (Stop and Smell the Roses;single;song)'])
        ])
        >>> word2vec.find_similar_words("リンゴ", top_n=3, group=True)
        OrderedDict([
            ('リンゴ (word)',
                OrderedDict([('word', ['サクランボ', 'イチゴ', 'スターキングデリシャス'])])),
            ('リンゴ (fruit;fruit of Maloideae;pome)',
                OrderedDict([('taxon', ['イチゴ', 'モモ']), ('grape juice', ['ブドウ']), ('berry', ['ブドウ'])])),
            ('リンゴ (アルバム) (album)',
                OrderedDict([('album', ['グッドナイト・ウィーン']), ('studio album', ['グッドナイト・ウィーン']), ('Ringo', ['想い出のフォトグラフ']), ('single', ['想い出のフォトグラフ', '明日への願い']), ('song', ['想い出のフォトグラフ'])]))
        ])
        >>> word2vec = Pororo("word2vec", lang="zh")
        >>> word2vec("苹果")
        OrderedDict([
            ('苹果 (word)', tensor([-0.1839,  0.5122, -0.1008,  0.0722, ..., 0.3404, -0.2146,  0.3418, -0.3336], device='cuda:0')),
            ('苹果 (fruit;fruit of Maloideae;pome)', tensor([-0.5241,  0.2368, -1.1965, -0.5834,  ..., 0.3141, -0.7297,  0.5291, -0.2308], device='cuda:0')),
            ('苹果 (电影) (film)', tensor([-0.7060,  0.0215,  0.6849,  0.4374, ..., -0.1802,  0.3402, -0.9224, -0.1029], device='cuda:0')),
            ('蘋果公司 (NASDAQ-100;giants of the web;business;enterprise;Dow Jones Industrial Average)', tensor([-0.8581,  0.2706,  0.0931,  0.1566,  ..., -0.3404, -0.6099,  0.3207, -1.0029], device='cuda:0'))
        ])
        >>> word2vec.find_similar_words("苹果")
        OrderedDict([
            ('苹果 (word)', ['苹果公司 (word)', '黑莓 (word)', '苹果皮 (word)', '树莓 (word)', 'ibookstore (word)']),
            ('苹果 (fruit;fruit of Maloideae;pome)', ['杏仁 (apricot;stone;culinary nuts)', '梨 (taxon)', '無花果 (taxon)', '葡萄 (grape juice;berry)', '桃 (taxon)']),
            ('苹果 (电影) (film)', ['盲山 (film)', '我的父親母親 (misc)', '闯关东 (电视剧) (television program)', '摇摆de婚约 (misc)', '北京遇上西雅圖 (misc)']),
            ('蘋果公司 (NASDAQ-100;giants of the web;business;enterprise;Dow Jones Industrial Average)', ['苹果公司 (word)', 'IOS 9 (iOS;operating system;mobile operating system)', '苹果公司 (misc)', 'MacBook Air (Ultrabook;computer model;MacBook;Apple Macintosh)', 'WWDC (misc)'])
        ])
        >>> word2vec.find_similar_words("苹果", top_n=3, group=True)
        OrderedDict([
            ('苹果 (word)',
                OrderedDict([('word', ['苹果公司', '黑莓', '苹果皮'])])),
            ('苹果 (fruit;fruit of Maloideae;pome)',
                OrderedDict([('apricot', ['杏仁']), ('stone', ['杏仁']), ('culinary nuts', ['杏仁']), ('taxon', ['梨', '無花果'])])),
            ('苹果 (电影) (film)',
                OrderedDict([('film', ['盲山']), ('misc', ['我的父親母親']), ('television program', ['闯关东 (电视剧)'])])),
            ('蘋果公司 (NASDAQ-100;giants of the web;business;enterprise;Dow Jones Industrial Average)',
                OrderedDict([('word', ['苹果公司']), ('iOS', ['IOS 9']), ('operating system', ['IOS 9']), ('mobile operating system', ['IOS 9']), ('misc', ['苹果公司'])]))
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
                # note that the first result is the word itself.
                if len(results) > 1:
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

            # returned by indexer <Hit {'categories': 'human', 'display': 'Messi', 'wiki_title': 'Messi (2014 film)'}>
            for hit in hits:
                wiki_title = hit["wiki_title"]
                Entity = self._model.get_entity(wiki_title)
                entity = Entity.title
                category = hit["categories"]
                headword = f"{entity} ({category})"

                results = self._model.most_similar(Entity, top_n + 1)
                # note that the first result is the word itself.
                if len(results) > 1:
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

    def predict(self, query: str, **kwargs):
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
