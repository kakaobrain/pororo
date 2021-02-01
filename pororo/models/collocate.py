# Copyright (c) Kakao Brain. All Rights Reserved

import re
from collections import Counter

import whoosh.index as index
from whoosh.qparser import QueryParser


class Collocate(object):

    def __init__(self, index_path: str):
        ix = index.open_dir(index_path)
        self.searcher = ix.searcher()
        self.query = QueryParser("analysis", ix.schema)
        self.zh_map = {
            "n": "noun",
            "nr": "noun",
            "ns": "noun",
            "nt": "noun",
            "nx": "noun",
            "i": "noun",
            "nz": "noun",
            "an": "noun",
            "j": "noun",
            "r": "noun",
            "m": "noun",
            "f": "noun",
            "t": "noun",
            "s": "noun",
            "l": "noun",
            "vn": "noun",
            "v": "verb",
            "vx": "verb",
            "z": "verb",
            "vd": "adverb",
            "ad": "adverb",
            "d": "adverb",
            "a": "adjective",
            "b": "adjective",
        }

    def adjust(self, analysis):
        tokens = re.split("[ +]", analysis)
        words, poses = [], []
        for token in tokens:
            try:
                word, pos = token.split("/")
            except:
                continue

            if pos[0] == "N":
                pos = "noun"
            elif pos in ("VV", "VXV"):
                pos = "verb"
            elif pos in ("VA", "VXA", "VCN"):
                pos = "adjective"
            elif pos[:2] == "MD":
                pos = "determiner"
            elif pos[:2] == "MA":
                pos = "adverb"
            elif pos in self.zh_map:
                pos = self.zh_map[pos]
            else:
                continue
            words.append(word)
            poses.append(pos)
        return words, poses

    def parse_results(self, query, results, min_cnt):
        entry = dict()

        for result in results:
            sent = result["analysis"]
            words, poses = self.adjust(sent)

            for i, (w, p) in enumerate(zip(words, poses)):
                if w == query:
                    if i != 0:
                        w_col = words[i - 1]
                        p_col = poses[i - 1]
                        if p in entry:
                            if p_col in entry[p]:
                                entry[p][p_col].append(w_col)
                            else:
                                entry[p][p_col] = [w_col]
                        else:
                            entry[p] = dict()
                            entry[p][p_col] = [w_col]

                    if i != len(words) - 1:
                        w_col = words[i + 1]
                        p_col = poses[i + 1]
                        if p in entry:
                            if p_col in entry[p]:
                                entry[p][p_col].append(w_col)
                            else:
                                entry[p][p_col] = [w_col]
                        else:
                            entry[p] = dict()
                            entry[p][p_col] = [w_col]

        _entry = dict()
        for pos, cols in entry.items():
            _entry[pos] = dict()
            for col_pos, collocates in cols.items():
                collocate2cnt = Counter(collocates)
                _collocates = []
                for collocate, cnt in collocate2cnt.most_common(
                        len(collocate2cnt)):
                    if cnt > min_cnt:
                        _collocates.append((collocate, cnt))
                _entry[pos][col_pos] = _collocates

        return _entry

    def __call__(self, word: str, min_cnt: int = 5):
        query = self.query.parse(word)
        results = self.searcher.search(query, limit=None)
        collocates = self.parse_results(word, results, min_cnt)
        return collocates
