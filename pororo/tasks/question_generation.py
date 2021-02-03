"""Question generation related modeling class"""

import re
import string
from collections import Counter, OrderedDict
from typing import List, Optional, Union

from whoosh.qparser import QueryParser

from pororo.tasks.utils.base import PororoFactoryBase, PororoGenerationBase
from pororo.tasks.utils.download_utils import download_or_load


class PororoQuestionGenerationFactory(PororoFactoryBase):
    """
    Question generation using BART model

    Korean (`kobart.base.ko.qg`)

        - dataset: KorQuAD 1.0 (Lim et al. 2019) + AI hub Reading Comprehension corpus + AI hub Commonsense corpus
        - metric: Model base evaluation using PororoMrc (`brainbert.base`)
            - EM (82.59), F1 (94.06)
        - ref: https://www.aihub.or.kr/aidata/86
        - ref: https://www.aihub.or.kr/aidata/84

    Args:
        answer (str): answer text
        context (str): source article
        beam (int): beam search size
        temperature (float): temperature scale
        top_k (int): top-K sampling vocabulary size
        top_p (float): top-p sampling ratio
        no_repeat_ngram_size (int): no repeat ngram size
        len_penalty (float): length penalty ratio
        n_wrong (int): number of wrong answer candidate
        return_context (bool): return context together or not

    Returns:
        str : question (if `n_wrong` < 1)
        Tuple[str, List[str]] : question, wrong_answers (if `n_wrong` > 1)

    Examples:
        >>> # question generation has argument named `n_wrong` which creates wrong answers together
        >>> qg = Pororo(task="qg", lang="ko")
        >>> qg(
        ...     "카카오톡",
        ...     "카카오톡은 스마트폰의 데이터 통신 기능을 이용하여, 문자 과금 없이 사람들과 메시지를 주고받을 수 있는 애플리케이션이다. 스마트폰 대중화 이 후 기존 인스턴트 메신저 앱의 번거로운 친구 추가 절차 없이, 스마트폰 주소록의 전 화번호만으로 손쉽게 메시지를 주고받을 수 있는 것이 특징이다.",
        ...     n_wrong=3
        ... )
        (('스마트폰의 데이터 통신 기능을 이용해서 문자 과금 없이 사람들과 메시지를 주고받을 수 있는 앱을 뭐라고 해?',
        ['텔레그램', '챗온', '디스코드'])
        >>> # question generation task supports batchwise inference (1:N, N:1, N:N)
        >>> # answer : context = 1 : N
        >>> qg(
        ...     "카카오톡",
        ...     ["카카오톡은 스마트폰의 데이터 통신 기능을 이용하여, 문자 과금 없이 사람들과 메시지를 주고받을 수 있는 애플리케이션이다. 스마트폰 대중화 이 후 기존 인스턴트 메신저 앱의 번거로운 친구 추가 절차 없이, 스마트폰 주소록의 전화번호만으로 손쉽게 메시지를 주고받을 수 있는 것이 특징이다.",
        ...     "메시징 서비스와 사회관계망서비스(SNS) 등 소셜미디어를 사용하는 사람들이 뉴스를 볼 때 가장 선호하는 플랫폼은 카카오톡인 것으로 나타났다. 30일 한국언론진흥재단이 최근 공개한 '2017 소셜미디어 이용자 조사' 결과를 보면 소셜미디어로 뉴스를 본 경험이 있는 우리나라 국민 1천747명에게 뉴스 이용 플랫폼을 중복으로 선택하게 한 결과 50.4%가 카카오톡을 사용했다고 답했다. 카카오톡 다음으로는 페이스북(42%) 사용률이 높았으며 유튜브(31.8%)가 뒤를 이었다."],
        ...     n_wrong=3
        ... )
        [('스마트폰의 데이터 통신 기능을 이용해서 문자 과금 없이 사람들과 메시지를 주고받을 수 있는 앱을 뭐라고 해?', ['텔레그램', '챗온', '디스코드']),
        ('메시징 서비스와 사회관계망서비스(SNS) 등 소셜미디어를 사용하는 사람들이 뉴스를 볼 때 가장 선호하는 플랫폼은 뭐야?', ['텔레그램', '챗온', '디스코드'])]
        >>> # answer : context = N : 1
        >>> qg(
        ...     ["토트넘", "손흥민", "인스타그램"],
        ...     "‘2020년 더 베스트 국제축구연맹(FIFA) 풋볼 어워드’에서 FIFA 푸스카스를 수상한 잉글랜드 프로축구 1부리그 프리미어리 그(EPL) 소속 토트넘 홋스퍼 FC의 손흥민이 기쁜 마음을 드러냈다. 푸스카스는 1년간 전 세계 축구경기에 서 나온 골 중 가장 멋진 골을 뽑는다. 손흥민은 스위스 취리히 소재 FIFA 본부에서 온라인으로 열린 ‘2020년 더 베스트 FIFA 풋볼 어워드’ 시 상식에서 푸스카스를 받았다. 손흥민의 이번 수상은 한국 선수로는  최초이자 아시아에서는 2016년 모하메 드 파이즈 수브리(말레이시아)에 이어 두 번째다. 상을 받은 손흥민은 이날 오전 인스타그램에 “아주 특별한 밤이었다. 저를 지지해주고 제게 투표해 주어서 감사하다”며 셀 프 카메라가 담긴 사진 한 장을 게시했 다. 공개된 사진 속 손흥민은 환한 미소와 엄지를 들어 보이고 있 다. 여기에 손흥민은 이 순간을 절대 잊 지 않겠다고 덧붙였다.  앞서 손흥민은 지난해 12월8일 번리 FC와 가진 EPL 경기에서 환상적인 골을 터뜨 렸다. 당시 손흥민은 토트넘 진영에서 얀 베르통언(벨기에)의 패 스를 받고 공을 잡고 약 70ｍ를 혼자 내달리며 무려 번리 선수 6명을 따돌린 뒤 페널티 지역에서 오른발  슈팅으로 골망을 흔들었다. 이후 이 골은 EPL ‘12월의 골’을 시작으로 영국 공영방송 BBC의 ‘올해의 골’, 영국 스포츠 매체 디 애슬레틱의 ‘올해의  골’에 이어 EPL 사무국이 선정하는 2019∼20시즌 ‘올해의 골’ 등으로 선정되며 최고의 골로 인정받은 바 있다.",
        ...     n_wrong=3
        ... )
        [('손흥민은 어느 팀 소속이야?', ['이즐링턴', '웸블리', '첼시']),
        ('누가 ‘2020년 더 베스트 국제축구연맹(FIFA) 풋볼 어워드’에서 FIFA 푸스카스를 수상했어?', ['지동원', '구자철', '이청용']),
        ('손흥민은 셀프 카메라가 담긴 사진 한 장을 어디에 게시했어?', ['트위터', '페이스북', '사운드클라우드'])]
        >>> # answer : context = N : N (answers list and contexts list must be same length. not N : M)
        >>> qg(
        ...     ["카카오톡", "손흥민"],
        ...     ["메시징 서비스와 사회관계망서비스(SNS) 등 소셜미디어를 사용하는 사 람들이 뉴스를 볼 때 가장 선호하는 플랫폼은 카카오톡인 것으로 나타났다. 30일 한국언론진흥재단이 최근 공개한 '2017 소셜미디어 이용자 조사' 결과를 보면 소셜미디어로 뉴스를 본 경험이 있는 우리나라 국민 1천747명에게 뉴스 이용 플랫폼을 중복으로 선택하게 한 결과 50.4%가 카카오톡을 사용했다고 답했다. 카 카오톡 다음으로는 페이스북(42%) 사용률이 높았으며 유튜브(31.8%)가 뒤를 이었다.",
        ...     "‘2020년 더 베스트 국제축구연맹(FIFA) 풋볼 어워드’에서 FIFA 푸스카스를 수상한 잉글랜드 프로축구 1부리그 프리미어리그(EPL) 소속 토트넘 홋스퍼 FC의 손흥민이 기쁜 마음을 드러냈다. 푸스카스는 1년간 전 세계 축구경기에서 나온 골 중 가장 멋진 골을 뽑는다. 손흥민은 스위스 취리히 소 재 FIFA 본부에서 온라인으로 열린 ‘2020년 더 베스트 FIFA 풋볼 어워드’ 시 상식에서 푸스카스를 받았다. 손흥민의 이번 수상은 한국 선수로는 최초이자 아시아에서는 2016년 모하메 드 파이즈 수브리(말레이시아)에 이어 두 번째다. 상을 받은 손흥민은 이날 오전 인스타그램에 “아주 특별한 밤이었다. 저를 지지해주 고 제게 투표해 주어서 감사하다”며 셀프 카메라가 담긴 사진 한 장을 게시했 다. 공개된 사진 속 손흥민 은 환한 미소와 엄지를 들어 보이고 있다. 여기에 손흥민은 이 순간을 절대 잊 지 않겠다고 덧붙였다.  앞서 손흥민은 지난해 12월8일 번리 FC와 가진 EPL 경기에서 환상적인 골을 터뜨 렸다. 당시 손흥민은 토트 넘 진영에서 얀 베르통언(벨기에)의 패스를 받고 공을 잡고 약 70ｍ를 혼자 내달리며 무려 번리 선수 6명 을 따돌린 뒤 페널티 지역에서 오른발 슈팅으로 골망을 흔들었다. 이후 이 골은 EPL ‘12월의 골’을 시작으로 영국 공영방송 BBC의 ‘올해의 골’, 영국 스포츠 매체 디 애슬레틱의 ‘올해의  골’에 이어 EPL 사무국이 선정하는 2019∼20시즌 ‘올해의 골’ 등으로 선정되며 최고의 골로 인정받은 바 있다."],
        ...     n_wrong=3
        ... )
        [('메시징 서비스와 사회관계망서비스(SNS) 등 소셜미디어를 사용하는 사 람들이 뉴스를 볼 때 가장 선호하는 플랫폼은 뭐야?', ['텔레그램', '챗온', '디스코드']),
        ('누가 ‘2020년 더 베스트 국제축구연맹(FIFA) 풋볼 어워드’에서 FIFA 푸스카스를 수상했어?', ['지동원', '구자철', '이청용'])]

    """

    def __init__(self, task: str, lang: str, model: Optional[str]):
        super().__init__(task, lang, model)

    @staticmethod
    def get_available_langs():
        return ["ko"]

    @staticmethod
    def get_available_models():
        return {
            "ko": ["kobart.base.ko.qg"],
        }

    def load(self, device: str):
        """
        Load user-selected task-specific model

        Args:
            device (str): device information

        Returns:
            object: User-selected task-specific model

        """
        if "bart" in self.config.n_model:
            from whoosh import index

            from pororo.models.bart.KoBART import KoBartModel
            from pororo.models.wikipedia2vec import Wikipedia2Vec
            from pororo.tasks import PororoTokenizationFactory

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

            f_index = download_or_load(
                f"misc/{self.config.lang}_indexdir.zip",
                self.config.lang,
            )

            model = Wikipedia2Vec(model_file=f_wikipedia2vec, device=device)
            idx = index.open_dir(f_index)

            sim_words = SimilarWords(model, idx)

            model_path = download_or_load(
                f"bart/{self.config.n_model}",
                self.config.lang,
            )

            model = KoBartModel.from_pretrained(
                device=device,
                model_path=model_path,
            )

            sent_tok = (lambda text: PororoTokenizationFactory(
                task="tokenization",
                lang=self.config.lang,
                model=f"sent_{self.config.lang}",
            ).load(device).predict(text))

            return PororoKoBartQuestionGeneration(
                model,
                sim_words,
                sent_tok,
                self.config,
            )


class PororoKoBartQuestionGeneration(PororoGenerationBase):

    def __init__(self, model, sim_words, sent_tok, config):
        super(PororoKoBartQuestionGeneration, self).__init__(config)
        self._model = model
        self._max_len = 1024
        self._start_token = "<unused0>"
        self._end_token = "<unused1>"
        self._sim_words = sim_words
        self._sent_tok = sent_tok

    def _focus_answer(
        self,
        context: str,
        answer: str,
        truncate: bool = True,
    ) -> str:
        """
        add answer start and end token
        and truncate context text to make inference speed fast

        Args:
            context (str): context string
            answer (str): answer string
            truncate (bool): truncate or not

        Returns:
            context (str): preprocessed context string
        """

        answer_start_idx = context.find(answer)
        answer_end_idx = answer_start_idx + len(answer) + len(self._start_token)
        context = self._insert_token(context, answer_start_idx,
                                     self._start_token)
        context = self._insert_token(context, answer_end_idx, self._end_token)

        if (len(context) < self._max_len) or (not truncate):
            return context

        sentences = self._sent_tok(context)
        answer_sentence_idx = None
        for i in range(len(sentences)):
            if self._start_token in sentences[i]:
                answer_sentence_idx = i

        i, j = answer_sentence_idx, answer_sentence_idx
        truncated_context = [sentences[answer_sentence_idx]]

        while len(" ".join(truncated_context)) < self._max_len:
            prev_context_length = len(" ".join(truncated_context))
            i -= 1
            j += 1

            if i > 0:
                truncated_context = [sentences[i]] + truncated_context
            if j < len(sentences):
                truncated_context = truncated_context + [sentences[j]]
            if len(" ".join(truncated_context)) == prev_context_length:
                break

        return " ".join(truncated_context)

    def _insert_token(self, src: str, index: int, token: str) -> str:
        """
        insert token to context

        Args:
            src (str): source string
            index (int): index that you want
            token (str): token string

        Returns:
            token_added_string (str): token added string

        """

        return src[:index] + token + src[index:]

    def _postprocess(self, text: str):
        text = text.strip()

        if text[-1] != "?":
            text = text + "?"

        return text

    def predict(
        self,
        answer: Union[str, List[str]],
        context: Union[str, List[str]],
        beam: int = 5,
        temperature: float = 1.0,
        top_k: int = -1,
        top_p: float = -1,
        no_repeat_ngram_size: int = 4,
        len_penalty: float = 1.0,
        n_wrong: int = 0,
    ):
        """
        Conduct question generation

        Args:
            answer (str): answer text
            context (str): source article
            beam (int): beam search size
            temperature (float): temperature scale
            top_k (int): top-K sampling vocabulary size
            top_p (float): top-p sampling ratio
            no_repeat_ngram_size (int): no repeat ngram size
            len_penalty (float): length penalty ratio
            n_wrong (int): number of wrong answer candidate

        """

        sampling = False

        if top_k != -1 or top_p != -1:
            sampling = True

        output = self._model.translate(
            text=context,
            beam=beam,
            sampling=sampling,
            temperature=temperature,
            sampling_topk=top_k,
            sampling_topp=top_p,
            max_len_a=1,
            max_len_b=50,
            no_repeat_ngram_size=no_repeat_ngram_size,
            length_penalty=len_penalty,
        )

        if isinstance(output, str):
            output = self._postprocess(output)
        else:
            output = [self._postprocess(o) for o in output]

        if n_wrong > 0:
            if isinstance(context, str) and isinstance(answer, str):
                wrong_answers = self._sim_words._extract_wrongs(answer)
                output = output, wrong_answers[:n_wrong]

            elif isinstance(context, list) and isinstance(answer, str):
                wrong_answers = self._sim_words._extract_wrongs(answer)[:n_wrong]
                output = [(o, wrong_answers) for o in output]

            elif isinstance(context, str) and isinstance(answer, list):
                wrong_answers = [
                    self._sim_words._extract_wrongs(a)[:n_wrong] for a in answer
                ]
                output = [(output, w) for w in wrong_answers]

            else:
                wrong_answers = [
                    self._sim_words._extract_wrongs(a)[:n_wrong] for a in answer
                ]
                output = [(o, w) for o, w in zip(output, wrong_answers)]

        return output

    def __call__(
        self,
        answer: Union[str, List[str]],
        context: Union[str, List[str]],
        beam: int = 5,
        temperature: float = 1.0,
        top_k: int = -1,
        top_p: float = -1,
        no_repeat_ngram_size: int = 4,
        len_penalty: float = 1.0,
        n_wrong: int = 0,
        return_context: bool = False,
    ):
        """
        Conducnt question generation

        Args:
            answer (str): answer text
            context (str): source article
            beam (int): beam search size
            temperature (float): temperature scale
            top_k (int): top-K sampling vocabulary size
            top_p (float): top-p sampling ratio
            no_repeat_ngram_size (int): no repeat ngram size
            len_penalty (float): length penalty ratio
            n_wrong (int): number of wrong answer candidate
            return_context (bool): return context together or not

        """

        if isinstance(answer, str) and isinstance(context, str):
            context = self._focus_answer(context, answer)

        elif isinstance(answer, list) and isinstance(context, str):
            context = [self._focus_answer(context, a) for a in answer]

        elif isinstance(answer, str) and isinstance(context, list):
            context = [self._focus_answer(c, answer) for c in context]

        elif isinstance(answer, list) and isinstance(context, list):
            assert len(answer) == len(
                context), "length of answer list and context list must be same."
            context = [
                self._focus_answer(c, a) for c, a in zip(context, answer)
            ]

        result = self.predict(
            answer,
            context,
            beam,
            temperature,
            top_k,
            top_p,
            no_repeat_ngram_size,
            len_penalty,
            n_wrong,
        )

        if return_context:
            result = result, context

        return result


class SimilarWords(object):

    def __init__(self, model, idx):
        self._wikipedia2vec = model
        self._ix = idx
        self._searcher = self._ix.searcher()

    def _normalize(self, word: str):
        """
        normalize input string

        Args:
            word (str): input string

        Returns:
            normalized string

        """
        return word.lower().replace(" ", "_")

    def _entity(self, entity: str):
        """
        find entity in entity dictionary

        Args:
            entity (str): entity string

        Returns:
            wikipedia2vec entity

        """
        entity = self._normalize(entity)
        entity = self._wikipedia2vec.get_entity(entity)
        return entity

    def _similar(self, entity: str):
        """
        find similar word with given entity

        Args:
            entity (str): answer that user inputted

        Returns:
            dict: category to entity list

        """
        entity_hit = None
        entity_ = self._entity(entity)
        headword2relatives = {}

        if not entity_:
            return headword2relatives

        from_searchterms = QueryParser(
            "searchterms",
            self._ix.schema,
        ).parse(entity)
        hits = self._searcher.search(from_searchterms)

        for hit in hits:
            wiki_title = hit["wiki_title"]
            if wiki_title == entity:
                entity_hit = hit

        if not entity_hit:
            return headword2relatives

        results = self._wikipedia2vec.most_similar(entity_)
        categories = entity_hit["categories"].split(";")
        category2entities = {category: [] for category in categories}

        if not results:
            return category2entities

        for result in results:
            if hasattr(result[0], "text"):
                continue

            if result[0].title == entity_.title or "분류" in result[0].title:
                continue

            idx = result[0].index.item()
            from_idx = QueryParser("entity_idx", self._ix.schema).parse(str(idx))
            hits2 = self._searcher.search(from_idx)

            if hits2:
                categories2 = hits2[0]["categories"].split(";")
                for each in categories2:
                    if each in category2entities:
                        category2entities[each].append(result[0].title)

        return category2entities

    def _extract_wrongs(self, entity: str) -> List[str]:
        """
        extract wrong answers candidates

        Args:
            entity: entity string

        Returns:
            wrong_list (List[str]): wrong answer candidates list

        """

        entity_list = []
        answer = self._normalize_answer(entity)
        sims = self._similar(answer)
        sims = list(sims.items())

        if len(sims) == 0:
            return entity_list

        for key, val in sims:
            if key.lower() != "word":
                entity_list += self._compare_with_answer(val, answer)

        return list(OrderedDict.fromkeys(entity_list))

    def _compare_with_answer(
        self,
        entity_list: List[str],
        answer: str,
    ) -> List[str]:
        """
        add wrong answer candidate to list
        after compare with answer using n-gram (f1 score)

        Args:
            entity_list (List[str]): wrong answers candidates
            answer (str): answer that will be compared wrong answer

        Returns:
            result_list (List[str]): wrong answer candidates list

        """

        result_list = []
        for e in entity_list:
            if "분류" not in e:
                e = re.sub(r"\([^)]*\)", "", e).strip()
                if self._f1_score(e, answer) < 0.5:
                    result_list.append(e)

        return result_list

    def _normalize_answer(self, s):
        """
        normalize answer string

        Args:
            s: sentence

        Returns:
            normalized sentence

        References:
            https://korquad.github.io/

        """

        def remove_(text):
            text = re.sub("'", " ", text)
            text = re.sub('"', " ", text)
            text = re.sub("《", " ", text)
            text = re.sub("》", " ", text)
            text = re.sub("<", " ", text)
            text = re.sub(">", " ", text)
            text = re.sub("〈", " ", text)
            text = re.sub("〉", " ", text)
            text = re.sub("\\(", " ", text)
            text = re.sub("\\)", " ", text)
            text = re.sub("‘", " ", text)
            text = re.sub("’", " ", text)
            return text

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_punc(lower(remove_(s))))

    def _f1_score(self, prediction, ground_truth):
        """
        compute F1 score

        Args:
            prediction: prediction answer
            ground_truth: ground truth answer

        Returns:
            F1 score between prediction and ground truth

        References:
            https://korquad.github.io/

        """

        prediction_tokens = self._normalize_answer(prediction).split()
        ground_truth_tokens = self._normalize_answer(ground_truth).split()

        # F1 by character
        prediction_Char = []
        for tok in prediction_tokens:
            now = [a for a in tok]
            prediction_Char.extend(now)

        ground_truth_Char = []
        for tok in ground_truth_tokens:
            now = [a for a in tok]
            ground_truth_Char.extend(now)

        common = Counter(prediction_Char) & Counter(ground_truth_Char)
        num_same = sum(common.values())
        if num_same == 0:
            return 0

        precision = 1.0 * num_same / len(prediction_Char)
        recall = 1.0 * num_same / len(ground_truth_Char)
        f1 = (2 * precision * recall) / (precision + recall)

        return f1
