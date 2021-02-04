"""Text Summarization related modeling class"""

from typing import List, Optional, Union

import torch
from fairseq import hub_utils
from torch import nn

from pororo.models.brainbert.BrainRoBERTa import BrainRobertaHubInterface
from pororo.tasks import download_or_load
from pororo.tasks.utils.base import (
    PororoFactoryBase,
    PororoGenerationBase,
    PororoSimpleBase,
)


class PororoSummarizationFactory(PororoFactoryBase):
    """
    Text summarization using various pretrained models

    Korean (`kobart.base.ko.summary`)

        - dataset: Dacon summarization corpus + AI Hub summarization corpus (1st release)
        - metric: Rouge-1 (52.03), Rouge-2 (45.18), Rouge-L (49.48)
        - ref: https://dacon.io/competitions/official/235671/data/
        - ref: https://www.aihub.or.kr/node/9176

    Korean (`kobart.base.ko.bullet`)

        - dataset: Internal Corpus
        - metric: Rouge-1 (8.03), Rouge-2 (2.38), Rouge-L (7.23)

    Korean (`brainbert.base.ko.summary`)

        - dataset: Dacon summarization corpus + AI Hub summarization corpus (1st release)
        - metric: Rouge-1 (42.67), Rouge-2 (31.80), Rouge-L (43.12)
        - ref: https://dacon.io/competitions/official/235671/data/
        - ref: https://www.aihub.or.kr/node/9176

    Notes:
        Pororo supports 3 different types of summarization like below.

        1. Abtractive summarization : Model generate a summary in the form of a complete sentence.
        2. Bullet-point summarization : Model generate multiple summaries in the form of a short phrase.
        3. Extractive summarization : Model extract 3 important sentences from article.

    Args:
        text (Union[str, List[str]]): input text to be extracted
        beam (int): beam search size
        temperature (float): temperature scale
        top_k (int): top-K sampling vocabulary size
        top_p (float): top-p sampling ratio
        no_repeat_ngram_size (int): no repeat ngram size
        len_penalty (float): length penalty ratio

    Returns:
        (str) summarized text

    Examples:
        >>> # text summarization task has 3 difference models
        >>> summ = Pororo(task="summarization", model="abstractive", lang="ko")
        >>> summ("20년 4월 8일 자로 아카이브에 올라온 뜨끈뜨끈한 논문을 찾았다. 카카오 브레인에서 한국어 자연어 처리를 위한 새로운 데이터셋을 공개했다는 내용이다. 자연어 추론(NLI)와 텍스트의 의미적 유사성(STS)는 자연어 이해(NLU)에서 핵심 과제. 영어나 다른 언어들은 데이터셋이 몇 개 있는데, 한국어로 된 NLI나 STS 공개 데이터셋이 없다. 이에 동기를 얻어 새로운 한국어 NLI와 STS 데이터 셋을 공개한다. 이전 의 접근 방식에 따라 기존의 영어 훈련 세트를 기계 번역(machine-translate)하고 develop set과 test set을 수동으로 한국어로 번역한다. 한국어 NLU에 대한 연구가 더 활성화되길 바라며, KorNLI와 KorSTS에 baseline을 설정하며, Github에 공개한다. NLI와 STS는 자연어 이해의 중심 과제들로 많이 이야기가 된다. 이에 따라 몇몇 벤치마크 데이터셋은 영어로 된 NLI와 STS를 공개했었다. 그러나 한국어 NLI와 STS 벤치마크  데이터셋은 존재하지 않았다. 대부분의 자연어 처리 연구가 사람들이 많이 쓰는 언어들을 바탕으로 연구  가 되기 때문. 유명한 한국어 NLU 데이터 셋이 전형적으로 QA나 감정 분석은 포함은 되어있는데 NLI나 STS는 아니다. 한국어로 된 공개 NLI나 STS 벤치마크 데이터셋이 없어서 이런 핵심과제에 적합한 한국어 NLU 모델 구축에 대한 관심이 부족했다고 생각한다. 이에 동기를 얻어 KorNLI와 KorSTS를 만들었다.")
        '카카오 브레인에서 자연어 이해의 중심 과제들로 많이 이야기되는 한국어 자연어 처리를 위한 새로운 데이터셋인 KorNLI와 KorSTS 데이터셋을 공개했다.'

        >>> summ = Pororo(task="summarization", model="bullet", lang="ko")
        >>> summ("20년 4월 8일 자로 아카이브에 올라온 뜨끈뜨끈한 논문을 찾았다. 카카오 브레인에서 한국어 자연어 처리를 위한 새로운 데이터셋을 공개했다는 내용이다. 자연어 추론(NLI)와 텍스트의 의미적 유사성(STS)는 자연어 이해(NLU)에서 핵심 과제. 영어나 다른 언어들은 데이터셋이 몇 개 있는데, 한국어로 된 NLI나 STS 공개 데이터셋이 없다. 이에 동기를 얻어 새로운 한국어 NLI와 STS 데이터 셋을 공개한다. 이전 의 접근 방식에 따라 기존의 영어 훈련 세트를 기계 번역(machine-translate)하고 develop set과 test set을 수동으로 한국어로 번역한다. 한국어 NLU에 대한 연구가 더 활성화되길 바라며, KorNLI와 KorSTS에 baseline을 설정하며, Github에 공개한다. NLI와 STS는 자연어 이해의 중심 과제들로 많이 이야기가 된다. 이에 따라 몇몇 벤치마크 데이터셋은 영어로 된 NLI와 STS를 공개했었다. 그러나 한국어 NLI와 STS 벤치마크  데이터셋은 존재하지 않았다. 대부분의 자연어 처리 연구가 사람들이 많이 쓰는 언어들을 바탕으로 연구  가 되기 때문. 유명한 한국어 NLU 데이터 셋이 전형적으로 QA나 감정 분석은 포함은 되어있는데 NLI나 STS는 아니다. 한국어로 된 공개 NLI나 STS 벤치마크 데이터셋이 없어서 이런 핵심과제에 적합한 한국어 NLU 모델 구축에 대한 관심이 부족했다고 생각한다. 이에 동기를 얻어 KorNLI와 KorSTS를 만들었다.")
        ['KorNLI와 KorSTS에 baseline 설정', ' 새로운 NLI와 STS 데이터 셋 공개']

        >>> summ = Pororo(task="summarization", model="extractive", lang="ko")
        >>> summ("20년 4월 8일 자로 아카이브에 올라온 뜨끈뜨끈한 논문을 찾았다. 카카오 브레인에서 한국어 자연어 처리를 위한 새로운 데이터셋을 공개했다는 내용이다. 자연어 추론(NLI)와 텍스트의 의미적 유사성(STS)는 자연어 이해(NLU)에서 핵심 과제. 영어나 다른 언어들은 데이터셋이 몇 개 있는데, 한국어로 된 NLI나 STS 공개 데이터셋이 없다. 이에 동기를 얻어 새로운 한국어 NLI와 STS 데이터 셋을 공개한다. 이전 의 접근 방식에 따라 기존의 영어 훈련 세트를 기계 번역(machine-translate)하고 develop set과 test set을 수동으로 한국어로 번역한다. 한국어 NLU에 대한 연구가 더 활성화되길 바라며, KorNLI와 KorSTS에 baseline을 설정하며, Github에 공개한다. NLI와 STS는 자연어 이해의 중심 과제들로 많이 이야기가 된다. 이에 따라 몇몇 벤치마크 데이터셋은 영어로 된 NLI와 STS를 공개했었다. 그러나 한국어 NLI와 STS 벤치마크  데이터셋은 존재하지 않았다. 대부분의 자연어 처리 연구가 사람들이 많이 쓰는 언어들을 바탕으로 연구  가 되기 때문. 유명한 한국어 NLU 데이터 셋이 전형적으로 QA나 감정 분석은 포함은 되어있는데 NLI나 STS는 아니다. 한국어로 된 공개 NLI나 STS 벤치마크 데이터셋이 없어서 이런 핵심과제에 적합한 한국어 NLU 모델 구축에 대한 관심이 부족했다고 생각한다. 이에 동기를 얻어 KorNLI와 KorSTS를 만들었다.")
        '카카오 브레인에서 한국어 자연어 처리를 위한 새로운 데이터셋을 공개했다는 내용이다. 이에 동기를 얻어 새로운 한국어 NLI와 STS 데이터 셋을 공개한다. 한국어 NLU에 대한 연구가 더 활성화되길 바라며, KorNLI와 KorSTS에 baseline을 설정하며, Github에 공개한다.'

        >>> # text summarization task supports batchwise inference
        >>> summ = Pororo(task="summarization", model="abstractive", lang="ko")
        >>> summ([
        ... "목성과 토성이 약 400년 만에 가장 가까이 만났습니다. 국립과천과학관 등 천문학계에 따르면 21일 저녁 목성과 토성은 1623년 이후 397년 만에 가장 가까워졌는데요. 크리스마스 즈음까지 남서쪽 하늘을 올려다보면 목성과 토성이 가까워지는 현상을 관측할 수 있습니다. 목성의 공전주기는 11.9년, 토성의 공전주기는 29.5년인데요. 공전주기의 차이로 두 행성은 약 19.9년에 한 번 가까워집니다. 이번 근접 때  목성과 토성 사이 거리는 보름달 지름의 5분의 1 정도로 가까워졌습니다. 맨눈으로 보면 두 행성이 겹쳐져 하나의 별처럼 보이는데요. 지난 21일 이후 목성과 토성의 대근접은 2080년 3월 15일로 예측됩니다. 과천과학관 측은 우리가 대근접을 볼 수 있는 기회는 이번이 처음이자 마지막이 될 가능성이 크다라고 설명했 습니다.",
        ... "가수 김태연은 걸 그룹 소녀시대, 소녀시대-태티서 및 소녀시대-Oh!GG의 리더이자 메인보컬이다. 2004년 SM에서 주최한 청소년 베스트 선발 대회에서 노래짱 대상을 수상하며 SM 엔터테인먼트에 캐스팅되었다. 이후 3년간의 연습생을 거쳐 2007년 소녀시대의 멤버로 데뷔했다. 태연은 1989년 3월 9일 대한민국 전라북도 전주시 완산구에서 아버지 김종구, 어머니 김희자 사이의 1남 2녀 중 둘째로 태어났다. 가족으로는 오빠 김지웅, 여동생 김하연이 있다. 어릴 적부터 춤을 좋아했고 특히 명절 때는 친척들이 춤을 시키면 곧잘 추었다던 태연은 TV에서 보아를 보고 가수의 꿈을 갖게 되었다고 한다. 전주양지초등학교를 졸업하였고 전주양지중학교 2학년이던 2003년 SM아카데미 스타라이트 메인지방보컬과 4기에 들어가게 되면서 아버지와 함께 주말마다 전주에서 서울로 이동하며 가수의 꿈을 키웠다. 2004년에 당시 보컬 트레이너였던 더 원의 정규 2집 수록곡 〈You Bring Me Joy (Part 2)〉에 피처링으로 참여했다. 당시 만 15세였던 태연은 현재 활동하는 소속사 SM 엔터테인먼트에 들어가기 전이었다. 이후 태연은 2004년 8월에 열린 제8회 SM 청소년 베스트 선발 대회에서 노래짱 부문에 출전해 1위(대상)를 수상하였고 SM 엔터테인먼트에 정식 캐스팅되어 연습생 생활을 시작하게 되었다. 2005년 청담고등학교에 입학하였으나, 학교 측에서 연예계 활동을 용인하지 않아 전주예술고등학교 방송문화예술과로 전학하였고 2008년 졸업하면서 학교를 빛낸 공로로 공로상을 수상했다. 태연은 연습생 생활이 힘들어 숙소에서 몰래 뛰쳐나갔다가 하루 만에 다시 돌아오기도 했다고 이야기하기도 했다. 이후 SM엔터테인먼트에서 3년여의 연습생 기간을 거쳐 걸 그룹 소녀시대의 멤버로 정식 데뷔하게 되었다."
        ... ])
        ['국립과천과학관 등 천문학계에 따르면 21일 저녁 목성과 토성은 1623년 이후 397년 만에 가장 가까워졌는데 크리스마스 즈음까지 남서쪽 하늘을 올려다보면 목성과 토성이 가까워지는 현상을 관측할 수 있다.',
        '가수 태연은 2004년 SM 청소년 베스트 선발 대회에서 노래짱 대상을 수상하고 SM 엔터테인먼트에 캐스팅되어 3년간의 연습생 기간을 거쳐 2007년 소녀시대의 멤버로 데뷔했다.']

        >>> summ = Pororo(task="summarization", model="bullet", lang="ko")
        >>> summ([
        ... "목성과 토성이 약 400년 만에 가장 가까이 만났습니다. 국립과천과학관 등 천문학계에 따르면 21일 저녁 목성과 토성은 1623년 이후 397년 만에 가장 가까워졌는데요. 크리스마스 즈음까지 남서쪽 하늘을 올려다보면 목성과 토성이 가까워지는 현상을 관측할 수 있습니다. 목성의 공전주기는 11.9년, 토성의 공전주기는 29.5년인데요. 공전주기의 차이로 두 행성은 약 19.9년에 한 번 가까워집니다. 이번 근접 때  목성과 토성 사이 거리는 보름달 지름의 5분의 1 정도로 가까워졌습니다. 맨눈으로 보면 두 행성이 겹쳐져 하나의 별처럼 보이는데요. 지난 21일 이후 목성과 토성의 대근접은 2080년 3월 15일로 예측됩니다. 과천과학관 측은 우리가 대근접을 볼 수 있는 기회는 이번이 처음이자 마지막이 될 가능성이 크다라고 설명했 습니다.",
        ... "가수 김태연은 걸 그룹 소녀시대, 소녀시대-태티서 및 소녀시대-Oh!GG의 리더이자 메인보컬이다. 2004년 SM에서 주최한 청소년 베스트 선발 대회에서 노래짱 대상을 수상하며 SM 엔터테인먼트에 캐스팅되었다. 이후 3년간의 연습생을 거쳐 2007년 소녀시대의 멤버로 데뷔했다. 태연은 1989년 3월 9일 대한민국 전라북도 전주시 완산구에서 아버지 김종구, 어머니 김희자 사이의 1남 2녀 중 둘째로 태어났다. 가족으로는 오빠 김지웅, 여동생 김하연이 있다. 어릴 적부터 춤을 좋아했고 특히 명절 때는 친척들이 춤을 시키면 곧잘 추었다던 태연은 TV에서 보아를 보고 가수의 꿈을 갖게 되었다고 한다. 전주양지초등학교를 졸업하였고 전주양지중학교 2학년이던 2003년 SM아카데미 스타라이트 메인지방보컬과 4기에 들어가게 되면서 아버지와 함께 주말마다 전주에서 서울로 이동하며 가수의 꿈을 키웠다. 2004년에 당시 보컬 트레이너였던 더 원의 정규 2집 수록곡 〈You Bring Me Joy (Part 2)〉에 피처링으로 참여했다. 당시 만 15세였던 태연은 현재 활동하는 소속사 SM 엔터테인먼트에 들어가기 전이었다. 이후 태연은 2004년 8월에 열린 제8회 SM 청소년 베스트 선발 대회에서 노래짱 부문에 출전해 1위(대상)를 수상하였고 SM 엔터테인먼트에 정식 캐스팅되어 연습생 생활을 시작하게 되었다. 2005년 청담고등학교에 입학하였으나, 학교 측에서 연예계 활동을 용인하지 않아 전주예술고등학교 방송문화예술과로 전학하였고 2008년 졸업하면서 학교를 빛낸 공로로 공로상을 수상했다. 태연은 연습생 생활이 힘들어 숙소에서 몰래 뛰쳐나갔다가 하루 만에 다시 돌아오기도 했다고 이야기하기도 했다. 이후 SM엔터테인먼트에서 3년여의 연습생 기간을 거쳐 걸 그룹 소녀시대의 멤버로 정식 데뷔하게 되었다."
        ... ])
        [['21일 저녁 목성과 토성 1623년 이후 397년 만에 가까워져', ' 2080년 3월 15일 대근접 예측', ' 크리스마스 즈음 남서쪽 하늘 올려보면 관측 가능'],
        ['태연, 2004년 청소년 베스트 선발 대회에서 노래짱 대상 수상', ' 태연, SM엔터테인먼트에서 3년여의 연습생 기간 거쳐 걸 그룹 소녀시대의 멤버로 정식 데뷔']]

        >>> summ = Pororo(task="summarization", model="extractive", lang="ko")
        >>> summ([
        ... "목성과 토성이 약 400년 만에 가장 가까이 만났습니다. 국립과천과학관 등 천문학계에 따르면 21일 저녁 목성과 토성은 1623년 이후 397년 만에 가장 가까워졌는데요. 크리스마스 즈음까지 남서쪽 하늘을 올려다보면 목성과 토성이 가까워지는 현상을 관측할 수 있습니다. 목성의 공전주기는 11.9년, 토성의 공전주기는 29.5년인데요. 공전주기의 차이로 두 행성은 약 19.9년에 한 번 가까워집니다. 이번 근접 때  목성과 토성 사이 거리는 보름달 지름의 5분의 1 정도로 가까워졌습니다. 맨눈으로 보면 두 행성이 겹쳐져 하나의 별처럼 보이는데요. 지난 21일 이후 목성과 토성의 대근접은 2080년 3월 15일로 예측됩니다. 과천과학관 측은 우리가 대근접을 볼 수 있는 기회는 이번이 처음이자 마지막이 될 가능성이 크다라고 설명했 습니다.",
        ... "가수 김태연은 걸 그룹 소녀시대, 소녀시대-태티서 및 소녀시대-Oh!GG의 리더이자 메인보컬이다. 2004년 SM에서 주최한 청소년 베스트 선발 대회에서 노래짱 대상을 수상하며 SM 엔터테인먼트에 캐스팅되었다. 이후 3년간의 연습생을 거쳐 2007년 소녀시대의 멤버로 데뷔했다. 태연은 1989년 3월 9일 대한민국 전라북도 전주시 완산구에서 아버지 김종구, 어머니 김희자 사이의 1남 2녀 중 둘째로 태어났다. 가족으로는 오빠 김지웅, 여동생 김하연이 있다. 어릴 적부터 춤을 좋아했고 특히 명절 때는 친척들이 춤을 시키면 곧잘 추었다던 태연은 TV에서 보아를 보고 가수의 꿈을 갖게 되었다고 한다. 전주양지초등학교를 졸업하였고 전주양지중학교 2학년이던 2003년 SM아카데미 스타라이트 메인지방보컬과 4기에 들어가게 되면서 아버지와 함께 주말마다 전주에서 서울로 이동하며 가수의 꿈을 키웠다. 2004년에 당시 보컬 트레이너였던 더 원의 정규 2집 수록곡 〈You Bring Me Joy (Part 2)〉에 피처링으로 참여했다. 당시 만 15세였던 태연은 현재 활동하는 소속사 SM 엔터테인먼트에 들어가기 전이었다. 이후 태연은 2004년 8월에 열린 제8회 SM 청소년 베스트 선발 대회에서 노래짱 부문에 출전해 1위(대상)를 수상하였고 SM 엔터테인먼트에 정식 캐스팅되어 연습생 생활을 시작하게 되었다. 2005년 청담고등학교에 입학하였으나, 학교 측에서 연예계 활동을 용인하지 않아 전주예술고등학교 방송문화예술과로 전학하였고 2008년 졸업하면서 학교를 빛낸 공로로 공로상을 수상했다. 태연은 연습생 생활이 힘들어 숙소에서 몰래 뛰쳐나갔다가 하루 만에 다시 돌아오기도 했다고 이야기하기도 했다. 이후 SM엔터테인먼트에서 3년여의 연습생 기간을 거쳐 걸 그룹 소녀시대의 멤버로 정식 데뷔하게 되었다."
        ... ])
        ['국립과천과학관 등 천문학계에 따르면 21일 저녁 목성과 토성은 1623년 이후 397년 만에 가장 가까워졌는데요. 크리스마스 즈음까지 남서쪽 하늘을 올려다보면 목성과 토성이 가까워지는 현상을 관측할 수 있습니다. 지난 21일 이후 목성과 토성의 대근접은 2080년 3월 15일로 예측됩니다.',
        '2004년 SM에서 주최한 청소년 베스트 선발 대회에서 노래짱 대상을 수상하며 SM 엔터테인먼트에 캐스팅되었다. 이후 태연은 2004년 8월에 열린 제8회 SM 청소년 베스트 선발 대회에서 노래짱 부문에 출전해 1위(대상)를 수상하였고 SM 엔터테인먼트에 정식 캐스팅되어 연습생 생활을 시작하게 되었다. 이후 SM엔터테인먼트에서 3년여의 연습생 기간을 거쳐 걸 그룹 소녀시대의 멤버로 정식 데뷔하게 되었다.']

    """

    def __init__(self, task: str, lang: str, model: Optional[str]):
        super().__init__(task, lang, model)

    @staticmethod
    def get_available_langs():
        return ["ko"]

    @staticmethod
    def get_available_models():
        return {
            "ko": [
                "abstractive",
                "bullet",
                "extractive",
                "kobart.base.ko.summary",
                "kobart.base.ko.bullet",
                "brainbert.base.ko.summary",
            ],
        }

    def load(self, device: str):
        """
        Load user-selected task-specific model

        Args:
            device (str): device information

        Returns:
            object: User-selected task-specific model

        """
        from pororo.tasks.tokenization import PororoTokenizationFactory

        if self.config.n_model == "abstractive":
            self.config.n_model = "kobart.base.ko.summary"

        if self.config.n_model == "bullet":
            self.config.n_model = "kobart.base.ko.bullet"

        if self.config.n_model == "extractive":
            self.config.n_model = "brainbert.base.ko.summary"

        if "kobart" in self.config.n_model:
            from pororo.models.bart.KoBART import KoBartModel
            model_path = download_or_load(
                f"bart/{self.config.n_model}",
                self.config.lang,
            )

            model = KoBartModel.from_pretrained(
                device=device,
                model_path=model_path,
            )

            if "bullet" in self.config.n_model:
                sent_tokenizer = (lambda text: PororoTokenizationFactory(
                    task="tokenization",
                    lang=self.config.lang,
                    model=f"sent_{self.config.lang}",
                ).load(device).predict(text))

                ext_model_name = "brainbert.base.ko.summary"
                ext_summary = PororoRobertaSummary(
                    sent_tokenizer,
                    device,
                    ext_model_name,
                    self.config,
                )

                return PororoKoBartBulletSummary(
                    model=model,
                    config=self.config,
                    ext_summary=ext_summary,
                )

            return PororoKoBartSummary(model=model, config=self.config)

        if "brainbert" in self.config.n_model:
            sent_tokenizer = (lambda text: PororoTokenizationFactory(
                task="tokenization",
                lang=self.config.lang,
                model=f"sent_{self.config.lang}",
            ).load(device).predict(text))

            return PororoRobertaSummary(
                sent_tokenizer,
                device,
                self.config.n_model,
                self.config,
            )


class PororoKoBartSummary(PororoGenerationBase):

    def __init__(self, model, config):
        super(PororoKoBartSummary, self).__init__(config)
        self._model = model

    @torch.no_grad()
    def predict(
        self,
        text: Union[str, List[str]],
        beam: int = 5,
        temperature: float = 1.0,
        top_k: int = -1,
        top_p: float = -1,
        no_repeat_ngram_size: int = 4,
        len_penalty: float = 1.0,
        **kwargs,
    ):
        """
        Conduct abstractive summarization

        Args:
            text (Union[str, List[str]]): input text to be extracted
            beam (int): beam search size
            temperature (float): temperature scale
            top_k (int): top-K sampling vocabulary size
            top_p (float): top-p sampling ratio
            no_repeat_ngram_size (int): no repeat ngram size
            len_penalty (float): length penalty ratio

        Returns:
            (str) summarized text

        """
        sampling = False

        if top_k != -1 or top_p != -1:
            sampling = True

        output = self._model.translate(
            text,
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

        return output

    def __call__(
        self,
        text: Union[str, List[str]],
        beam: int = 5,
        temperature: float = 1.0,
        top_k: int = -1,
        top_p: float = -1,
        no_repeat_ngram_size: int = 4,
        len_penalty: float = 1.0,
        **kwargs,
    ):
        return self.predict(
            text,
            beam,
            temperature,
            top_k,
            top_p,
            no_repeat_ngram_size,
            len_penalty,
        )


class PororoKoBartBulletSummary(PororoGenerationBase):

    def __init__(self, model, config, ext_summary):
        super(PororoKoBartBulletSummary, self).__init__(config)
        self._model = model
        self._ext_summary = ext_summary

    def _postprocess(self, output: Union[str, List[str]]):
        """
        Postprocess output sentence

        Args:
            output (Union[str, List[str]]): output sentence generated by model

        Returns:
            str: postprocessed output sentence

        """

        output = "".join(output).replace("▁", " ")

        for token in ["<s>", "</s>", "<pad>"]:
            output = output.replace(token, "")

        return output.strip().split("<unused0>")

    @torch.no_grad()
    def predict(
        self,
        text: Union[str, List[str]],
        beam: int = 12,
        temperature: float = 1.0,
        top_k: int = -1,
        top_p: float = -1,
        no_repeat_ngram_size: int = 4,
        len_penalty: float = 1.0,
    ):
        """
        Conduct bullet-point summarization

        Args:
            text (Union[str, List[str]]): input text to be extracted
            beam (int): beam search size
            temperature (float): temperature scale
            top_k (int): top-K sampling vocabulary size
            top_p (float): top-p sampling ratio
            no_repeat_ngram_size (int): no repeat ngram size
            len_penalty (float): length penalty ratio

        Returns:
            (str) summarized text

        """
        sampling = False

        if top_k != -1 or top_p != -1:
            sampling = True

        if isinstance(text, str):
            texts = self._ext_summary(text)
        else:
            texts = [self._ext_summary(i) for i in text]

        output = self._model.translate(
            texts,
            beam=beam,
            sampling=sampling,
            temperature=temperature,
            sampling_topk=top_k,
            sampling_topp=top_p,
            max_len_a=1,
            max_len_b=50,
            no_repeat_ngram_size=no_repeat_ngram_size,
            length_penalty=len_penalty,
            return_tokens=True,
            bad_words_ids=[
                [self._model.tokenizer.convert_tokens_to_ids("[")],
                [self._model.tokenizer.convert_tokens_to_ids("]")],
                [self._model.tokenizer.convert_tokens_to_ids("▁[")],
                [self._model.tokenizer.convert_tokens_to_ids("▁]")],
                [self._model.tokenizer.convert_tokens_to_ids("】")],
                [self._model.tokenizer.convert_tokens_to_ids("【")],
            ],
        )

        return self._postprocess(output) if isinstance(
            text,
            str,
        ) else [self._postprocess(o) for o in output]

    def __call__(
        self,
        text: Union[str, List[str]],
        beam: int = 12,
        temperature: float = 1.0,
        top_k: int = -1,
        top_p: float = -1,
        no_repeat_ngram_size: int = 4,
        len_penalty: float = 1.0,
    ):

        return self.predict(
            text,
            beam,
            temperature,
            top_k,
            top_p,
            no_repeat_ngram_size,
            len_penalty,
        )


class PororoRobertaSummary(PororoSimpleBase):

    def __init__(
        self,
        sent_tokenizer,
        device: str,
        ext_model_name: str,
        config,
    ):
        super().__init__(config)
        ckpt_dir = download_or_load(f"bert/{ext_model_name}", config.lang)
        tok_path = download_or_load(
            f"tokenizers/bpe32k.{config.lang}.zip",
            config.lang,
        )

        x = hub_utils.from_pretrained(
            ckpt_dir,
            "model.pt",
            load_checkpoint_heads=True,
        )

        wrapper = BrainRobertaHubInterface(
            x["args"],
            x["task"],
            x["models"][0],
            tok_path,
        )

        clf_dict = torch.load(
            f"{ckpt_dir}/classifier.pt",
            map_location=device,
        )

        classifier_size = 768 if "base" in config.n_model else 1024

        self._device = device
        self._classifier = nn.Linear(classifier_size, 1).to(device).eval()
        self._classifier.load_state_dict(clf_dict)
        self._model = wrapper.model.encoder.sentence_encoder.to(device).eval()

        if "cuda" in device.type:
            self._model = self._model.half()
            self._classifier = self._classifier.half()

        self._tokenizer = BertSumTokenizer(
            bpe=wrapper.bpe,
            dictionary=wrapper.task.source_dictionary,
            sent_tokenizer=sent_tokenizer,
        )

    @torch.no_grad()
    def predict(self, text: str, return_list: bool = False):
        """
        Conduct extractive summarization

        Args:
            text (str): input text
            return_list (bool): whether to return as list

        Returns:
            (str) summarized text
            (List[str]) list of text if return_list is True

        """
        encoded = self._tokenizer.encode_batch(text, max_length=512)
        input_ids = encoded["input_ids"].to(self._device)
        segment_ids = encoded["segment_ids"].to(self._device)
        sentences = encoded["sentences"][0]  # list of str
        cls_ids, mask_cls = self._make_class_ids(input_ids)

        output = self._model(
            input_ids,
            segment_labels=segment_ids,
            last_state_only=True,
        )

        bert_representation = output[0][0].transpose(0, 1)
        batch_arange = (torch.arange(
            bert_representation.size(0)).unsqueeze(1).to(self._device))
        sentence_vector = bert_representation[batch_arange, cls_ids]
        sentence_vector *= mask_cls[:, :, None].float().to(self._device)

        final_logits = self._classifier(sentence_vector).squeeze()
        final_logits = torch.sigmoid(final_logits)
        final_logits = final_logits.clone() * mask_cls.float()

        prediction = final_logits.argsort(descending=True)
        prediction = sorted(prediction[:, :3][0].tolist())
        prediction = [sentences[i] for i in prediction]

        if not return_list:
            prediction = " ".join(prediction)

        return prediction

    def _make_class_ids(self, input_ids: torch.Tensor):
        """
        make [CLS] token index tensor from source tokens

        Args:
            input_ids (torch.Tensor): input token ids

        Returns:
            [CLS] token ids,
            masking of [CLS] token ids for padding
        """
        cls_ids = []

        for batch_tokens in input_ids:
            cls_id = []

            for i, token in enumerate(batch_tokens):
                if token == self._tokenizer._bos_index:
                    cls_id.append(i)
            cls_ids.append(cls_id)

        padded_cls = torch.tensor(cls_ids).long().to(self._device)
        mask_cls = ~(padded_cls == -1).to(self._device)
        return padded_cls, mask_cls

    def __call__(self, text: Union[str, List[str]], return_list: bool = False):
        if isinstance(text, str):
            return self.predict(text, return_list)
        elif isinstance(text, list):
            return [self.predict(t, return_list) for t in text]


class BertSumTokenizer(object):

    def __init__(self, bpe, dictionary, sent_tokenizer):
        self._bpe = bpe
        self._dictionary = dictionary
        self._pad_index = dictionary.pad_index
        self._bos_index = dictionary.bos_index
        self._sent_tokenizer = sent_tokenizer

    def encode_line(self, text):
        """
        encode sentence to token ids and segment ids

        Args:
            text (str or List[str]): input article

        Returns:
            token ids, segment ids, splitted sentences

        References:
            you can find implementation details in here:
            https://arxiv.org/abs/1903.10318

        """
        if isinstance(text, str):
            sentences = self._sent_tokenizer(text)
        else:
            sentences = text

        token_ids, segment_ids = [], []
        for i, sentence in enumerate(sentences):
            bpe_sentence = " ".join(self._bpe.encode(sentence).tokens)
            bpe_sentence = f"<s> {bpe_sentence}"
            # <s> is equivalent with [CLS]

            sent_token_ids = self._dictionary.encode_line(
                bpe_sentence,
                append_eos=False,
                add_if_not_exist=False,
            )

            sent_segment_ids = torch.ones(sent_token_ids.size()).int()
            sent_segment_ids *= i % 2  # to make 0 and 1 alternately.

            token_ids.append(sent_token_ids)
            segment_ids.append(sent_segment_ids)

        token_ids = torch.cat(token_ids, dim=0)
        segment_ids = torch.cat(segment_ids, dim=0)
        return token_ids, segment_ids, sentences

    def encode_batch(self, texts, max_length):
        """
        encode text to token ids batchwise

        Args:
            texts (List[str]): list of input articles
            max_length (int): max token length (for truncation)

        Returns:
            dict: token ids, segment ids, splitted sentences

        """
        result_tokenization = []
        result_segmentation = []
        result_split_sentences = []
        max_len = 0

        if isinstance(texts, str):
            texts = [texts]

        for text in texts:
            token_ids, segment_ids, sentences = self.encode_line(text)
            token_ids = token_ids[:max_length]
            segment_ids = segment_ids[:max_length]

            result_tokenization.append(token_ids)
            result_segmentation.append(segment_ids)
            result_split_sentences.append(sentences)

            if len(token_ids) > max_len:
                max_len = len(token_ids)

        padded_tokens, padded_segments = [], []
        for token_ids, segment_ids in zip(
                result_tokenization,
                result_segmentation,
        ):
            padding_token = (torch.ones(max_len) * self._pad_index).long()
            padding_token[:len(token_ids)] = token_ids
            padded_tokens.append(padding_token.unsqueeze(0))

            padding_segment = (torch.ones(max_len) * self._pad_index).long()
            padding_segment[:len(segment_ids)] = segment_ids
            padded_segments.append(padding_segment.unsqueeze(0))

        padded_tokens = torch.cat(padded_tokens, dim=0).long()
        padded_segments = torch.cat(padded_segments, dim=0).long()

        return {
            "input_ids": padded_tokens,
            "segment_ids": padded_segments,
            "sentences": result_split_sentences,
        }
