FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

WORKDIR /app

COPY . .

RUN apt-get update && \
    apt-get install -y apt-utils \
    wget \
    git \
    gcc \
    build-essential \
    cmake \
    libpq-dev \
    libsndfile-dev \
    libboost-system-dev \
    libboost-thread-dev \
    libboost-program-options-dev \
    libboost-test-dev \
    libeigen3-dev \
    zlib1g-dev \
    libbz2-dev \
    liblzma-dev \
    libsndfile1-dev \
    libopenblas-dev \
    libfftw3-dev \
    libgflags-dev \
    libgoogle-glog-dev \
    libgl1-mesa-glx \
    libomp-dev

# 1. install pororo
RUN pip install \
    --trusted-host kakaobrain-pypi.dev.9rum.cc \
    --extra-index-url http://kakaobrain-pypi.dev.9rum.cc/ \
    pororo==0.2.0

# 2. install brainspeech
RUN pip install soundfile \
    torchaudio==0.6.0 \
    pydub \
    youtube-dl

RUN conda install -y -c conda-forge librosa

# 3. install tts libs
RUN pip install fastdtw \
    epitran \
    phonemizer \
    ko_pron

# 4. install etc modules
RUN pip install librosa \
    kollocate \
    koparadigm \
    g2pk \
    fugashi \
    romkan \
    ipadic \
    g2pM \
    jieba \
    opencv-python \
    scikit-image

WORKDIR /app/external_lib

RUN git clone https://github.com/kpu/kenlm.git
WORKDIR /app/external_lib/kenlm/build
RUN cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON
RUN make -j 16
ENV KENLM_ROOT_DIR="/app/external_lib/kenlm/"

WORKDIR /app/external_lib
RUN git clone -b v0.2 https://github.com/facebookresearch/wav2letter.git
WORKDIR /app/external_lib/wav2letter/bindings/python
RUN pip install -e .

WORKDIR /app

ARG ACTIVE_PROFILE
RUN echo $ACTIVE_PROFILE
ENV ACTIVE_PROFILE=$ACTIVE_PROFILE
