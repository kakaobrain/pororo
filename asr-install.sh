if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    ABSOLUTE_PATH=$HOME
    echo "OSTYPE: "$OSTYPE
elif [[ "$OSTYPE" == "darwin"* ]]; then
    ABSOLUTE_PATH='~'
    echo "OSTYPE: "$OSTYPE
elif [[ "$OSTYPE" == "cygwin" ]]; then
    ABSOLUTE_PATH='~'
    echo "OSTYPE: "$OSTYPE
elif [[ "$OSTYPE" == "msys" ]]; then
    ABSOLUTE_PATH='~'
    echo "OSTYPE: "$OSTYPE
elif [[ "$OSTYPE" == "win32" ]]; then
    ABSOLUTE_PATH='C:\'
    echo "OSTYPE: "$OSTYPE
elif [[ "$OSTYPE" == "freebsd"* ]]; then
    ABSOLUTE_PATH='~'
    echo "OSTYPE: "$OSTYPE
else
    echo 'Unknown Operating System'
fi

# Install python libraries
pip install soundfile torchaudio==0.6.0
conda install -y -c conda-forge librosa
pip install pydub editdistance==0.5.3 SoundFile==0.10.2 numba==0.48

# Update apt-get & Install soundfile
apt-get update -y \
&& apt-get upgrade -y \
&& apt-get install -y \
&& apt-get -y install apt-utils gcc libpq-dev libsndfile-dev

# additional prerequisites - use equivalents for your distro
apt-get install -y build-essential cmake libatlas-base-dev libfftw3-dev liblzma-dev libbz2-dev libzstd-dev

cd $ABSOLUTE_PATH

apt install -y build-essential cmake libboost-system-dev libboost-thread-dev libboost-program-options-dev libboost-test-dev libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev
git clone https://github.com/kpu/kenlm.git
cd kenlm
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON
make -j 16
export KENLM_ROOT_DIR=$ABSOLUTE_PATH'/kenlm/'
cd ../..

# Install Additional Dependencies (ATLAS, OpenBLAS, Accelerate, Intel MKL)
apt-get install -y libsndfile1-dev libopenblas-dev libfftw3-dev libgflags-dev libgoogle-glog-dev

# Install wav2letter
git clone -b v0.2 https://github.com/facebookresearch/wav2letter.git
cd wav2letter/bindings/python
pip install -e .
cd ../../..
