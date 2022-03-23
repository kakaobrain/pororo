"""Module download related function from. Tenth"""

import logging
import os
import platform
import sys
import zipfile
from dataclasses import dataclass
from typing import Tuple, Union

import wget

from pororo.tasks.utils.config import CONFIGS

DEFAULT_PREFIX = {
    "model": "https://twg.kakaocdn.net/pororo/{lang}/models",
    "dict": "https://twg.kakaocdn.net/pororo/{lang}/dicts",
}


@dataclass
class TransformerInfo:
    r"Dataclass for transformer-based model"
    path: str
    dict_path: str
    src_dict: str
    tgt_dict: str
    src_tok: Union[str, None]
    tgt_tok: Union[str, None]


@dataclass
class DownloadInfo:
    r"Download information such as defined directory, language and model name"
    n_model: str
    lang: str
    root_dir: str


def get_save_dir(save_dir: str = None) -> str:
    """
    Get default save directory

    Args:
        savd_dir(str): User-defined save directory

    Returns:
        str: Set save directory

    """
    # If user wants to manually define save directory
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        return save_dir

    pf = platform.system()

    if pf == "Windows":
        save_dir = "C:\\pororo"
    else:
        home_dir = os.path.expanduser("~")
        save_dir = os.path.join(home_dir, ".pororo")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    return save_dir


def get_download_url(n_model: str, key: str, lang: str) -> str:
    """
    Get download url using default prefix

    Args:
        n_model (str): model name
        key (str): key name either `model` or `dict`
        lang (str): language name

    Returns:
        str: generated download url

    """
    default_prefix = DEFAULT_PREFIX[key].format(lang=lang)
    return f"{default_prefix}/{n_model}"


def download_or_load_bert(info: DownloadInfo) -> str:
    """
    Download fine-tuned BrainBert & BrainSBert model and dict

    Args:
        info (DownloadInfo): download information

    Returns:
        str: downloaded bert & sbert path

    """
    model_path = os.path.join(info.root_dir, info.n_model)

    if not os.path.exists(model_path):
        info.n_model += ".zip"
        zip_path = os.path.join(info.root_dir, info.n_model)

        type_dir = download_from_url(
            info.n_model,
            zip_path,
            key="model",
            lang=info.lang,
        )

        zip_file = zipfile.ZipFile(zip_path)
        zip_file.extractall(type_dir)
        zip_file.close()

    return model_path


def download_or_load_transformer(info: DownloadInfo) -> TransformerInfo:
    """
    Download pre-trained Transformer model and corresponding dict

    Args:
        info (DownloadInfo): download information

    Returns:
        TransformerInfo: information dataclass for transformer construction

    """
    config = CONFIGS[info.n_model.split("/")[-1]]

    src_dict_in = config.src_dict
    tgt_dict_in = config.tgt_dict
    src_tok = config.src_tok
    tgt_tok = config.tgt_tok

    info.n_model += ".pt"
    model_path = os.path.join(info.root_dir, info.n_model)

    # Download or load Transformer model
    model_type_dir = "/".join(model_path.split("/")[:-1])
    if not os.path.exists(model_path):
        model_type_dir = download_from_url(
            info.n_model,
            model_path,
            key="model",
            lang=info.lang,
        )

    dict_type_dir = str()
    src_dict, tgt_dict = str(), str()

    # Download or load corresponding dictionary
    if src_dict_in:
        src_dict = f"{src_dict_in}.txt"
        src_dict_path = os.path.join(info.root_dir, f"dicts/{src_dict}")
        dict_type_dir = "/".join(src_dict_path.split("/")[:-1])
        if not os.path.exists(src_dict_path):
            dict_type_dir = download_from_url(
                src_dict,
                src_dict_path,
                key="dict",
                lang=info.lang,
            )

    if tgt_dict_in:
        tgt_dict = f"{tgt_dict_in}.txt"
        tgt_dict_path = os.path.join(info.root_dir, f"dicts/{tgt_dict}")
        if not os.path.exists(tgt_dict_path):
            download_from_url(
                tgt_dict,
                tgt_dict_path,
                key="dict",
                lang=info.lang,
            )

    # Download or load corresponding tokenizer
    src_tok_path, tgt_tok_path = None, None
    if src_tok:
        src_tok_path = download_or_load(
            f"tokenizers/{src_tok}.zip",
            lang=info.lang,
        )
    if tgt_tok:
        tgt_tok_path = download_or_load(
            f"tokenizers/{tgt_tok}.zip",
            lang=info.lang,
        )

    return TransformerInfo(
        path=model_type_dir,
        dict_path=dict_type_dir,
        # Drop prefix "dict." and postfix ".txt"
        src_dict=".".join(src_dict.split(".")[1:-1]),
        # to follow fairseq's dictionary load process
        tgt_dict=".".join(tgt_dict.split(".")[1:-1]),
        src_tok=src_tok_path,
        tgt_tok=tgt_tok_path,
    )


def download_or_load_misc(info: DownloadInfo) -> str:
    """
    Download (pre-trained) miscellaneous model

    Args:
        info (DownloadInfo): download information

    Returns:
        str: miscellaneous model path

    """
    # Add postfix <.model> for sentencepiece
    if "sentencepiece" in info.n_model:
        info.n_model += ".model"

    # Generate target model path using root directory
    model_path = os.path.join(info.root_dir, info.n_model)
    if not os.path.exists(model_path):
        type_dir = download_from_url(
            info.n_model,
            model_path,
            key="model",
            lang=info.lang,
        )

        if ".zip" in info.n_model:
            zip_file = zipfile.ZipFile(model_path)
            zip_file.extractall(type_dir)
            zip_file.close()

    if ".zip" in info.n_model:
        model_path = model_path[:model_path.rfind(".zip")]
    return model_path


def download_or_load_bart(info: DownloadInfo) -> Union[str, Tuple[str, str]]:
    """
    Download BART model

    Args:
        info (DownloadInfo): download information

    Returns:
        Union[str, Tuple[str, str]]: BART model path (with. corresponding SentencePiece)

    """
    info.n_model += ".pt"

    model_path = os.path.join(info.root_dir, info.n_model)
    if not os.path.exists(model_path):
        download_from_url(
            info.n_model,
            model_path,
            key="model",
            lang=info.lang,
        )

    return model_path


def download(url, out=None, bar=wget.bar_adaptive):
    """High level function, which downloads URL into tmp file in current
    directory and then renames it to filename autodetected from either URL
    or HTTP headers.
    
    Public domain by anatoly techtonik <techtonik@gmail.com>
    Also available under the terms of MIT license
    Copyright (c) 2010-2015 anatoly techtonik

    :param bar: function to track download progress (visualize etc.)
    :param out: output filename or directory
    :return:    filename where URL is downloaded to
    """
    # detect of out is a directory
    outdir = None
    if out and os.path.isdir(out):
        outdir = out
        out = None

    # get filename for temp file in current directory
    prefix = wget.detect_filename(url, out)
    tmpdir = wget.tempfile.TemporaryDirectory()
    (fd, tmpfile) = wget.tempfile.mkstemp(".tmp", prefix=prefix, dir=tmpdir.name)
    os.close(fd)
    os.unlink(tmpfile)

    # set progress monitoring callback
    def callback_charged(blocks, block_size, total_size):
        # 'closure' to set bar drawing function in callback
        wget.callback_progress(blocks, block_size, total_size, bar_function=bar)
    if bar:
        callback = callback_charged
    else:
        callback = None

    if wget.PY3K:
        # Python 3 can not quote URL as needed
        binurl = list(wget.urlparse.urlsplit(url))
        binurl[2] = wget.urlparse.quote(binurl[2])
        binurl = wget.urlparse.urlunsplit(binurl)
    else:
        binurl = url
    (tmpfile, headers) = wget.ulib.urlretrieve(binurl, tmpfile, callback)
    filename = wget.detect_filename(url, out, headers)
    if outdir:
        filename = outdir + "/" + filename

    # add numeric ' (x)' suffix if filename already exists
    if os.path.exists(filename):
        filename = wget.filename_fix_existing(filename)
    wget.shutil.move(tmpfile, filename)
    return filename


def download_from_url(
    n_model: str,
    model_path: str,
    key: str,
    lang: str,
) -> str:
    """
    Download specified model from Tenth

    Args:
        n_model (str): model name
        model_path (str): pre-defined model path
        key (str): type key (either model or dict)
        lang (str): language name

    Returns:
        str: default type directory

    """
    # Get default type dir path
    type_dir = "/".join(model_path.split("/")[:-1])
    os.makedirs(type_dir, exist_ok=True)

    # Get download tenth url
    url = get_download_url(n_model, key=key, lang=lang)

    logging.info("Downloading user-selected model...")
    download(url, type_dir)
    sys.stderr.write("\n")
    sys.stderr.flush()

    return type_dir


def download_or_load(
    n_model: str,
    lang: str,
    custom_save_dir: str = None,
) -> Union[TransformerInfo, str, Tuple[str, str]]:
    """
    Download or load model based on model information

    Args:
        n_model (str): model name
        lang (str): language information
        custom_save_dir (str, optional): user-defined save directory path. defaults to None.

    Returns:
        Union[TransformerInfo, str, Tuple[str, str]]

    """
    root_dir = get_save_dir(save_dir=custom_save_dir)
    info = DownloadInfo(n_model, lang, root_dir)

    if "transformer" in n_model:
        return download_or_load_transformer(info)
    if "bert" in n_model:
        return download_or_load_bert(info)
    if "bart" in n_model and "bpe" not in n_model:
        return download_or_load_bart(info)

    return download_or_load_misc(info)
