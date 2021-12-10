#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   utils.py
@Time    :   2021/10/20 19:29:30
@Author  :   sakurakdx
@Contact :   sakurakdx@163.com
"""

import sys

sys.path.extend(["./", "../"])
import torch
import numpy as np
import random
import pickle
import json
from utils.log import logger
import re

def string_compiler(str):
    """判断是否以字母结尾

    Args:
        str (str): 字符串

    Returns:
        bool: 
    """
    text = re.compile(r".*[a-zA-Z]$")
    if text.match(str):
        return True
    else:
        return False


def load_pkl(file_path):
    """加载pkl

    Args:
        file_path (str): 加载文件路径

    Returns:
        dict: 加载之后的dict
    """
    with open(file_path, "rb") as f:
        return pickle.load(f)


def dump_pkl(obj, file_path):
    """将obj保存为pkl文件

    Args:
        obj (dict): 保存的数据
        file_path (str): 保存的路径一般是.pkl
    """
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


def load_json(file_path):
    """加载json

    Args:
        file_path (str): 加载的路径

    Returns:
        list: 数据的列表
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj, file_path):
    """保存json

    Args:
        obj ([type]): data
        file_path (str): 保存路径
    """
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4)


def same_seed(seed):
    """设置随机种子，使得结果可复现

    Args:
        seed (int): 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)


def set_seed():
    """初始化随机种子大小，并进行打印"""
    seed = random.randint(0, 10000)
    logger.info(f"random seed is {seed}")
    same_seed(same_seed)


def _is_whitespace(c):
    """判断字符串是否是空白的

    Args:
        c (str): 要判断的字符串

    Returns:
        bool: 空白返回True，否则返回False
    """
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def whitespace_tokenize(text):
    """清除空格并返回按空格划分的token

    Args:
        text (str): 需要分割的文本

    Returns:
        list: 文本按空格划分的token
    """
    text = text.strip()
    if not text:
        return []

    token = text.split()
    return token


def to_list(tensor):
    """将Tensor转为list

    Args:
        tensor (torch.Tensor): Tensor

    Returns:
        list: Tensor转为list
    """
    return tensor.detach().cpu().tolist()


def _improve_answer_span(
    doc_tokens, input_start, input_end, tokenizer, orig_answer_text
):
    """Returns tokenized answer spans that better match the annotated answer."""
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start : (new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _new_check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    # if len(doc_spans) == 1:
    # return True
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span["start"] + doc_span["length"] - 1
        if position < doc_span["start"]:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span["start"]
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span["length"]
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index
