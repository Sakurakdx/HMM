#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   Data.py
@Time    :   2021/12/05 13:51:13
@Author  :   sakurakdx
@Contact :   sakurakdx@163.com
"""

from utils.utils import string_compiler
import pandas as pd
from utils.log import logger
import random


class Instance(object):
    """ """

    def __init__(self, id, texts, labels) -> None:
        super().__init__()
        self.id = id
        assert len(texts) == len(labels)
        self.texts = texts
        self.state = labels
        


def read_corpus(file, is_training=True):
    """获得数据
    如果是csv，返回pd
    如果是json文件，返回dict的列表，dict中包含context和qas两个keys

    Args:
        file (str): 读取数据的路径

    Returns:
        instances: instance对象的列表
    """
    with open(file, "r", encoding="utf-8") as f:
        data = pd.read_csv(f, delimiter="\t")

    instances = []
    for idx, info in enumerate(data["语料"]):
        instance = info2instance(idx, info)
        instances.append(instance)
    logger.info("Successfully load %d instance from %s" %(len(instances), file))

    return instances


def info2instance(id, info):
    """将info转为instance实例

    Args:
        info (dict): [description]
        is_training (bool): [description]

    Returns:
        [type]: [description]
    """
    texts = []
    labels = []
    flag = False
    temp = ""
    for word in info.strip().split(" "):
        if word == "":
            continue
        
        if not string_compiler(word):
            temp = word.replace("[", "").replace("]", "") + temp
            flag = True
            continue

        if flag:
            word = temp.strip() + word.strip()
            temp = ""
            flag = False
        text = word.split("/")[0]
        label = word.split("/")[1]
        if not label.encode('UTF-8').isalpha():
            continue
        texts.append(text)
        labels.append(label)
        
    instance = Instance(id, texts, labels)

    return instance

def split_instance(instances, ratio=0.8):
    batch = len(instances)
    train_idx = random.sample(range(batch), int(batch * ratio))
    test_idx = []
    for idx in range(batch):
        if idx not in train_idx:
            test_idx.append(idx)

    train_instances = map(lambda i: instances[i], train_idx)
    test_instaces = map(lambda i: instances[i], test_idx)

    return list(train_instances), list(test_instaces)    