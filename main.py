#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   main.py
@Time    :   2021/12/08 11:08:12
@Author  :   sakurakdx
@Contact :   sakurakdx@163.com
"""
import argparse

from data.Data import read_corpus, split_instance
from data.Vocab import create_vocab
from modules.HiddenMarkov import HiddenMarkov
from script.evaluate import compute_acc
from utils.log import init_logger, logger


def main(args):
    instances = read_corpus(args.data_file)
    vocab = create_vocab(instances, 100000)
    train_instances, test_instances = split_instance(instances)
    model = HiddenMarkov(vocab, instances)

    # 计算准确率
    preds = []
    labels = []
    for instance in test_instances:
        pred = model.predict(instance.texts)
        preds.extend(pred)
        labels.extend(instance.state)
    acc = compute_acc(preds, labels)
    logger.info("acc: %.2f" % (acc))

    # 挑选样例进行预测
    # for instance in test_instances[:10]:
    #     pred = model.predict(instance.texts)
    #     logger.info("true: %s" %instance.state)
    #     logger.info("pred: %s" %pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", default="dataset/corpus.txt")

    args = parser.parse_args()
    init_logger()

    main(args)
