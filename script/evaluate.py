import numpy as np


def compute_acc(predicts, targets):
    """计算准确率

    Args:
        predicts (list): 预测值
        targets (list): 真实值
    """
    assert len(predicts) == len(targets)  
    acc_num = 0
    for pred, true in zip(predicts, targets):
        if pred == true:
            acc_num += 1

    return acc_num / len(predicts)
