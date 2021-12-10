#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   HiddenMarkov.py
@Time    :   2021/12/08 15:34:59
@Author  :   sakurakdx
@Contact :   sakurakdx@163.com
'''
from re import S
import numpy as np

class HiddenMarkov:
    def __init__(self, vocab, instances) -> None:
        self.vocab = vocab
        self.state_size = vocab.state_size
        self.instances = instances
        self.TransitionProbability()
        self.StartProbability()
        self.EmissionProbability()

        
    def TransitionProbability(self):
        transition_probabilily = np.zeros((self.state_size, self.state_size))
        for instance in self.instances:
            state_ids = self.vocab.state2id(instance.state)
            for idx, state_id in enumerate(state_ids):
                if not (idx + 1 < len(state_ids)):
                    break
                   
                transition_probabilily[state_id][state_ids[idx + 1]] += 1
        
        self.transition_probabilily = self.normalize(transition_probabilily)

    def normalize(self, x):
        """归一化

        Args:
            x (np.array): 归一化的矩阵，n*m
        """
        if len(x.shape) == 2:
            return x / np.sum(x, axis=1, keepdims=True)
        elif len(x.shape) == 1:
            return x / np.sum(x, axis=0, keepdims=True)
    
    def StartProbability(self):
        start_probability = np.zeros(self.state_size)

        for instance in self.instances:
            state_ids = self.vocab.state2id(instance.state)
            start_probability[state_ids[0]] += 1

        self.start_probability = self.normalize(start_probability)
    
    def EmissionProbability(self):
        emission_probability = np.zeros((self.vocab.state_size, self.vocab.word_size))

        for instance in self.instances:
            state_ids = self.vocab.state2id(instance.state)
            word_ids = self.vocab.word2id(instance.texts)

            for idx, idy in zip(state_ids, word_ids):
                emission_probability[idx][idy] += 1
            
        self.emission_probability = self.normalize(emission_probability)
    
    def predict(self, texts):
        length = len(texts)
        score = np.zeros((length, self.vocab.state_size))
        path = np.zeros((length, self.vocab.state_size))

        texts_ids = self.vocab.word2id(texts)

        start_probability = self.start_probability
        emission_probability = self.emission_probability
        transition_probabilily = self.transition_probabilily

        for idx, idz in enumerate(texts_ids):  # idx: t, idy: i, idz: o_t 
            for idy in range(self.state_size):
                if idx == 0:
                    score[0][idy] = start_probability[idy] * emission_probability[idy][idz]
                    path[0][idy] = 0
                else:
                    score[idx][idy] = np.max(score[idx - 1] * transition_probabilily[:][idy]) * emission_probability[idy][idz]
                    path[idx][idy] = np.argmax(score[idx - 1] * transition_probabilily[:][idy])

        p = np.max(score[len(texts_ids) - 1])
        results = np.zeros(length, dtype=np.int32)
        results[-1] = np.argmax(score[len(texts_ids) - 1])
        for t in range(len(texts_ids) - 2, 0, -1):
            results[t] = path[t + 1][int(results[t + 1])]
        
        return self.vocab.id2state(results.tolist())
