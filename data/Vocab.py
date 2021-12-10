from collections import Counter
import numpy as np
from utils.log import logger


class Vocab(object):
    ROOT, PAD, UNK = 0, 1, 2
    def __init__(self, word_counter, state_counter, max_vocab_size = 1000):
        self._id2word = ['<root>', '<pad>', '<unk>']
        self._id2extword = ['<root>', '<pad>', '<unk>']
        self._id2state = []

        for word, count in word_counter.most_common():
            self._id2word.append(word)
            if len(self._id2word) >= max_vocab_size:
                break

        for state, count in state_counter.most_common():
            self._id2state.append(state)


        reverse = lambda x: dict(zip(x, range(len(x))))
        self._word2id = reverse(self._id2word)
        if len(self._word2id) != len(self._id2word):
            logger.error("serious bug: words dumplicated, please check!")

        self._state2id = reverse(self._id2state)
        if len(self._state2id) != len(self._id2state):
            logger.error("serious bug: relations dumplicated, please check!")

        logger.info("state: %s" %(self._id2state))
        logger.info("state size: %d" %(len(self._state2id)))


    def word2id(self, xs):
        if isinstance(xs, list):
            return [self._word2id.get(x, self.UNK) for x in xs]
        return self._word2id.get(xs, self.UNK)

    def id2word(self, xs):
        if isinstance(xs, list):
            return [self._id2word[x] for x in xs]
        return self._id2word[xs]

    def extword2id(self, xs):
        if isinstance(xs, list):
            return [self._extword2id.get(x, self.UNK) for x in xs]
        return self._extword2id.get(xs, self.UNK)

    def id2extword(self, xs):
        if isinstance(xs, list):
            return [self._id2extword[x] for x in xs]
        return self._id2extword[xs]

    def state2id(self, xs):
        if isinstance(xs, list):
            return [self._state2id.get(x) for x in xs]
        return self._state2id.get(xs)

    def id2state(self, xs):
        if isinstance(xs, list):
            return [self._id2state[x] for x in xs]
        return self._id2state[xs]


    @property
    def word_size(self):
        return len(self._id2word)

    @property
    def extword_size(self):
        return len(self._id2extword)

    @property
    def state_size(self):
        return len(self._id2state)

    def load_pretrained_embs(self, embfile):
        embedding_dim = -1
        word_count = 0
        with open(embfile, encoding='utf-8') as f:
            for line in f.readlines():
                if word_count < 1:
                    values = line.split()
                    embedding_dim = len(values) - 1
                word_count += 1
        print('Total words: ' + str(word_count) + '\n')
        print('The dim of pretrained embeddings: ' + str(embedding_dim) + '\n')

        index = len(self._id2extword)
        embeddings = np.zeros((word_count + index, embedding_dim))
        with open(embfile, encoding='utf-8') as f:
            for line in f.readlines():
                values = line.split()
                self._id2extword.append(values[0])
                vector = np.array(values[1:], dtype='float64')
                embeddings[self.UNK] += vector
                embeddings[index] = vector
                index += 1

        embeddings[self.UNK] = embeddings[self.UNK] / word_count
        #embeddings = embeddings / np.std(embeddings)

        reverse = lambda x: dict(zip(x, range(len(x))))
        self._extword2id = reverse(self._id2extword)

        if len(self._extword2id) != len(self._id2extword):
            print("serious bug: extern words dumplicated, please check!")

        return embeddings

def create_vocab(instances, max_vocab_size = 1000):
    word_counter = Counter()
    state_counter = Counter()
    for instance in instances:
        for idx, word in enumerate(instance.texts):
            word_counter[word] += 1
        for state in instance.state:
            state_counter[state] += 1
    return Vocab(word_counter, state_counter, max_vocab_size)
