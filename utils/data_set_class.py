"""
This file is the interface class for data_loader.

Author: Haotian Xue
"""
from abc import abstractmethod
import numpy as np


class DataFetcher:

    def __init__(self, data_path, w2v_path, emb_dim, padding=False, max_sen_len=150):
        self.data_path = data_path
        self.emb_dim = emb_dim
        self.max_sen_len = max_sen_len
        self.OOV = 'OOV'  # 'out of vocabulary'
        self.BLANK = 'BLANK'
        self.padding = padding
        if w2v_path is not None:
            self.word2id, self.word_embedding = self._load_w2v(w2v_path)

    def _load_w2v(self, w2v_path):
        """
        load pre-trained word embedding file
        :param w2v_path: file path
        :return: word2id :: {token: token_id}, word_embedding :: np.array(n, emd_dim)
        """
        vec = []
        word2id = {}
        word2id[self.BLANK] = len(word2id)
        word2id[self.OOV] = len(word2id)
        vec.append(np.random.normal(size=self.emb_dim, loc=0, scale=0.05))
        vec.append(np.random.normal(size=self.emb_dim, loc=0, scale=0.05))
        with open(w2v_path, 'r') as f:
            for line in f:
                o_line = line
                line = line.replace(u'\xa0', u'')
                tokens = line.split()
                if tokens[0].isnumeric():
                    continue
                word2id[tokens[0]] = len(word2id)
                if np.array([float(x) for x in tokens[1:]]).shape[0] != self.emb_dim:
                    del word2id[tokens[0]]
                    continue
                vec.append(np.array([float(x) for x in tokens[1:]]))

        word_embedding = np.array(vec, dtype=np.float32)
        return word2id, word_embedding

    @abstractmethod
    def load_data(self):
        pass
