"""
    Implement different data format for different data set

    Author: Haotian Xue
"""

from utils.data_set_class import DataFetcher
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch import cuda


class SenSemEvalDataSet(Dataset):

    """
    Sentence level SemEval2020 task6 data set
    """

    def __init__(self, data_path, w2v_path, emb_dim, padding=False, max_sen_len=150, is_gpu=cuda.is_available()):
        self.is_gpu = is_gpu
        self.data_fetcher = SenSemEvalHelper(data_path, w2v_path, emb_dim, padding, max_sen_len)
        self.x = self.data_fetcher.data_x
        self.y = self.data_fetcher.data_y
        self.num_data = self.y.shape[0]
        self.word_embedding = self.data_fetcher.word_embedding

    def __getitem__(self, index):
        if self.is_gpu:
            return torch.LongTensor(self.x[index]).cuda(), torch.LongTensor(self.y[index]).cuda()
        return self.data_fetcher.data_x[index], self.data_fetcher.data_y[index]

    def __len__(self):
        return self.data_fetcher.data_y.shape[0]


class SenSemEvalHelper(DataFetcher):

    def __init__(self, data_path, w2v_path, emb_dim, padding=False, max_sen_len=150):
        super(SenSemEvalHelper, self).__init__(data_path, w2v_path, emb_dim, padding, max_sen_len)
        self.data_x, self.data_y, self.max_sen_len = self.load_data()

    def get_x_id(self, x_tokens):
        """
        Given tokens of the sentence, return the ids of each token
        :param x_tokens:
        :return: [token_id]
        """
        x = []
        for x_token in x_tokens:
            if x_token in self.word2id:
                x.append(self.word2id[x_token])
            else:
                x.append(self.word2id[self.OOV])
        return x

    def load_data(self):
        """
        Extract raw input into matrix form
        :return: data_x :: ndarray (if padding, [ndarray] otherwise);  data_y: ndarray
        """
        data_x = []
        data_y = []
        max_len = 0
        with open(self.data_path, 'r') as file:
            for i, line in enumerate(file):
                tokens = line.strip().split()[1:]  # remove first ""
                if len(tokens) == 0:
                    continue
                x_tokens = tokens[:-1]
                if self.padding and len(x_tokens) > max_len:
                    max_len = len(x_tokens)
                y_token = int(tokens[-1][1])
                data_x.append(torch.tensor(self.get_x_id(x_tokens), dtype=torch.int64))
                data_y.append(y_token)
        if self.padding:
            data_x = pad_sequence(data_x, batch_first=True).numpy()
        data_y = np.array(data_y, dtype=np.int64)
        return data_x, data_y, max_len


if __name__ == "__main__":
    ds = SenSemEvalDataSet("../data/train.txt", "../data/word_embedding/glove.6B.50d.txt", 50, True)
    ds_loader = DataLoader(ds, 4, False)
    print(ds.num_data)
    for i, d in enumerate(ds_loader):
        x, y = d
        print(x.size())
        print(x)
        # print(y.size())
        print(y)
        break
