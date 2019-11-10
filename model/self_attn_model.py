"""
THe implementation of multi-head self-attention model

Author: Haotian Xue
"""

import copy
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sen_tensor_model_class import SenTensorModel
from utils import layers, attention


class MultiHeadSelfAttnModel(SenTensorModel):
    def __init__(self,
                 train_data_set,
                 test_data_set,
                 hyper_parameter,
                 train_requirement,
                 is_gpu=torch.cuda.is_available(),
                 model_save_path="../trained_model/self_attn_model.pt"):
        super(MultiHeadSelfAttnModel, self).__init__(train_data_set,
                                                     test_data_set,
                                                     hyper_parameter,
                                                     train_requirement,
                                                     is_gpu,
                                                     model_save_path)
        self.batch_size = self.train_requirement["batch_size"]
        self.train_data_loader = DataLoader(self.train_data_set, self.batch_size, shuffle=True)
        self.test_data_loader = DataLoader(self.test_data_set, self.batch_size, shuffle=False)
        self.model = self.build_model()
        if is_gpu:
            self.model = self.model.cuda()
        self.train_test()

    def build_model(self):
        d_w, hidden_dim, num_layers, num_heads = self.extract_hyper_parameters()
        print("-----Start building model-----")
        model = MultiHeadSelfAttnModelHelper(d_w=d_w,
                                             word_emb_weight=torch.from_numpy(self.test_data_set.word_embedding),
                                             hidden_dim=hidden_dim,
                                             num_layers=num_layers,
                                             num_heads=num_heads)
        print("-----Finish building model-----")
        return model

    def extract_hyper_parameters(self):
        return self.hyper_parameter["d_w"], \
               self.hyper_parameter["hidden_dim"], \
               self.hyper_parameter["num_layers"], \
               self.hyper_parameter["num_heads"]


class MultiHeadSelfAttnModelHelper(nn.Module):
    def __init__(self, d_w, hidden_dim, word_emb_weight, num_layers=4,
                 num_heads=5, dropout=0.1, num_classes=2):
        super(MultiHeadSelfAttnModelHelper, self).__init__()
        self.w2v = nn.Embedding.from_pretrained(word_emb_weight, freeze=False)
        c = copy.deepcopy
        d_model = d_w
        self_attn = attention.MultiHeadAttention(h=num_heads, d_model=d_model, dropout=dropout)
        ff = layers.PositionwiseFeedForward(d_model=d_model, d_ff=hidden_dim, dropout=dropout)
        word_attn = attention.WordAttention(d_model)  # (batch, sen, d_model) => (batch, d_model)
        self.model = nn.Sequential(
            layers.Encoder(layers.EncoderLayer(d_model, c(self_attn), c(ff), dropout), num_layers),
            word_attn,
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_classes)
        )
        for p in self.model.parameters():
            if p.dim() > 1:  # dim: 维度数
                nn.init.xavier_uniform_(p)

    def forward(self, x):  # x: (batch, max_sen_len)
        x = self.w2v(x)   # (batch_size, max_sen_len, d_w)
        output = self.model(x)  # (batch_size, num_classes)
        return output


if __name__ == "__main__":
    from data_fetcher.dataFetcher import SenSemEvalDataSet
    train_requirement = {"num_epoch": 20, "batch_size": 32}
    hyper_parameter = {"d_w": 50, "hidden_dim": 256, "num_layers": 4, "num_heads": 5}
    train_data_set = SenSemEvalDataSet("../data/train.txt", "../data/word_embedding/glove.6B.50d.txt", 50, True)
    test_data_set = SenSemEvalDataSet("../data/test.txt", "../data/word_embedding/glove.6B.50d.txt", 50, 150, True)
    model = MultiHeadSelfAttnModel(train_data_set, test_data_set, hyper_parameter, train_requirement)
