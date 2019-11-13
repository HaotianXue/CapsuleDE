"""
The impelmentation of Attention-CNN model≥

Author: Haotian Xue
"""

import copy
import torch
import torch.nn as nn
from sen_tensor_model_class import SenTensorModel
from utils import layers, attention


class AttnCnnModel(SenTensorModel):
    def __init__(self,
                 train_data_set,
                 test_data_set,
                 hyper_parameter,
                 train_requirement,
                 is_gpu=torch.cuda.is_available(),
                 model_save_path="../trained_model/self_attn_model.pt"):
        super(AttnCnnModel, self).__init__(train_data_set,
                                           test_data_set,
                                           hyper_parameter,
                                           train_requirement,
                                           is_gpu,
                                           model_save_path)
        self.model = self.build_model()
        if is_gpu:
            self.model = self.model.cuda()
        self.train_test()

    def build_model(self):
        d_w, hidden_dim, num_layers, num_heads, window_size, num_filter = self.extract_hyper_parameters()
        print("-----Start building model-----")
        model = AttnCnnModelHelper(d_w=d_w,
                                   word_emb_weight=torch.from_numpy(self.test_data_set.word_embedding),
                                   hidden_dim=hidden_dim,
                                   num_layers=num_layers,
                                   num_heads=num_heads,
                                   window_size=window_size,
                                   num_filter=num_filter)
        print("-----Finish building model-----")
        return model

    def extract_hyper_parameters(self):
        return self.hyper_parameter["d_w"], \
               self.hyper_parameter["hidden_dim"], \
               self.hyper_parameter["num_layers"], \
               self.hyper_parameter["num_heads"], \
               self.hyper_parameter["window_size"], \
               self.hyper_parameter["num_filter"]


class AttnCnnModelHelper(nn.Module):
    def __init__(self, d_w, hidden_dim, word_emb_weight, num_layers=4,
                 num_heads=5, window_size=3, num_filter=128, dropout_p=0.1, num_classes=2):
        super(AttnCnnModelHelper, self).__init__()
        self.w2v = nn.Embedding.from_pretrained(word_emb_weight, freeze=False)
        c = copy.deepcopy
        d_model = d_w
        self_attn = attention.MultiHeadAttention(h=num_heads, d_model=d_model, dropout=dropout_p)
        ff = layers.PositionwiseFeedForward(d_model=d_model, d_ff=hidden_dim, dropout=dropout_p)
        self.attn_layer = nn.Sequential(
            layers.Encoder(layers.EncoderLayer(d_model, c(self_attn), c(ff), dropout_p), num_layers)
        )  # (batch, sen_len, d_model)
        self.cnn_layer = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=num_filter,
                      kernel_size=(window_size, d_w),
                      stride=(1, 1),
                      padding=(1, 0)),  # out_shape: (batch_size, num_filter, max_sen_len, 1)
            nn.MaxPool2d(kernel_size=(150, 1),
                         stride=(1, 1)),  # out_shape: (batch_size, num_filter, 1, 1)
            nn.Dropout(dropout_p)
        )
        self.linear_layer = nn.Sequential(
            nn.Linear(num_filter, num_filter // 3),
            nn.Tanh(),
            nn.Linear(num_filter // 3, num_classes)
        )
        for p in self.attn_layer.parameters():
            if p.dim() > 1:  # dim: 维度数
                nn.init.xavier_uniform_(p)
        for p in self.cnn_layer.parameters():
            if p.dim() > 1:  # dim: 维度数
                nn.init.xavier_uniform_(p)
        for p in self.linear_layer.parameters():
            if p.dim() > 1:  # dim: 维度数
                nn.init.xavier_uniform_(p)

    def forward(self, x):  # x: (batch, max_sen_len)
        x = self.w2v(x)   # (batch_size, max_sen_len, d_w)
        out = self.attn_layer(x)  # (batch_size, max_sen_len, d_w)
        out = torch.unsqueeze(out, dim=1)  # (batch_size, 1, max_sen_len, d_w)
        out = self.cnn_layer(out)  # (batch_size, num_filter, 1, 1)
        out = out.view(out.shape[0], -1)  # (batch_size, num_filter)
        out = self.linear_layer(out)  # (batch_size, 2)
        return out


if __name__ == "__main__":
    from data_fetcher.dataFetcher import SenSemEvalDataSet
    train_requirement = {"num_epoch": 20, "batch_size": 32, "lr": 3e-4}
    hyper_parameter = {"d_w": 50, "hidden_dim": 128, "num_layers": 2, "num_heads": 1, "window_size": 3, "num_filter": 128}
    train_data_set = SenSemEvalDataSet("../data/train.txt", "../data/word_embedding/glove.6B.50d.txt", 50, True)
    test_data_set = SenSemEvalDataSet("../data/test.txt", "../data/word_embedding/glove.6B.50d.txt", 50, 150, True)
    model = AttnCnnModel(train_data_set, test_data_set, hyper_parameter, train_requirement)

