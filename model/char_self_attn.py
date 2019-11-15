"""
The implementation of character level multi-head self attention

Author: Haotian Xue
"""

import numpy as np
import torch
import copy
import torch.nn as nn
import attention
import layers
from sen_tensor_model_class import SenTensorModel


class CharAttnModel(SenTensorModel):

    def __init__(self,
                 train_data_set,
                 test_data_set,
                 hyper_parameter,
                 train_requirement,
                 is_gpu=torch.cuda.is_available(),
                 model_save_path="../trained_model/char_cnn_model.pt"):
        super(CharAttnModel, self).__init__(train_data_set,
                                            test_data_set,
                                            hyper_parameter,
                                            train_requirement,
                                            is_gpu,
                                            model_save_path)
        self.model = self.build_model()
        if is_gpu:
            self.model = self.model.cuda()
        self.train_test()
        # self.load_test()

    def build_model(self):
        d_w, d_e, num_heads, num_layers, hidden_dim, window_sizes, num_filter, dropout_p = \
            self.extract_hyper_parameters()
        print("-----Start building model-----")
        model = CharAttnModelHelper(d_w,
                                    d_e,
                                    num_heads,
                                    num_layers,
                                    hidden_dim,
                                    window_sizes,
                                    num_filter,
                                    dropout_p,
                                    self.is_gpu)
        print("-----Finish building model-----")
        return model

    def extract_hyper_parameters(self):
        return self.hyper_parameter["d_w"], \
               self.hyper_parameter["d_e"], \
               self.hyper_parameter["num_heads"], \
               self.hyper_parameter["num_layers"], \
               self.hyper_parameter["hidden_dim"], \
               self.hyper_parameter["window_sizes"], \
               self.hyper_parameter["num_filter"], \
               self.hyper_parameter["dropout_p"]


class CharAttnModelHelper(nn.Module):

    def __init__(self, d_w, d_e, num_heads, num_layers, hidden_dim,
                 window_sizes, num_filter, dropout_p, is_gpu, num_classes=2):
        super(CharAttnModelHelper, self).__init__()
        self.w2v = nn.Embedding(97, d_w)
        self.pos_embedding = nn.Embedding(842, d_e)
        self.is_gpu = is_gpu
        c = copy.deepcopy
        d_model = d_w + d_e
        self.cnn_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=d_model,
                      kernel_size=(3, d_model),
                      stride=(1, 1),
                      padding=(1, 0))  # (batch, d_model, max_sen_len, 1)
        )
        self.cnn_layer1.apply(self.weights_init)
        self_attn = attention.MultiHeadAttention(h=num_heads, d_model=d_model, dropout=dropout_p)
        ff = layers.PositionwiseFeedForward(d_model=d_model, d_ff=hidden_dim, dropout=dropout_p)
        self.self_attn_layer = nn.Sequential(
            layers.Encoder(layers.EncoderLayer(d_model, c(self_attn), c(ff), dropout_p), num_layers)
        )  # (batch, max_sen_len, d_w + d_e)
        for p in self.self_attn_layer.parameters():
            if p.dim() > 1:  # dim: 维度数
                nn.init.xavier_uniform_(p)
        self.cnn_layer2 = CNNLayers(d_model, num_filter, window_sizes, dropout_p, is_gpu)
        # (batch, len(window_sizes), num_filter) => (batch, num_filter)
        self.word_attn = attention.WordAttention(num_filter)
        for p in self.word_attn.parameters():
            if p.dim() > 1:  # dim: 维度数
                nn.init.xavier_uniform_(p)
        self.linear_layer = nn.Sequential(
            nn.Linear(num_filter, num_filter // 2),
            nn.Dropout(dropout_p),
            nn.Tanh(),
            nn.Linear(num_filter // 2, num_classes)
        )
        self.linear_layer.apply(self.weights_init)

    def forward(self, x):
        batch_size, sen_len = x.shape[0], x.shape[1]
        pos_x = torch.from_numpy(np.vstack([np.arange(sen_len, dtype=np.int64)] * batch_size))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pos_x = pos_x.to(device)
        pos_x = self.pos_embedding(pos_x)  # (batch, sen_len, d_e)
        x = self.w2v(x)  # (batch, sen_len, d_w)
        x = torch.cat([x, pos_x], dim=2)  # (batch, sen_len, d_model=d_w+d_e)
        x = torch.unsqueeze(x, dim=1)  # (batch, 1, sen_len, d_model)
        out = self.cnn_layer1(x)  # (batch, d_model, sen_len, 1)
        out = out.view(out.shape[0], out.shape[2], -1)  # (batch, sen_len, d_model)
        out = self.self_attn_layer(out)  # (batch, sen_len, d_model)
        out = torch.unsqueeze(out, dim=1)  # (batch, 1, sen_len, d_model)
        out = self.cnn_layer2(out)  # (batch, len(window_size), num_filter)
        out = self.word_attn(out)  # (batch, num_filter)
        out = self.linear_layer(out)  # (batch, 2)
        return out

    # method to initialize the model weights (in order to improve performance)
    def weights_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class CNNLayers(nn.Module):

    def __init__(self, d_model, num_filter, window_sizes, dropout_p, is_gpu):
        super(CNNLayers, self).__init__()
        self.is_gpu = is_gpu
        self.cnn_layers = []
        for window_size in window_sizes:
            cnn_layer = nn.Sequential(
                nn.Conv2d(in_channels=1,
                          out_channels=num_filter,
                          kernel_size=(window_size, d_model),
                          stride=(1, 1),
                          padding=(0, 0)),  # (batch, num_filter, max_sen_len - window_size + 1, 1)
                nn.MaxPool2d(kernel_size=(842 - window_size + 1, 1),
                             stride=(1, 1)),  # (batch, num_filter, 1, 1)
                nn.Dropout(dropout_p),
            )
            cnn_layer.apply(self.weights_init)
            self.cnn_layers.append(cnn_layer)

    def forward(self, x):
        out_list = []
        for i, cnn_layer in enumerate(self.cnn_layers):
            if self.is_gpu:
                cnn_layer = cnn_layer.cuda()
            out = cnn_layer(x)  # (batch_size, num_filter, 1, 1)
            out = out.view(out.shape[0], -1)  # (batch_size, num_filter)
            out_list.append(out)
        out = torch.cat(out_list, dim=1)  # (batch_size, num_filter * len(out_list))
        out = out.view(out.shape[0], len(out_list), -1)  # (batch_size, len(out_lsit), num_filter)
        return out

    # method to initialize the model weights (in order to improve performance)
    def weights_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


if __name__ == "__main__":
    from data_fetcher.dataFetcher import CharSemEvalDataSet
    train_requirement = {"num_epoch": 50, "batch_size": 32, "lr": 3e-4}
    hyper_parameter = {"d_w": 50, "d_e": 10, "num_heads": 3, "num_layers": 3,
                       "hidden_dim": 128, "window_sizes": [3, 5, 7, 11], "num_filter": 256, "dropout_p": 0.5}
    train_data_set = CharSemEvalDataSet("../data/train.txt", None, 50, True, 842)
    test_data_set = CharSemEvalDataSet("../data/test.txt", None, 50, True, 842)
    model = CharAttnModel(train_data_set, test_data_set, hyper_parameter, train_requirement)