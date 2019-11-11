"""
The implementation of cnn + word attention model

Author: Haotian Xue
"""

import copy
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import attention, layers
from sen_tensor_model_class import SenTensorModel


class CnnAttnModel(SenTensorModel):

    def __init__(self,
                 train_data_set,
                 test_data_set,
                 hyper_parameter,
                 train_requirement,
                 is_gpu=torch.cuda.is_available(),
                 model_save_path="../trained_model/cnn_model.pt"):
        super(CnnAttnModel, self).__init__(train_data_set,
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
        # self.load_test()

    def build_model(self):
        d_w, num_filter, window_size, num_heads, hidden_dim, num_layers, dropout = self.extract_hyper_parameters()
        print("-----Start building model-----")
        model = CnnAttnModelHelper(d_w,
                                   torch.from_numpy(self.test_data_set.word_embedding),
                                   num_filter,
                                   window_size,
                                   num_heads,
                                   hidden_dim,
                                   num_layers,
                                   dropout)
        print("-----Finish building model-----")
        return model

    def extract_hyper_parameters(self):
        return self.hyper_parameter["d_w"], \
               self.hyper_parameter["num_filter"], \
               self.hyper_parameter["window_size"], \
               self.hyper_parameter["num_heads"], \
               self.hyper_parameter["hidden_dim"], \
               self.hyper_parameter["num_layers"], \
               self.hyper_parameter["dropout"]


class CnnAttnModelHelper(nn.Module):

    def __init__(self, d_w, word_emb_weight, num_filter, window_size, num_heads, hidden_dim, num_layers, dropout, num_classes=2):
        super(CnnAttnModelHelper, self).__init__()
        self.num_filter = num_filter
        self.w2v = nn.Embedding.from_pretrained(word_emb_weight, freeze=False)
        self.cnn_layer = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=num_filter,
                      kernel_size=(window_size, d_w),
                      stride=(1, 1),
                      padding=(1, 0)),  # out_shape: (batch_size, num_filter, max_sen_len, 1)
            nn.MaxPool2d(kernel_size=(window_size, 1),
                         stride=(1, 1)),  # out_shape: (batch_size, num_filter, max_sen_len-window_size+1, 1)
            nn.Tanh()
        )
        self.cnn_layer.apply(self.weights_init)
        c = copy.deepcopy
        d_model = num_filter
        self_attn = attention.MultiHeadAttention(h=num_heads, d_model=d_model, dropout=dropout)
        ff = layers.PositionwiseFeedForward(d_model=d_model, d_ff=hidden_dim, dropout=dropout)
        self.word_attn = attention.WordAttention(num_filter)  # out_shape: (batch_size, num_filter)
        self.attn_model = nn.Sequential(
            layers.Encoder(layers.EncoderLayer(d_model, c(self_attn), c(ff), dropout), num_layers),
            self.word_attn,
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_classes)
        )
        for p in self.attn_model.parameters():
            if p.dim() > 1:  # dim: 维度数
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        x = self.w2v(x)  # (batch_size, max_sen_len, d_w)
        x = torch.unsqueeze(x, dim=1)  # (batch_size, 1, max_sen_len, d_w)
        out = self.cnn_layer(x)  # (batch_size, num_filter, max_sen_len-window_size+1, 1)
        out = out.view(out.shape[0], out.shape[2], -1)  # (batch_size, max_sen_len-window_size+1, num_filter)
        out = self.attn_model(out)  # (batch_size, 2)
        return out

    # method to initialize the model weights (in order to improve performance)
    def weights_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


if __name__ == "__main__":
    from data_fetcher.dataFetcher import SenSemEvalDataSet
    train_requirement = {"num_epoch": 30, "batch_size": 8}
    # num_heads, hidden_dim, num_layers, dropout
    hyper_parameter = {"d_w": 50, "num_filter": 256, "window_size": 3, "num_heads": 4, "hidden_dim": 128, "num_layers": 2, "dropout": 0.2}
    train_data_set = SenSemEvalDataSet("../data/train.txt", "../data/word_embedding/glove.6B.50d.txt", 50, True)
    test_data_set = SenSemEvalDataSet("../data/test.txt", "../data/word_embedding/glove.6B.50d.txt", 50, True, 150, is_gpu=False)
    model = CnnAttnModel(train_data_set, test_data_set, hyper_parameter, train_requirement)

