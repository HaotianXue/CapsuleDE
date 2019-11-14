"""
The implementation of CNN + positional embedding

Author: Haotian Xue
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sen_tensor_model_class import SenTensorModel


class CnnPosModel(SenTensorModel):

    def __init__(self,
                 train_data_set,
                 test_data_set,
                 hyper_parameter,
                 train_requirement,
                 is_gpu=torch.cuda.is_available(),
                 model_save_path="../trained_model/cnn_pos_model.pt"):
        super(CnnPosModel, self).__init__(train_data_set,
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
        d_w, d_b, num_filter, window_size, dropout_p = self.extract_hyper_parameters()
        print("-----Start building model-----")
        model = CnnPosModelHelper(d_w,
                                  d_b,
                                  torch.from_numpy(self.test_data_set.word_embedding),
                                  num_filter,
                                  window_size,
                                  dropout_p)
        print("-----Finish building model-----")
        return model

    def extract_hyper_parameters(self):
        return self.hyper_parameter["d_w"], \
               self.hyper_parameter["d_b"], \
               self.hyper_parameter["num_filter"], \
               self.hyper_parameter["window_size"], \
               self.hyper_parameter["dropout_p"]


class CnnPosModelHelper(nn.Module):
    def __init__(self, d_w, d_b, word_emb_weight, num_filter, window_size, dropout_p, num_classes=2):
        super(CnnPosModelHelper, self).__init__()
        self.pos_embedding = nn.Embedding(150, d_b)
        self.w2v = nn.Embedding.from_pretrained(word_emb_weight, freeze=False)
        self.cnn_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=num_filter,
                      kernel_size=(window_size, d_w + d_b),
                      stride=(1, 1),
                      padding=(1, 0)),  # out_shape: (batch_size, num_filter, max_sen_len, 1)
            nn.MaxPool2d(kernel_size=(window_size, 1),
                         stride=(1, 1)),  # out_shape: (batch_size, num_filter, max_sen_len - window_size + 1, 1)
            nn.Dropout(dropout_p)
        )  # out_shape: (batch_size, num_filter, 1, 1)
        self.cnn_layer1.apply(self.weights_init)
        self.cnn_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=num_filter // 2,
                      kernel_size=(window_size, num_filter),
                      stride=(1, 1),
                      padding=(1, 0)),  # out_shape: (batch_size, num_filter/2, max_sen_len - window_size + 1, 1)
            nn.MaxPool2d(kernel_size=(150 - window_size + 1, 1),
                         stride=(1, 1)),  # out_shape: (batch_size, num_filter // 2, 1, 1)
            nn.Dropout(dropout_p)
        )
        self.linear_layer = nn.Sequential(
            nn.Linear(num_filter // 2, num_classes)
        )  # out_shape: (batch_size, num_classes)
        self.linear_layer.apply(self.weights_init)

    def forward(self, x):
        batch_size, sen_len = x.shape[0], x.shape[1]
        pos_x = torch.from_numpy(np.vstack([np.arange(sen_len)] * batch_size))
        pos_x = self.pos_embedding(pos_x)  # (batch_size, max_sen_len, d_b)
        x = self.w2v(x)  # (batch_size, max_sen_len, d_w)
        x = torch.cat([x, pos_x], dim=2)  # (batch_size, max_sen_len, d_w + d_b)
        x = torch.unsqueeze(x, dim=1)  # (batch_size, 1, max_sen_len, d_w + d_b)
        out = self.cnn_layer1(x)  # (batch_size, num_filter, max_sen_len - window_size + 1, 1)
        out = out.transpose(1, 3)  # (batch_size, 1, max_sen_len - window_size + 1, num_filter)
        out = self.cnn_layer2(out)  # (batch_size, num_filter / 2, 1, 1)
        out = out.view(out.shape[0], -1)  # (batch_size, num_filter/2)
        out = self.linear_layer(out)  # (batch_size, 2)
        return out

    # method to initialize the model weights (in order to improve performance)
    def weights_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


if __name__ == "__main__":
    from data_fetcher.dataFetcher import SenSemEvalDataSet
    train_requirement = {"num_epoch": 20, "batch_size": 32, "lr": 3e-4}
    hyper_parameter = {"d_w": 50, "num_filter": 256, "window_size": 3, "dropout_p": 0.5, "d_b": 10}
    train_data_set = SenSemEvalDataSet("../data/train.txt", "../data/word_embedding/glove.6B.50d.txt", 50, True)
    test_data_set = SenSemEvalDataSet("../data/test.txt", "../data/word_embedding/glove.6B.50d.txt", 50, True, 150)
    model = CnnPosModel(train_data_set, test_data_set, hyper_parameter, train_requirement)

