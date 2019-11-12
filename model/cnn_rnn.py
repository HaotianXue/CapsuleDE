"""
The implementation of cnn + rnn model

Author: Haotian Xue
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sen_tensor_model_class import SenTensorModel


class CnnRnnModel(SenTensorModel):

    def __init__(self,
                 train_data_set,
                 test_data_set,
                 hyper_parameter,
                 train_requirement,
                 is_gpu=torch.cuda.is_available(),
                 model_save_path="../trained_model/cnn_model.pt"):
        super(CnnRnnModel, self).__init__(train_data_set,
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
        d_w, num_filter, window_size, hidden_dim, num_layers, dropout_p = self.extract_hyper_parameters()
        print("-----Start building model-----")
        model = CnnRnnModelHelper(d_w=d_w,
                                  word_emb_weight=torch.from_numpy(self.test_data_set.word_embedding),
                                  num_filter=num_filter,
                                  window_size=window_size,
                                  num_layers=num_layers,
                                  hidden_dim=hidden_dim,
                                  dropout_p=dropout_p)
        print("-----Finish building model-----")
        return model

    def extract_hyper_parameters(self):
        return self.hyper_parameter["d_w"], \
               self.hyper_parameter["num_filter"], \
               self.hyper_parameter["window_size"], \
               self.hyper_parameter["hidden_dim"], \
               self.hyper_parameter["num_layers"], \
               self.hyper_parameter["dropout_p"]


class CnnRnnModelHelper(nn.Module):

    def __init__(self, d_w, word_emb_weight, num_filter, window_size, num_layers, hidden_dim, dropout_p, num_classes=2):
        super(CnnRnnModelHelper, self).__init__()
        self.w2v = nn.Embedding.from_pretrained(word_emb_weight, freeze=False)
        self.cnn_layer = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=num_filter,
                      kernel_size=(window_size, d_w),
                      stride=(1, 1),
                      padding=(1, 0)),  # out_shape: (batch_size, num_filter, max_sen_len, 1)
            nn.MaxPool2d(kernel_size=(window_size, 1),
                         stride=(1, 1)),  # out_shape: (batch_size, num_filter, max_sen_len-window_size+1, 1)
            nn.ReLU()
        )
        self.cnn_layer.apply(self.weights_init)
        self.rnn_layer = nn.GRU(input_size=num_filter,
                                hidden_size=hidden_dim,
                                num_layers=num_layers,
                                bias=True,
                                batch_first=True,
                                dropout=dropout_p,
                                bidirectional=True
        )  # out_shape: (batch_size, max_sen_len-window_size+1, hidden_dim*2)
        self.rnn_layer.apply(self.weights_init)
        self.linear_layer = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )  # out_shape: (batch_size, num_classes)
        self.linear_layer.apply(self.weights_init)

    def forward(self, x):
        x = self.w2v(x)  # (batch_size, max_sen_len, d_w)
        x = torch.unsqueeze(x, dim=1)  # (batch_size, 1, max_sen_len, d_w)
        out = self.cnn_layer(x)  # (batch_size, num_filter, max_sen_len-window_size+1, 1)
        out = torch.squeeze(out, dim=3)  # (batch_size, num_filter, max_sen_len-window_size+1)
        out = out.view(out.shape[0], out.shape[-1], -1)  # (batch_size, max_sen_len-window_size+1, num_filter)
        out, _ = self.rnn_layer(out)  # (batch_size, max_sen_len-window_size+1, hidden_dim*2)
        out = out[:, -1, :]  # (batch_size, 1, hidden_dim*2)  # TODO: try word attention
        out = torch.squeeze(out, dim=1)  # (batch_size, hidden_dim*2)
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

    train_requirement = {"num_epoch": 30, "batch_size": 32, "lr": 3e-4}
    hyper_parameter = {"d_w": 50, "num_filter": 128, "window_size": 3, "num_layers": 2, "hidden_dim": 128, "dropout_p": 0.2}
    train_data_set = SenSemEvalDataSet("../data/train.txt", "../data/word_embedding/glove.6B.50d.txt", 50, True)
    test_data_set = SenSemEvalDataSet("../data/test.txt", "../data/word_embedding/glove.6B.50d.txt", 50, True, 150)
    model = CnnRnnModel(train_data_set, test_data_set, hyper_parameter, train_requirement)