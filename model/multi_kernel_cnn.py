"""
The implementation of Multi-kernel CNN model

Author: Haotian Xue
"""


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sen_tensor_model_class import SenTensorModel


class MultiKernelCnnModel(SenTensorModel):

    def __init__(self,
                 train_data_set,
                 test_data_set,
                 hyper_parameter,
                 train_requirement,
                 is_gpu=torch.cuda.is_available(),
                 model_save_path="../trained_model/multi_kernel_cnn_model.pt",
                 lr=5e-4):
        super(MultiKernelCnnModel, self).__init__(train_data_set,
                                                  test_data_set,
                                                  hyper_parameter,
                                                  train_requirement,
                                                  is_gpu,
                                                  model_save_path,
                                                  lr)
        self.batch_size = self.train_requirement["batch_size"]
        self.train_data_loader = DataLoader(self.train_data_set, self.batch_size, shuffle=True)
        self.test_data_loader = DataLoader(self.test_data_set, self.batch_size, shuffle=False)
        self.model = self.build_model()
        if is_gpu:
            self.model = self.model.cuda()
        self.train_test()
        # self.load_test()

    def build_model(self):
        # now window_sizes is a list [2, 3, 4, ...]
        d_w, num_filter, window_sizes, dropout_p = self.extract_hyper_parameters()
        print("-----Start building model-----")
        model = MultiKernelCnnModelHelper(d_w,
                                          torch.from_numpy(self.test_data_set.word_embedding),
                                          num_filter,
                                          window_sizes,
                                          dropout_p,
                                          self.is_gpu)
        print("-----Finish building model-----")
        return model

    def extract_hyper_parameters(self):
        return self.hyper_parameter["d_w"], \
               self.hyper_parameter["num_filter"], \
               self.hyper_parameter["window_size"], \
               self.hyper_parameter["dropout_p"]


class MultiKernelCnnModelHelper(nn.Module):

    def __init__(self, d_w, word_emb_weight, num_filter, window_sizes, dropout_p, is_gpu, num_classes=2):
        super(MultiKernelCnnModelHelper, self).__init__()
        self.w2v = nn.Embedding.from_pretrained(word_emb_weight, freeze=False)
        self.cnn_layer = CNNLayers(d_w, num_filter, window_sizes, dropout_p, is_gpu)
        self.linear_layer = nn.Sequential(
            nn.Linear(num_filter * len(window_sizes), num_filter),
            nn.ReLU(),
            nn.Linear(num_filter, num_filter // 2),
            nn.ReLU(),
            nn.Linear(num_filter // 2, num_classes)
        )  # out_shape: (batch_size, num_classes)
        self.linear_layer.apply(self.weights_init)

    def forward(self, x):
        x = self.w2v(x)  # (batch_size, max_sen_len, d_w)
        x = torch.unsqueeze(x, dim=1)  # (batch_size, 1, max_sen_len, d_w)
        out = self.cnn_layer(x)  # (batch_size, num_filter * len(window_sizes))
        m = nn.Tanh()
        out = m(out)
        out = self.linear_layer(out)  # (batch_size, 2)
        return out

    # method to initialize the model weights (in order to improve performance)
    def weights_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class CNNLayers(nn.Module):

    def __init__(self, d_w, num_filter, window_sizes, dropout_p, is_gpu):
        super(CNNLayers, self).__init__()
        self.is_gpu = is_gpu
        self.cnn_layers = []
        for window_size in window_sizes:
            cnn_layer = nn.Sequential(
                nn.Conv2d(in_channels=1,
                          out_channels=num_filter,
                          kernel_size=(window_size, d_w),
                          stride=(1, 1),
                          padding=(0, 0)),  # (batch, num_filter, max_sen_len - window_size + 1, 1)
                nn.MaxPool2d(kernel_size=(150 - window_size + 1, 1),
                             stride=(1, 1)),  # (batch, num_filter, 1, 1)
                nn.Dropout(dropout_p),
            )
            cnn_layer.apply(self.weights_init)
            self.cnn_layers.append(cnn_layer)

    def forward(self, x):
        out_list = []
        for cnn_layer in self.cnn_layers:
            if self.is_gpu:
                cnn_layer = cnn_layer.cuda()
            out = cnn_layer(x)  # (batch_size, num_filter, 1, 1)
            out = out.view(out.shape[0], -1)  # (batch_size, num_filter)
            out_list.append(out)
        out = torch.cat(out_list, dim=1)  # (batch_size, num_filter * len(out_list))
        return out

    # method to initialize the model weights (in order to improve performance)
    def weights_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


if __name__ == "__main__":
    from data_fetcher.dataFetcher import SenSemEvalDataSet
    train_requirement = {"num_epoch": 1, "batch_size": 32}
    hyper_parameter = {"d_w": 50, "num_filter": 100, "window_size": [2, 3, 4], "dropout_p": 0.4}
    train_data_set = SenSemEvalDataSet("../data/train.txt", "../data/word_embedding/glove.6B.50d.txt", 50, True)
    test_data_set = SenSemEvalDataSet("../data/test.txt", "../data/word_embedding/glove.6B.50d.txt", 50, True, 150)
    model = MultiKernelCnnModel(train_data_set, test_data_set, hyper_parameter, train_requirement)
