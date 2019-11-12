"""
The implementation of character level cnn model

Author: Haotian Xue
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sen_tensor_model_class import SenTensorModel


class CharCnnModel(SenTensorModel):

    def __init__(self,
                 train_data_set,
                 test_data_set,
                 hyper_parameter,
                 train_requirement,
                 is_gpu=torch.cuda.is_available(),
                 model_save_path="../trained_model/char_cnn_model.pt",
                 lr=5e-4):
        super(CharCnnModel, self).__init__(train_data_set,
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
        d_w, num_filter, window_sizes, dropout_p = self.extract_hyper_parameters()
        print("-----Start building model-----")
        model = CharCnnModelHelper(d_w,
                                   num_filter,
                                   window_sizes,
                                   dropout_p)
        print("-----Finish building model-----")
        return model

    def extract_hyper_parameters(self):
        return self.hyper_parameter["d_w"], \
               self.hyper_parameter["num_filter"], \
               self.hyper_parameter["window_size"], \
               self.hyper_parameter["dropout_p"]


class CharCnnModelHelper(nn.Module):

    def __init__(self, d_w, num_filter, window_sizes, dropout_p, num_classes=2):
        super(CharCnnModelHelper, self).__init__()
        self.w2v = nn.Embedding(97, d_w)  # char embedding
        self.cnn_layers = []
        for window_size in window_sizes:
            cnn_layer = nn.Sequential(
                nn.Conv2d(in_channels=1,
                          out_channels=num_filter,
                          kernel_size=(window_size, d_w),
                          stride=(1, 1),
                          padding=(0, 0)),  # (batch, num_filter, max_sen_len - window_size + 1, 1)
                nn.MaxPool2d(kernel_size=(842 - window_size + 1, 1),
                             stride=(1, 1)),  # (batch, num_filter, 1, 1)
                nn.Dropout(dropout_p),
            )
            cnn_layer.apply(self.weights_init)
            self.cnn_layers.append(cnn_layer)
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
        out_list = []
        for cnn_layer in self.cnn_layers:
            out = cnn_layer(x)  # (batch_size, num_filter, 1, 1)
            out = out.view(out.shape[0], -1)  # (batch_size, num_filter)
            out_list.append(out)
        out = torch.cat(out_list, dim=1)  # (batch_size, num_filter * len(out_list))
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


if __name__ == "__main__":
    from data_fetcher.dataFetcher import CharSemEvalDataSet
    train_requirement = {"num_epoch": 1, "batch_size": 32}
    hyper_parameter = {"d_w": 32, "num_filter": 64, "window_size": [1, 2, 3, 4, 5, 6, 7, 8], "dropout_p": 0.4}
    train_data_set = CharSemEvalDataSet("../data/train.txt", None, 50, True)
    test_data_set = CharSemEvalDataSet("../data/test.txt", None, 50, True, 842)
    model = CharCnnModel(train_data_set, test_data_set, hyper_parameter, train_requirement)
