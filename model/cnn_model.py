"""
The implementation of cnn + word attention model

Author: Haotian Xue
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sen_tensor_model_class import SenTensorModel


class CnnModel(SenTensorModel):

    def __init__(self,
                 train_data_set,
                 test_data_set,
                 hyper_parameter,
                 train_requirement,
                 is_gpu=torch.cuda.is_available(),
                 model_save_path="../trained_model/cnn_model.pt"):
        super(CnnModel, self).__init__(train_data_set,
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
        d_w, num_filter, window_size = self.extract_hyper_parameters()
        print("-----Start building model-----")
        model = CnnModelHelper(d_w,
                               torch.from_numpy(self.test_data_set.word_embedding),
                               num_filter,
                               window_size)
        print("-----Finish building model-----")
        return model

    def extract_hyper_parameters(self):
        return self.hyper_parameter["d_w"], \
               self.hyper_parameter["num_filter"], \
               self.hyper_parameter["window_size"]


class CnnModelHelper(nn.Module):
    def __init__(self, d_w, word_emb_weight, num_filter, window_size, num_classes=2):
        super(CnnModelHelper, self).__init__()
        self.w2v = nn.Embedding.from_pretrained(word_emb_weight, freeze=False)
        self.cnn_layer = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=num_filter,
                      kernel_size=(window_size, d_w),
                      stride=(1, 1),
                      padding=(1, 0)),  # out_shape: (batch_size, num_filter, max_sen_len, 1)
            nn.MaxPool2d(kernel_size=(150, 1),
                         stride=(1, 1)),  # out_shape: (batch_size, num_filter, 1, 1)
            nn.ReLU()
        )  # out_shape: (batch_size, num_filter, 1, 1)
        self.cnn_layer.apply(self.weights_init)
        self.linear_layer = nn.Sequential(
            nn.Linear(num_filter, num_filter // 2),
            nn.ReLU(),
            nn.Linear(num_filter // 2, num_classes)
        )  # out_shape: (batch_size, num_classes)
        self.linear_layer.apply(self.weights_init)

    def forward(self, x):
        x = self.w2v(x)  # (batch_size, max_sen_len, d_w)
        x = torch.unsqueeze(x, dim=1)  # (batch_size, 1, max_sen_len, d_w)
        out = self.cnn_layer(x)  # (batch_size, num_filter, 1, 1)
        out = out.view(out.shape[0], -1)  # (batch_size, num_filter)
        m = nn.Tanh()
        out = m(out)
        out = self.linear_layer(out)  # (batch_size, 2)
        return out

    # method to initialize the model weights (in order to improve performance)
    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        if isinstance(m, nn.GRU) or isinstance(m, nn.LSTM) or isinstance(m, nn.RNN):
            ih = (param.data for name, param in m.named_parameters() if 'weight_ih' in name)
            hh = (param.data for name, param in m.named_parameters() if 'weight_hh' in name)
            b = (param.data for name, param in m.named_parameters() if 'bias' in name)
            # nn.init.uniform(m.embed.weight.data, a=-0.5, b=0.5)
            for t in ih:
                nn.init.xavier_uniform(t)
            for t in hh:
                nn.init.orthogonal(t)
            for t in b:
                nn.init.constant(t, 0)


if __name__ == "__main__":
    from data_fetcher.dataFetcher import SenSemEvalDataSet
    train_requirement = {"num_epoch": 30, "batch_size": 4}
    hyper_parameter = {"d_w": 50, "num_filter": 256, "window_size": 3}
    train_data_set = SenSemEvalDataSet("../data/train.txt", "../data/word_embedding/glove.6B.50d.txt", 50, True)
    test_data_set = SenSemEvalDataSet("../data/test.txt", "../data/word_embedding/glove.6B.50d.txt", 50, True, 150)
    model = CnnModel(train_data_set, test_data_set, hyper_parameter, train_requirement)

