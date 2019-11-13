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
                 model_save_path="../trained_model/char_cnn_model.pt"):
        super(CharCnnModel, self).__init__(train_data_set,
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
        d_w, num_filter, dropout_p = self.extract_hyper_parameters()
        print("-----Start building model-----")
        model = CharCnnModelHelper(d_w,
                                   num_filter,
                                   dropout_p,
                                   self.is_gpu)
        print("-----Finish building model-----")
        return model

    def extract_hyper_parameters(self):
        return self.hyper_parameter["d_w"], \
               self.hyper_parameter["num_filter"], \
               self.hyper_parameter["dropout_p"]


class CharCnnModelHelper(nn.Module):

    def __init__(self, d_w, num_filter, dropout_p, is_gpu):
        super(CharCnnModelHelper, self).__init__()
        self.w2v = nn.Embedding(97, d_w)
        self.is_gpu = is_gpu
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, num_filter, kernel_size=(7, d_w), stride=1),
            nn.ReLU()
        )
        self.conv1.apply(self.weights_init)

        self.maxpool1 = nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1))

        self.conv2 = nn.Sequential(
            nn.Conv2d(1, num_filter, kernel_size=(7, num_filter), stride=1),
            nn.ReLU()
        )
        self.conv2.apply(self.weights_init)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1))

        self.conv3 = nn.Sequential(
            nn.Conv2d(1, num_filter, kernel_size=(3, num_filter), stride=1),
            nn.ReLU()
        )
        self.conv3.apply(self.weights_init)

        self.conv4 = nn.Sequential(
            nn.Conv2d(1, num_filter, kernel_size=(3, num_filter), stride=1),
            nn.ReLU()
        )
        self.conv4.apply(self.weights_init)

        self.conv5 = nn.Sequential(
            nn.Conv2d(1, num_filter, kernel_size=(3, num_filter), stride=1),
            nn.ReLU()
        )
        self.conv5.apply(self.weights_init)

        self.conv6 = nn.Sequential(
            nn.Conv2d(1, num_filter, kernel_size=(3, num_filter), stride=1),
            nn.ReLU()
        )
        self.conv6.apply(self.weights_init)

        self.maxpool6 = nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1))

        self.fc1 = nn.Sequential(
            nn.Linear(34 * num_filter, 1024),
            nn.ReLU(),
            nn.Dropout(p=dropout_p)
        )
        self.fc1.apply(self.weights_init)
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=dropout_p)
        )
        self.fc2.apply(self.weights_init)

        self.fc3 = nn.Linear(1024, 2)
        self.fc3.apply(self.weights_init)

    def forward(self, x):
        x = self.w2v(x)  # (batch_size, max_sen_len, d_w)
        x = torch.unsqueeze(x, dim=1)  # (batch_size, 1, max_sen_len, d_w)

        x = self.conv1(x)
        x = x.transpose(1, 3)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = x.transpose(1, 3)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = x.transpose(1, 3)

        x = self.conv4(x)
        x = x.transpose(1, 3)

        x = self.conv5(x)
        x = x.transpose(1, 3)

        x = self.conv6(x)
        x = x.transpose(1, 3)
        x = self.maxpool6(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)

        x = self.fc2(x)

        out = self.fc3(x)
        return out

    # method to initialize the model weights (in order to improve performance)
    def weights_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    # method to initialize the model weights (in order to improve performance)
    def weights_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


if __name__ == "__main__":
    from data_fetcher.dataFetcher import CharSemEvalDataSet
    train_requirement = {"num_epoch": 1, "batch_size": 32, "lr": 3e-4}
    hyper_parameter = {"d_w": 32, "num_filter": 256, "dropout_p": 0.5}
    train_data_set = CharSemEvalDataSet("../data/train.txt", None, 50, True, 1014)
    test_data_set = CharSemEvalDataSet("../data/test.txt", None, 50, True, 1014)
    model = CharCnnModel(train_data_set, test_data_set, hyper_parameter, train_requirement)
