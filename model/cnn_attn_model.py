"""
The implementation of cnn + word attention model

Author: Haotian Xue
"""


import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import attention
from tensor_model_class import TensorModel


class CnnAttnModel(TensorModel):

    def __init__(self,
                 train_data_set,
                 test_data_set,
                 hyper_parameter,
                 train_requirement,
                 is_gpu=torch.cuda.is_available(),
                 model_save_path="../trained_model/cnn_model.pt"):
        super(CnnAttnModel, self).__init__(train_data_set, test_data_set, hyper_parameter)
        self.train_requirement = train_requirement  # include: batch_size, num_epoch, lr_rate, etc
        self.batch_size = self.train_requirement["batch_size"]
        self.is_gpu = is_gpu
        self.model_save_path = model_save_path
        self.train_data_loader = DataLoader(self.train_data_set, self.batch_size, shuffle=True)
        self.test_data_loader = DataLoader(self.test_data_set, self.batch_size, shuffle=False)
        self.model = self.build_model()
        if is_gpu:
            self.model = self.model.cuda()
        self.train_test()
        # self.load_test()

    def train_test(self):
        self.train()
        self.save_model()
        self.test()

    def build_model(self):
        d_w, num_filter, window_size = self.extract_hyper_parameters()
        print("-----Start building model-----")
        model = CnnAttnModelHelper(d_w,
                                   torch.from_numpy(self.test_data_set.word_embedding),
                                   num_filter,
                                   window_size)
        print("-----Finish building model-----")
        return model

    def extract_hyper_parameters(self):
        return self.hyper_parameter["d_w"], \
               self.hyper_parameter["num_filter"], \
               self.hyper_parameter["window_size"]

    def train(self):
        print("-----Start training-----")
        criterion = nn.CrossEntropyLoss()
        parameters = self.model.parameters()
        optimizer = optim.Adam(parameters, lr=3e-4)
        num_epoch = self.train_requirement["num_epoch"]
        for i in range(num_epoch):
            running_loss = 0.0
            for j, (x, y) in enumerate(self.train_data_loader):
                if self.is_gpu:
                    x, y = x.cuda(), y.cuda()
                optimizer.zero_grad()
                outputs = self.model(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                # print statistics
                running_loss += loss.item()
                print('%d epoch: %d Done, loss = %f' % (i, j, running_loss))
                running_loss = 0.0
            self.save_model()
        print("-----Finish training-----")

    def test(self):
        print("-----Start testing-----")
        if self.is_gpu:
            self.model = self.model.cpu()  # for testing, we just need cpu

        correct = 0
        total = 0

        # matrix used for computing f1 score
        y_true = None
        y_pred = None

        with torch.no_grad():
            index = 0
            for d in self.test_data_loader:
                inputs, labels = d
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)  # predicted shape: [batch_size, 1]
                total += labels.size(0)  # labels shape: [batch_size, 1]
                correct += (predicted == labels).sum().item()
                if index == 0:
                    y_true = labels
                    y_pred = predicted
                else:
                    y_true = torch.cat((y_true, labels), 0)
                    y_pred = torch.cat((y_pred, predicted), 0)
                index += 1
        print('F1 score: ', f1_score(y_true.numpy(), y_pred.numpy()))
        print('Precision score: ', precision_score(y_true.numpy(), y_pred.numpy()))
        print('Recall score: ', recall_score(y_true.numpy(), y_pred.numpy()))
        print('Accuracy score: ', accuracy_score(y_true.numpy(), y_pred.numpy()))
        print("-----Finish testing-----")

    def save_model(self):
        print("-----Start saving trained model-----")
        torch.save(self.model, self.model_save_path)
        print("-----Finish saving trained model-----")

    def load_model(self):
        print("-----Loading trained model-----")
        model = torch.load(self.model_save_path)
        print("-----Finish loading-----")
        return model

    def plot(self):
        pass

    def load_test(self):
        self.model = self.load_model()
        self.test()


class CnnAttnModelHelper(nn.Module):
    def __init__(self, d_w, word_emb_weight, num_filter, window_size, num_classes=2):
        super(CnnAttnModelHelper, self).__init__()
        self.num_filter = num_filter
        self.w2v = nn.Embedding.from_pretrained(word_emb_weight, freeze=False)
        self.cnn_layer = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=num_filter,
                      kernel_size=(window_size, d_w),
                      stride=(1, 1),
                      padding=(1, 0)),  # out_shape: (batch_size, num_filter, max_sen_len, 1)
            nn.Tanh()
        )  # out_shape: (batch_size, num_filter, max_sen_len, 1)
        self.cnn_layer.apply(self.weights_init)
        self.word_attn = attention.WordAttention(num_filter)  # out_shape: (batch_size, num_filter)
        self.linear_layer = nn.Sequential(
            nn.Linear(num_filter, num_filter // 4),
            nn.Tanh(),
            nn.Linear(num_filter // 4, num_classes)
        )  # out_shape: (batch_size, num_classes)
        self.linear_layer.apply(self.weights_init)

    def forward(self, x):
        x = self.w2v(x)  # (batch_size, max_sen_len, d_w)
        x = torch.unsqueeze(x, dim=1)  # (batch_size, 1, max_sen_len, d_w)
        out = self.cnn_layer(x)  # (batch_size, num_filter, max_sen_len, 1)
        out = out.view(out.shape[0], -1, self.num_filter)  # (batch_size, max_sen_len, num_filter)
        out = self.word_attn(out)  # (batch_size, num_filter)
        out = F.tanh(out)
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
    train_requirement = {"num_epoch": 30, "batch_size": 32}
    hyper_parameter = {"d_w": 50, "num_filter": 256, "window_size": 3}
    train_data_set = SenSemEvalDataSet("../data/train.txt", "../data/word_embedding/glove.6B.50d.txt", 50, True)
    test_data_set = SenSemEvalDataSet("../data/test.txt", "../data/word_embedding/glove.6B.50d.txt", 50, True, 150, is_gpu=False)
    model = CnnAttnModel(train_data_set, test_data_set, hyper_parameter, train_requirement)

