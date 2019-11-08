"""
THe implementation of multi-head self-attention model

Author: Haotian Xue
"""

import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from tensor_model_class import TensorModel
from utils import layers, attention


class MultiHeadSelfAttnModel(TensorModel):
    def __init__(self,
                 train_data_set,
                 test_data_set,
                 hyper_parameter,
                 train_requirement,
                 is_gpu=torch.cuda.is_available(),
                 model_save_path="../trained_model/self_attn_model.pt"):
        super(MultiHeadSelfAttnModel, self).__init__(train_data_set, test_data_set, hyper_parameter)
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

    def train_test(self):
        self.train()
        self.save_model()
        self.test()

    def build_model(self):
        d_w, hidden_dim, num_layers, num_heads = self.extract_hyper_parameters()
        print("-----Start building model-----")
        model = MultiHeadSelfAttnModelHelper(d_w=d_w,
                                             word_emb_weight=torch.from_numpy(self.test_data_set.word_embedding),
                                             hidden_dim=hidden_dim,
                                             num_layers=num_layers,
                                             num_heads=num_heads)
        print("-----Finish building model-----")
        return model

    def extract_hyper_parameters(self):
        return self.hyper_parameter["d_w"], \
               self.hyper_parameter["hidden_dim"], \
               self.hyper_parameter["num_layers"], \
               self.hyper_parameter["num_heads"]

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


class MultiHeadSelfAttnModelHelper(nn.Module):
    def __init__(self, d_w, hidden_dim, word_emb_weight, num_layers=4,
                 num_heads=5, dropout=0.1, num_classes=2):
        super(MultiHeadSelfAttnModelHelper, self).__init__()
        self.w2v = nn.Embedding.from_pretrained(word_emb_weight, freeze=False)
        c = copy.deepcopy
        d_model = d_w
        self_attn = attention.MultiHeadAttention(h=num_heads, d_model=d_model, dropout=dropout)
        ff = layers.PositionwiseFeedForward(d_model=d_model, d_ff=hidden_dim, dropout=dropout)
        word_attn = attention.WordAttention(d_model)  # (batch, sen, d_model) => (batch, d_model)
        self.model = nn.Sequential(
            layers.Encoder(layers.EncoderLayer(d_model, c(self_attn), c(ff), dropout), num_layers),
            word_attn,
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_classes)
        )
        for p in self.model.parameters():
            if p.dim() > 1:  # dim: 维度数
                nn.init.xavier_uniform_(p)

    def forward(self, x):  # x: (batch, max_sen_len)
        x = self.w2v(x)   # (batch_size, max_sen_len, d_w)
        output = self.model(x)  # (batch_size, num_classes)
        return output


if __name__ == "__main__":
    from data_fetcher.dataFetcher import SenSemEvalDataSet
    train_requirement = {"num_epoch": 20, "batch_size": 32}
    hyper_parameter = {"d_w": 50, "hidden_dim": 256, "num_layers": 4, "num_heads": 5}
    train_data_set = SenSemEvalDataSet("../data/train.txt", "../data/word_embedding/glove.6B.50d.txt", 50, True)
    test_data_set = SenSemEvalDataSet("../data/test.txt", "../data/word_embedding/glove.6B.50d.txt", 50, True)
    model = MultiHeadSelfAttnModel(train_data_set, test_data_set, hyper_parameter, train_requirement)
