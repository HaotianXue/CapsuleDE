"""
The implementation of LSTM + word attention model

Author: Haotian Xue
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import attention
from tensor_model_class import TensorModel


class RnnAttnModel(TensorModel):

    def __init__(self,
                 train_data_set,
                 test_data_set,
                 hyper_parameter,
                 train_requirement,
                 is_gpu=torch.cuda.is_available(),
                 model_save_path="../trained_model/rnn_attn_model.pt"):
        super(RnnAttnModel, self).__init__(train_data_set, test_data_set, hyper_parameter)
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
        d_w, hidden_dim, num_layers, dropout_prob = self.extract_hyper_parameters()
        print("-----Start building model-----")
        model = RnnAttnModelHelper(d_w,
                                   torch.from_numpy(self.test_data_set.word_embedding),
                                   hidden_dim,
                                   num_layers=num_layers,
                                   dropout_p=dropout_prob)
        print("-----Finish building model-----")
        return model

    def extract_hyper_parameters(self):
        return self.hyper_parameter["d_w"], \
               self.hyper_parameter["hidden_dim"], \
               self.hyper_parameter["num_layers"], \
               self.hyper_parameter["dropout_prob"]

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


class RnnAttnModelHelper(nn.Module):

    def __init__(self, d_w, word_emb_weight, hidden_dim, num_layers, num_classes=2, dropout_p=0.2):
        super(RnnAttnModelHelper, self).__init__()
        self.hidden_dim = hidden_dim
        self.w2v = nn.Embedding.from_pretrained(word_emb_weight, freeze=False)
        self.rnn_layer = nn.LSTM(input_size=d_w,
                                 hidden_size=hidden_dim,
                                 num_layers=num_layers,
                                 bias=True,
                                 batch_first=True,
                                 dropout=dropout_p,
                                 bidirectional=True)  # shape: (batch_size, sen_len, hidden_size*2)
        self.rnn_layer.apply(self.weights_init)
        self.word_attn = attention.WordAttention(hidden_dim * 2)  # shape: (batch_size, hidden_dim*2)
        self.linear_layer = nn.Sequential(  # int_shape: (batch_size, hidden_size*2)
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, num_classes)  # out_shape: (batch_size, num_classes)
        )
        self.linear_layer.apply(self.weights_init)

    def forward(self, x):
        x = self.w2v(x)
        out, _ = self.rnn_layer(x)
        m = nn.Tanh()
        out = m(out)
        out = self.word_attn(out)
        out = self.linear_layer(out)
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
    train_requirement = {"num_epoch": 10, "batch_size": 32}
    hyper_parameter = {"d_w": 50, "hidden_dim": 256, "num_layers": 2, "dropout_prob": 0.1}
    train_data_set = SenSemEvalDataSet("../data/train.txt", "../data/word_embedding/glove.6B.50d.txt", 50, True)
    test_data_set = SenSemEvalDataSet("../data/test.txt", "../data/word_embedding/glove.6B.50d.txt", 50, True)
    model = RnnAttnModel(train_data_set, test_data_set, hyper_parameter, train_requirement)

