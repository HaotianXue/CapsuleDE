"""
The implementation of sentence level deep learning model

Author: Haotian Xue
"""
from abc import abstractmethod
from tensor_model_class import TensorModel
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


class SenTensorModel(TensorModel):

    def __init__(self,
                 train_data_set,
                 test_data_set,
                 hyper_parameter,
                 train_requirement,
                 is_gpu=torch.cuda.is_available(),
                 model_save_path='',
                 lr=3e-4):
        super(SenTensorModel, self).__init__(train_data_set,
                                             test_data_set,
                                             hyper_parameter,
                                             train_requirement)
        self.lr = lr
        self.is_gpu = is_gpu
        self.model_save_path = model_save_path
        self.model = None
        self.train_data_loader, self.test_data_loader = None, None

    @abstractmethod
    def build_model(self):
        pass

    def train(self):
        print("-----Start training-----")
        self.model.train(True)
        weight_class = torch.FloatTensor([1, 2])
        if self.is_gpu:
            weight_class = weight_class.cuda()
        criterion = nn.CrossEntropyLoss(weight_class)
        parameters = self.model.parameters()
        optimizer = optim.Adam(parameters, lr=self.lr)
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
            self.test()
            self.save_model()
        print("-----Finish training-----")

    def test(self):
        print("-----Start testing-----")
        self.model.eval()
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
        self.model.train(True)
        if self.is_gpu:
            self.model = self.model.cuda()

    def save_model(self):
        print("-----Start saving trained model-----")
        torch.save(self.model, self.model_save_path)
        print("-----Finish saving trained model-----")

    def load_model(self):
        print("-----Loading trained model-----")
        model = torch.load(self.model_save_path)
        print("-----Finish loading-----")
        return model

    def train_test(self):
        self.train()
        self.save_model()
        self.test()

    def load_test(self):
        self.model = self.load_model()
        self.test()

    def plot(self):
        pass
