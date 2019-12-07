"""
The implementation of capsule + lstm model

Author: Haotian Xue
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
from utils import attention
from utils.sen_tensor_model_class import SenTensorModel
from data_fetcher.torchTextDataFetcher import TorchTextSemEvalDataSet

T_epsilon = 1e-7
Routings = 5
Num_capsule = 10
Dim_capsule = 16
gru_len = 128
BATCH_SIZE = 32


class CapRNNModel(SenTensorModel):
    def __init__(self,
                 dataset,
                 hyper_parameter,
                 train_requirement,
                 is_gpu=torch.cuda.is_available(),
                 model_save_path="../trained_model/capsule_rnn_model.pt"):
        super(CapRNNModel, self).__init__(dataset,
                                          hyper_parameter,
                                          train_requirement,
                                          is_gpu,
                                          model_save_path)
        self.model = self.build_model()
        if is_gpu:
            self.model = self.model.cuda()
        self.train_test()

    def build_model(self):
        d_w, hidden_dim, dropout_prob = self.extract_hyper_parameters()
        print("-----Start building model-----")
        model = CapRNNModelHelper(d_w,
                                  self.dataset.TEXT.vocab.vectors,
                                  hidden_dim,
                                  dropout_p=dropout_prob)
        print("-----Finish building model-----")
        return model

    def extract_hyper_parameters(self):
        return self.hyper_parameter["d_w"], \
               self.hyper_parameter["hidden_dim"], \
               self.hyper_parameter["dropout_prob"]


class CapsuleLayer(nn.Module):

    def __init__(self,
                 input_dim_capsule=gru_len * 2,
                 num_capsule=Num_capsule,
                 dim_capsule=Dim_capsule,
                 routings=Routings,
                 kernel_size=(9, 1),
                 share_weights=True,
                 activation='default'):
        super(CapsuleLayer, self).__init__()
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size  # not used so far
        self.share_weights = share_weights
        if activation == 'default':
            self.activation = self.squash
        else:
            self.activation = nn.ReLU(inplace=True)

        if self.share_weights:
            self.W = nn.Parameter(
                nn.init.xavier_normal_(torch.empty(1, input_dim_capsule, self.num_capsule * self.dim_capsule)), requires_grad=True)
        else:
            self.W = nn.Parameter(
                torch.randn(BATCH_SIZE, input_dim_capsule, self.num_capsule * self.dim_capsule), requires_grad=True)

    def forward(self, x):
        if self.share_weights:
            u_hat_vecs = torch.matmul(x, self.W)  # (batch, sen, num_capsule * dim_capsule)
        else:
            print('add later')

        batch_size = x.size(0)
        input_num_capsule = x.size(1)  # input_num_capsule = sen_len
        u_hat_vecs = u_hat_vecs.view((batch_size, input_num_capsule,
                                      self.num_capsule, self.dim_capsule))
        u_hat_vecs = u_hat_vecs.permute(0, 2, 1, 3)  # (batch_size,num_capsule,input_num_capsule,dim_capsule)
        b = torch.zeros_like(u_hat_vecs[:, :, :, 0])  # (batch_size,num_capsule,input_num_capsule)

        for i in range(self.routings):
            b = b.permute(0, 2, 1)  # (batch, input_num_capsule, num_capsule)
            c = F.softmax(b, dim=2) # (batch, input_num_capsule, num_capsule)
            c = c.permute(0, 2, 1)  # (batch, num_capsule, input_num_capsule)
            b = b.permute(0, 2, 1)  # (batch, num_capsule, input_num_capsule)
            outputs = self.activation(torch.einsum('bij,bijk->bik', (c, u_hat_vecs)))
            # outputs shape (batch_size, num_capsule, dim_capsule)
            if i < self.routings - 1:
                # cannot use += as += will cause the problem of inplace operation
                b = b + torch.einsum('bik,bijk->bij', (outputs, u_hat_vecs))
        return outputs  # (batch_size, num_capsule, dim_capsule)

    def squash(self, x, axis=-1):
        s_squared_norm = (x ** 2).sum(axis, keepdim=True)
        scale = torch.sqrt(s_squared_norm + T_epsilon)
        return x / scale


class CapRNNModelHelper(nn.Module):

    def __init__(self, d_w, word_emb_weight, hidden_dim=gru_len, num_classes=2, dropout_p=0.2):
        super(CapRNNModelHelper, self).__init__()
        self.hidden_dim = hidden_dim
        self.w2v = nn.Embedding.from_pretrained(word_emb_weight, freeze=True)
        self.dropout = nn.Dropout(dropout_p, inplace=True)
        self.gru = nn.GRU(input_size=d_w,
                          hidden_size=hidden_dim,
                          num_layers=1,
                          bias=True,
                          batch_first=True,
                          dropout=dropout_p,
                          bidirectional=True)  # shape: (batch_size, sen_len, hidden_size*2)
        self.gru.apply(self.weights_init)
        self.cap_layer = CapsuleLayer()
        self.linear_layer = nn.Sequential(
            self.dropout,
            nn.Linear(Num_capsule * Dim_capsule, num_classes),
        )
        self.linear_layer.apply(self.weights_init)

    def forward(self, x):
        x = self.w2v(x)
        x = self.dropout(x)
        h_gru, _ = self.gru(x)  # (batch, sen, hidden_dim*2)
        # print("h_gru: ", h_gru.shape)  # (batch, sen, gru_len*2)
        cap_out = self.cap_layer(h_gru)
        # print("cap out: ", cap_out.shape)  # (batch, num_capsule=10, dim_capsule=16)
        cap_out = cap_out.view(cap_out.shape[0], -1)  # (batch, num_capsule * dim_capsule)
        out = self.linear_layer(cap_out)
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
    print(torch.cuda.is_available())
    train_requirement = {"num_epoch": 1, "batch_size": 32, "lr": 3e-4}
    hyper_parameter = {"d_w": 50, "hidden_dim": gru_len, "dropout_prob": 0.2}
    data_set = TorchTextSemEvalDataSet("../data/train.csv",
                                       "../data/test.csv",
                                       "../data/word_embedding/glove.6B.50d.txt",
                                       train_requirement["batch_size"],
                                       True,
                                       150,
                                       torch.cuda.is_available())
    model = CapRNNModel(data_set, hyper_parameter, train_requirement)
