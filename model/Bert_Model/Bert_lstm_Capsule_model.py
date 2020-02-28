"""
The implementation of Fine-tunned + Bert + LSTM + Capsule model

Author: Haotian Xue
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import BertTokenizer, BertModel

T_epsilon = 1e-7
Routings = 5
Num_capsule = 10
Dim_capsule = 16
gru_len = 128
BATCH_SIZE = 4


# TODO: 3. OOD Structure programming to support whether freeze BERT or not
class BertLSTMCapsuleNet(nn.Module):

    def __init__(self, freeze_bert=False, num_classes=2, dropout_p=0.2):
        super(BertLSTMCapsuleNet, self).__init__()
        self.freeze_bert = freeze_bert
        self.bert = BertModel.from_pretrained('bert-base-cased')
        if self.freeze_bert:
            for parm in self.bert.parameters():
                parm.requires_grad = False
        self.dropout = nn.Dropout(dropout_p, inplace=True)
        self.cap_layer = CapsuleLayer()
        self.gru = nn.GRU(input_size=768,
                          hidden_size=gru_len,
                          num_layers=1,
                          bias=True,
                          batch_first=True,
                          dropout=dropout_p,
                          bidirectional=True)  # shape: (batch_size, sen_len, hidden_size*2)
        self.gru.apply(self.weights_init)
        self.linear_layer = nn.Sequential(
            self.dropout,
            nn.Linear(Num_capsule * Dim_capsule, num_classes),
        )
        self.linear_layer.apply(self.weights_init)

    def forward(self, inputs):
        x, attn_msk = inputs
        bert_outputs = self.bert(x, attn_msk)
        encoded_layer = bert_outputs[0]  # (batch, len, dim=768)
        h_gru, _ = self.gru(encoded_layer)  # (batch, sen, hidden_dim*2)
        cap_out = self.cap_layer(h_gru)
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
                nn.init.xavier_normal_(torch.empty(1, input_dim_capsule, self.num_capsule * self.dim_capsule)),
                requires_grad=True)
        else:
            self.W = nn.Parameter(
                torch.randn(BATCH_SIZE, input_dim_capsule, self.num_capsule * self.dim_capsule),
                requires_grad=True)

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
            c = F.softmax(b, dim=2)  # (batch, input_num_capsule, num_capsule)
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
