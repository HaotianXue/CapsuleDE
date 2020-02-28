"""
This file is for Fine-tunned Bert classification model
"""


import torch.nn as nn
from transformers import BertModel
import attention


class BertClassifier(nn.Module):

    def __init__(self, freeze_bert=False, num_classes=2, dropout_p=0.2):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        if freeze_bert:
            for parm in self.bert.parameters():
                parm.requires_grad = False
        self.dropout = nn.Dropout(dropout_p, inplace=True)
        self.word_attn = attention.WordAttention(768)
        self.linear_layer = nn.Sequential(
            self.dropout,
            nn.Linear(768, 768 // 4),
            nn.ReLU(),
            nn.Linear(768 // 4, num_classes)
        )
        self.linear_layer.apply(self.weights_init)

    def forward(self, inputs):
        x, attn_msk = inputs
        bert_outputs = self.bert(x, attn_msk)
        encoded_layer = bert_outputs[0]  # (batch, len, dim=768)
        word_attn_out = self.word_attn(encoded_layer)  # (batch, dim=768)
        return self.linear_layer(word_attn_out)

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


