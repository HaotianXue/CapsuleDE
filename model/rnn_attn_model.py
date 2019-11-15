"""
The implementation of LSTM + word attention model

Author: Haotian Xue
"""

import torch
import torch.nn as nn
import attention
from sen_tensor_model_class import SenTensorModel


class RnnAttnModel(SenTensorModel):

    def __init__(self,
                 train_data_set,
                 test_data_set,
                 hyper_parameter,
                 train_requirement,
                 is_gpu=torch.cuda.is_available(),
                 model_save_path="../trained_model/rnn_attn_model.pt"):
        super(RnnAttnModel, self).__init__(train_data_set,
                                           test_data_set,
                                           hyper_parameter,
                                           train_requirement,
                                           is_gpu,
                                           model_save_path)
        self.model = self.build_model()
        if is_gpu:
            self.model = self.model.cuda()
        self.train_test()

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


class RnnAttnModelHelper(nn.Module):

    def __init__(self, d_w, word_emb_weight, hidden_dim, num_layers, num_classes=2, dropout_p=0.2):
        super(RnnAttnModelHelper, self).__init__()
        self.hidden_dim = hidden_dim
        self.w2v = nn.Embedding.from_pretrained(word_emb_weight, freeze=True)
        self.dropout = nn.Dropout(dropout_p)
        self.lstm = nn.LSTM(input_size=d_w,
                            hidden_size=hidden_dim,
                            num_layers=1,
                            bias=True,
                            batch_first=True,
                            dropout=dropout_p,
                            bidirectional=True)  # shape: (batch_size, sen_len, hidden_size*2)
        self.lstm.apply(self.weights_init)
        self.lstm_attn = attention.WordAttention(hidden_dim * 2)  # shape: (batch_size, hidden_dim*2)
        for p in self.lstm_attn.parameters():
            if p.dim() > 1:  # dim: 维度数
                nn.init.xavier_uniform_(p)
        self.gru = nn.GRU(input_size=hidden_dim * 2,
                          hidden_size=hidden_dim,
                          num_layers=1,
                          bias=True,
                          batch_first=True,
                          dropout=dropout_p,
                          bidirectional=True)  # shape: (batch_size, sen_len, hidden_size*2)
        self.gru.apply(self.weights_init)
        self.gru_attn = attention.WordAttention(hidden_dim * 2)  # (batch_size, hidden_dim*2)
        for p in self.gru_attn.parameters():
            if p.dim() > 1:  # dim: 维度数
                nn.init.xavier_uniform_(p)
        self.linear_layer1 = nn.Sequential(  # int_shape: (batch_size, hidden_size*4)
            nn.Linear(hidden_dim * 8, 16),
            nn.Dropout(dropout_p),
            # nn.BatchNorm1d(hidden_dim * 4),
            nn.ReLU()
        )  # (batch, hidden_size * 4)
        self.linear_layer1.apply(self.weights_init)
        self.linear_layer2 = nn.Sequential(
            nn.Linear(16, num_classes)
        )  # (batch, 2)

    def forward(self, x):
        x = self.w2v(x)
        x = self.dropout(x)

        h_lstm, _ = self.lstm(x)  # (batch, sen, hidden_dim*2)
        h_gru, _ = self.gru(x)  # (batch, sen, hidden_dim*2)

        h_lstm_attn = self.lstm_attn(h_lstm)  # (batch, hidden_dim*2)
        h_gru_attn = self.gru_attn(h_gru)  # (batch, hidden_dim*2)

        # global average pooling
        avg_pool = torch.mean(h_lstm, dim=1)  # (batch, hidden_dim*2)
        # # global max pooling
        max_pool = torch.mean(h_gru, dim=1)  # (batch, hidden_dim*2)

        out = torch.cat([h_lstm_attn, h_gru_attn, avg_pool, max_pool], dim=1)  # (batch, hidden_dim*8)
        out = self.linear_layer1(self.dropout(out))
        out = self.linear_layer2(self.dropout(out))
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
    print(torch.cuda.is_available())
    train_requirement = {"num_epoch": 30, "batch_size": 32, "lr": 3e-4}
    hyper_parameter = {"d_w": 50, "hidden_dim": 40, "num_layers": 2, "dropout_prob": 0.5}
    train_data_set = SenSemEvalDataSet("../data/train.txt", "../data/word_embedding/glove.6B.50d.txt", 50, True)
    test_data_set = SenSemEvalDataSet("../data/test.txt", "../data/word_embedding/glove.6B.50d.txt", 50, True, is_gpu=False)
    model = RnnAttnModel(train_data_set, test_data_set, hyper_parameter, train_requirement)

