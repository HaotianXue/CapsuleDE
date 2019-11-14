"""
The implementation of Recurrent CNN model.

Author: Haotian Xue
"""

import torch
import torch.nn as nn
import attention
from sen_tensor_model_class import SenTensorModel


class RCnnModel(SenTensorModel):

    def __init__(self,
                 train_data_set,
                 test_data_set,
                 hyper_parameter,
                 train_requirement,
                 is_gpu=torch.cuda.is_available(),
                 model_save_path="../trained_model/rnn_model.pt"):
        super(RCnnModel, self).__init__(train_data_set,
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
        d_w, hidden_dim, num_layers, dropout_prob, window_size, num_filter = self.extract_hyper_parameters()
        print("-----Start building model-----")
        model = RCnnModelHelper(d_w,
                                torch.from_numpy(self.test_data_set.word_embedding),
                                hidden_dim,
                                num_layers=num_layers,
                                window_size=window_size,
                                num_filter=num_filter,
                                dropout_p=dropout_prob)
        print("-----Finish building model-----")
        return model

    def extract_hyper_parameters(self):
        return self.hyper_parameter["d_w"], \
               self.hyper_parameter["hidden_dim"], \
               self.hyper_parameter["num_layers"], \
               self.hyper_parameter["dropout_prob"], \
               self.hyper_parameter["window_size"], \
               self.hyper_parameter["num_filter"]


class RCnnModelHelper(nn.Module):

    def __init__(self, d_w, word_emb_weight, hidden_dim, num_layers,
                 num_classes=2, window_size=3, num_filter=128, dropout_p=0.2):
        super(RCnnModelHelper, self).__init__()
        self.hidden_dim = hidden_dim
        self.w2v = nn.Embedding.from_pretrained(word_emb_weight, freeze=False)
        self.rnn_layer = nn.GRU(input_size=d_w,
                                hidden_size=hidden_dim,
                                num_layers=num_layers,
                                bias=True,
                                batch_first=True,
                                dropout=dropout_p,
                                bidirectional=True)  # shape: (batch_size, max_sen_len, hidden_size*2)
        self.rnn_layer.apply(self.weights_init)
        self.cnn_layer = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=num_filter,
                      kernel_size=(window_size, hidden_dim * 2),
                      stride=(1, 1),
                      padding=(1, 0)),  # out_shape: (batch_size, num_filter, max_sen_len, 1)
            nn.MaxPool2d(kernel_size=(150, 1),
                         stride=(1, 1)),  # out_shape: (batch_size, num_filter, 1, 1)
            nn.Dropout(dropout_p)
        )
        self.cnn_layer.apply(self.weights_init)
        self.linear_layer = nn.Sequential(  # int_shape: (batch_size, hidden_size*2)
            nn.Linear(num_filter, num_filter // 2),
            nn.Tanh(),
            nn.Dropout(dropout_p),
            nn.Linear(num_filter // 2, num_classes)  # out_shape: (batch_size, num_classes)
        )
        self.linear_layer.apply(self.weights_init)

    def forward(self, x):
        x = self.w2v(x)
        out, _ = self.rnn_layer(x)  # (batch, sen_len, hidden_size*2)
        out = torch.unsqueeze(out, dim=1)  # (batch, 1, sen_len, hidden_size*2)
        out = self.cnn_layer(out)  # (batch, num_filter, 1, 1)
        out = out.view(out.shape[0], -1)  # (batch, num_filter)
        out = self.linear_layer(out)
        return out

    # method to initialize the model weights (in order to improve performance)
    def weights_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
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
    train_requirement = {"num_epoch": 10, "batch_size": 32, "lr": 3e-4}
    hyper_parameter = {"d_w": 50, "hidden_dim": 128, "num_layers": 1, "dropout_prob": 0.5, "window_size": 3, "num_filter": 128}
    train_data_set = SenSemEvalDataSet("../data/train.txt", "../data/word_embedding/glove.6B.50d.txt", 50, True)
    test_data_set = SenSemEvalDataSet("../data/test.txt", "../data/word_embedding/glove.6B.50d.txt", 50, True, 150)
    model = RCnnModel(train_data_set, test_data_set, hyper_parameter, train_requirement)

