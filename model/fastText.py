"""
The implementation of FastText model.

Author: Haotian Xue
"""

import torch
import torch.nn as nn
from sen_tensor_model_class import SenTensorModel


class FastTextModel(SenTensorModel):

    def __init__(self,
                 train_data_set,
                 test_data_set,
                 hyper_parameter,
                 train_requirement,
                 is_gpu=torch.cuda.is_available(),
                 model_save_path="../trained_model/rnn_model.pt"):
        super(FastTextModel, self).__init__(train_data_set,
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
        d_w, hidden_dim, dropout_prob = self.extract_hyper_parameters()
        print("-----Start building model-----")
        model = FastTextModelHelper(d_w,
                                    torch.from_numpy(self.test_data_set.word_embedding),
                                    hidden_dim,
                                    dropout_p=dropout_prob)
        print("-----Finish building model-----")
        return model

    def extract_hyper_parameters(self):
        return self.hyper_parameter["d_w"], \
               self.hyper_parameter["hidden_dim"], \
               self.hyper_parameter["dropout_prob"], \



class FastTextModelHelper(nn.Module):

    def __init__(self, d_w, word_emb_weight, hidden_dim,
                 num_classes=2, dropout_p=0.2):
        super(FastTextModelHelper, self).__init__()
        self.hidden_dim = hidden_dim
        self.w2v = nn.Embedding.from_pretrained(word_emb_weight, freeze=False)
        self.linear_layer = nn.Sequential(  # int_shape: (batch_size, d_w)
            nn.Dropout(dropout_p),
            nn.Linear(d_w, num_classes)
        )
        self.linear_layer.apply(self.weights_init)

    def forward(self, x):
        x = self.w2v(x)  # (batch, max_sen_len, d_w)
        out = torch.mean(x, dim=1, keepdim=False)  # (batch, d_w)
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
    train_requirement = {"num_epoch": 30, "batch_size": 32, "lr": 3e-4}
    hyper_parameter = {"d_w": 50, "hidden_dim": 128, "dropout_prob": 0.5}
    train_data_set = SenSemEvalDataSet("../data/train.txt", "../data/word_embedding/glove.6B.50d.txt", 50, True)
    test_data_set = SenSemEvalDataSet("../data/test.txt", "../data/word_embedding/glove.6B.50d.txt", 50, True, 150)
    model = FastTextModel(train_data_set, test_data_set, hyper_parameter, train_requirement)

