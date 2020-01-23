"""
The data fetcher for Bert Model

Author: Haotian Xue
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer


# TODO: 1. how to do batch with BertTokenizer? => encode_plus has parameter: max_length, pad_to_max_length
# TODO: 2. Write a data fetcher for Bert
class BertDataSet(Dataset):

    def __init__(self, file_path, max_len=100):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.data_x, self.data_y = BertDataSet.data_reader(file_path)
        self.max_len = max_len

    def __getitem__(self, i):
        x_i = self.data_x[i]
        y_i = self.data_y[i]
        input_dict = self.tokenizer.encode_plus(text=x_i,
                                                add_special_tokens=True,
                                                max_length=self.max_len,
                                                pad_to_max_length=True)

        input_ids_list = input_dict['input_ids']
        attn_msk_list = input_dict['attention_mask']
        input_ids = torch.tensor(input_ids_list, dtype=torch.int64)
        attn_msk = torch.tensor(attn_msk_list, dtype=torch.float64)
        label = torch.tensor(y_i, dtype=torch.int64)
        return input_ids, label, attn_msk

    def __len__(self):
        return len(self.data_x)

    @staticmethod
    def data_reader(file_path):
        data_x = []
        data_y = []
        with open(file_path, 'r') as f:
            for line in f:
                tokens = line.strip().split()[1:]  # remove first ""
                if len(tokens) == 0:
                    continue
                x_tokens = tokens[:-1]
                if x_tokens[0].isdigit():
                    x_tokens = x_tokens[2:]  # ['5', '.', 'Science', ...] => ['Science', ...]
                y_token = int(tokens[-1][1])
                text = " ".join(x_tokens)
                data_y.append(y_token)
                data_x.append(text)
        return data_x, data_y


if __name__ == "__main__":
    ds = BertDataSet('../../data/train.txt')
    ds_loader = DataLoader(ds, batch_size=4, shuffle=False)
    for i, d in enumerate(ds_loader):
        x, y, attn_msk = d
        print(x.shape)
        print(attn_msk.shape)
        print(y.shape)
        break
