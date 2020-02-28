"""
Train script for training Bert + Capsule model
"""
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import numpy as np
from torch.utils.data import Dataset, DataLoader
from experiment_utils import train_model, write_result
from Bert_DataFetcher import BertDataSet
from Capsule_Bert_model import BertCapsuleNet


training_path = "../../data/train.txt"
validate_path = "../../data/test.txt"
testing_path = "../../data/test_files/final_test.txt"
save_path = '../../trained_model/final_bert_cap.pt'
record_path = '../../trained_model/final_result.txt'
write_to_path = '../../data/test_files/predicted.txt'

ds = BertDataSet(training_path)
ds_loader = DataLoader(ds, batch_size=16, shuffle=True)

validate_ds = BertDataSet(validate_path)
validate_ds_loader = DataLoader(validate_ds, batch_size=16, shuffle=False)

criterion = nn.CrossEntropyLoss()

lr = 2e-3  # 2e-3
max_grad_norm = 1.0
num_training_steps = len(ds.data_x) * 10
num_warmup_steps = 0
warmup_proportion = float(num_warmup_steps) / float(num_training_steps)  # 0.1


device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )


model = BertCapsuleNet(freeze_bert=False)
model = model.to(device)

optimizer = AdamW(model.parameters(),
                  lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                  )
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=num_warmup_steps,
                                            num_training_steps=num_training_steps)  # PyTorch scheduler


num_epoch = 8


train_model(model, device, ds_loader, criterion, optimizer, scheduler, max_grad_norm, num_epoch, save_path,
            test_ds_loader=validate_ds_loader, record_result=True, result_path=record_path)

write_result(model, device, testing_path, write_to_path)
