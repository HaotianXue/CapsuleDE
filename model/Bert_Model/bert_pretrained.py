"""
Use pretrained-BERT model to do definition extraction

Author: Haotian Xue
"""

import torch
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import numpy as np

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
model.eval()

correct = 0
total = 0

# matrix used for computing f1 score
y_true = None
y_pred = None

with torch.no_grad():
    # See the models docstrings for the detail of the inputs
    with open('../data/test.txt', 'r') as f:
        for i, line in enumerate(f):
            tokens = line.strip().split()[1:]  # remove first ""
            if len(tokens) == 0:
                continue
            x_tokens = tokens[:-1]
            if x_tokens[0].isdigit():
                x_tokens = x_tokens[2:]  # ['5', '.', 'Science', ...] => ['Science', ...]
            y_token = int(tokens[-1][1])
            text = " ".join(x_tokens)
            input_ids = tokenizer.encode(text, add_special_tokens=True)
            tokens_tensor = torch.tensor([input_ids])
            outputs = model(tokens_tensor)
            logits = outputs[0]
            labels = torch.tensor(np.array(y_token, ndmin=2))  # shape: (batch_size=1, 1)
            _, predicted = torch.max(logits.data, 1)  # predicted shape: [batch_size=1, 1]
            total += 1  # labels shape: [batch_size=1, 1]
            correct += (predicted == labels).sum().item()
            if i == 0:
                y_true = labels
                y_pred = predicted
            else:
                y_true = torch.cat((y_true, labels), 0)
                y_pred = torch.cat((y_pred, predicted), 0)
    print('F1 score: ', f1_score(y_true.numpy(), y_pred.numpy()))
    print('Precision score: ', precision_score(y_true.numpy(), y_pred.numpy()))
    print('Recall score: ', recall_score(y_true.numpy(), y_pred.numpy()))
    print('Accuracy score: ', accuracy_score(y_true.numpy(), y_pred.numpy()))

