"""
This file is aiming for providing utility function for conducting model experiments (training, testing, etc)

Author: Haotian Xue
"""
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import numpy as np
import random
from transformers import BertTokenizer


def shuffle_list(*ls):
    l = list(zip(*ls))
    random.shuffle(l)
    return zip(*l)


def write2file(data, file_path):
    with open(file_path, 'w') as file:
        for x, y in data:
            file.write("\" " + x + '\t' + "\"" + str(y) + "\"" + '\n')


# TODO: randomly permute dataset and then split into training and testing accroding to the given proportion
# return: training_path, testing_path
def split_and_write(data_path, proportion=0.1, num_times=2, write_to_file=False, file_path=''):
    # get all X's and Y's from data_path
    data_x = []
    data_y = []
    with open(data_path, 'r') as file:
        for line in file:
            tokens = line.strip().split()[1:]  # remove first ""
            if len(tokens) == 0:
                continue
            x_tokens = tokens[:-1]
            """
            if x_tokens[0].isdigit():
                x_tokens = x_tokens[2:]  # ['5', '.', 'Science', ...] => ['Science', ...]
            """
            y_token = int(tokens[-1][1])
            text = " ".join(x_tokens)
            data_y.append(y_token)
            data_x.append(text)
    num_data = len(data_x)
    assert num_data == len(data_y)
    shuffle_X, shuffle_Y = shuffle_list(data_x, data_y)
    num_testing = int(num_data * proportion)
    train_set, test_set = [], []
    for i in range(num_times):
        start = i * num_testing
        end = i * num_testing + num_testing
        testing_X, testing_Y = shuffle_X[start:end], shuffle_Y[start:end]
        test_set.append((testing_X, testing_Y))
        training_X, training_Y = shuffle_X[:start] + shuffle_X[end:], shuffle_Y[:start] + shuffle_Y[end:]
        train_set.append((training_X, training_Y))
        if write_to_file:
            train_file_name = file_path + 'training{}.txt'.format(i)
            test_file_name = file_path + 'testing{}.txt'.format(i)
            write2file(data=zip(training_X, training_Y), file_path=train_file_name)
            write2file(data=zip(testing_X, testing_Y), file_path=test_file_name)
    return train_set, test_set


def train_model(model, device, ds_loader, criterion, optimizer, scheduler, max_grad_norm, num_epoch, save_path,
                test_ds_loader, testing=True, loading=False, load_path='', record_result=False, result_path=''):
    if loading:
        model.load_state_dict(torch.load(load_path))
    model.train()
    for e in range(num_epoch):
        running_loss = 0.0
        print("{} epoch begins".format(e))
        for i, data in enumerate(ds_loader):
            X, Y, attn_msks = data
            X = X.to(device)
            Y = Y.to(device)
            attn_msks = attn_msks.to(device)
            outputs = model((X, attn_msks))
            loss = criterion(outputs, Y)
            running_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            # print statistics
            if i % 200 == 199:
                print('%d epoch: %d Done, loss = %f' % (e, i, running_loss / 200.0))
                running_loss = 0.0
        if testing:
            f1, precision, recall, accuracy = test_model(model, device, test_ds_loader)
            if record_result:
                with open(result_path, 'a') as file:
                    file.write("F1: {}, precision: {}, recall: {}, accuracy: {}\n".format(f1, precision, recall, accuracy))
            model.train()
    torch.save(model.state_dict(), save_path)
    print("------Finish training------")


def test_model(model, device, test_ds_loader, loading=False, load_path='', record_result=False, result_path=''):
    if loading:
        model.load_state_dict(torch.load(load_path))
    model.eval()
    correct = 0
    total = 0

    # matrix used for computing f1 score
    y_true = None
    y_pred = None

    model.eval()

    with torch.no_grad():
        for i, data in enumerate(test_ds_loader):
            X, labels, attn_msks = data
            X = X.to(device)
            labels = labels.to(device)
            attn_msks = attn_msks.to(device)
            outputs = model((X, attn_msks))
            _, predicted = torch.max(outputs.data, 1)  # predicted shape: [batch_size, 1]
            total += labels.size(0)  # labels shape: [batch_size, 1]
            correct += (predicted == labels).sum().item()
            if i == 0:
                y_true = labels
                y_pred = predicted
            else:
                y_true = torch.cat((y_true, labels), 0)
                y_pred = torch.cat((y_pred, predicted), 0)
        f1 = f1_score(y_true.cpu().numpy(), y_pred.cpu().numpy())
        precision = precision_score(y_true.cpu().numpy(), y_pred.cpu().numpy())
        recall = recall_score(y_true.cpu().numpy(), y_pred.cpu().numpy())
        accuracy = accuracy_score(y_true.cpu().numpy(), y_pred.cpu().numpy())
        if record_result:
            with open(result_path, 'a') as file:
                file.write("F1: {}, precision: {}, recall: {}, accuracy: {}\n".format(f1, precision, recall, accuracy))
        print('F1 score: ', f1)
        print('Precision score: ', precision)
        print('Recall score: ', recall)
        print('Accuracy score: ', accuracy)
        return f1, precision, recall, accuracy


# TODO: write predicted result to submitted format style to txt
def write_result(model, device, test_file, write_file_path, loading=False, load_path=''):
    if loading:
        model.load_state_dict(torch.load(load_path))
    model.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    with torch.no_grad():
        with open(test_file, 'r') as r_f:
            with open(write_file_path, 'w') as w_f:
                for i, line in enumerate(r_f):
                    tokens = line.strip().split()[1:]  # remove first ""
                    if len(tokens) == 0:
                        continue
                    x_tokens = tokens[:]
                    if x_tokens[0].isdigit():
                        x_tokens = x_tokens[2:]  # ['5', '.', 'Science', ...] => ['Science', ...]
                    text = " ".join(x_tokens)
                    input_dict = tokenizer.encode_plus(text=text,
                                                       add_special_tokens=True,
                                                       max_length=100,
                                                       pad_to_max_length=True)
                    input_ids_list = input_dict['input_ids']
                    attn_msk_list = input_dict['attention_mask']
                    input_ids = torch.tensor(input_ids_list, dtype=torch.int64)
                    attn_msk = torch.tensor(attn_msk_list, dtype=torch.float64)
                    input_ids.to(device)
                    attn_msk.to(device)
                    outputs = model(input_ids, attn_msk)
                    _, predicted = torch.max(outputs.data, 1)  # predicted shape: [batch_size, 1]
                    y_hat = predicted.cpu().item()
                    w_f.write(" {} {}\n".format(tokens, y_hat))
