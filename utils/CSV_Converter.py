import pandas as pd

train_data = []  # 16659
test_data = []  # 810

'''读取文件'''
print('Reading files...')
print('\n')
train_f = open("train.txt", encoding='utf-8')
for line in train_f:
    train_data.append(line)
train_f.close()

test_f = open("test.txt", encoding='utf-8')
for line in test_f:
    test_data.append(line)
test_f.close()
print(train_data[0])
print(test_data[0])
print('\n')
print('Files get!...')

'''去掉开头序号-> [[句子(单词间以,为分割)], 0/1]'''
print('Formatting...')
print('\n')
train_set = []
for unit in train_data:
    temp = unit.split(' ')
    if temp[1].isdigit():
        # 判断第一个字符是否为数字，如果是则去掉，如果不是则取句子
        temp = temp[3:-1]
        # 当前temp中的最后一部分为"\t.\n","最后一个引号" 为了保证格式规范，先去掉，最后在取corpus的时候加上句号即可
        # training_set.append(([temp, unit[-3:-2]]))
        train_set.append([' '.join(temp), unit[-3:-2]])
        # unit[-3:-2]为当前句子的label
    else:
        # temp[1]为引号，应去掉
        temp = temp[1:-1]
        # training_set.append(([temp, unit[-3:-2]]))
        train_set.append([' '.join(temp), unit[-3:-2]])

test_set = []
for unit in test_data:
    temp = unit.split(' ')
    if temp[1].isdigit():
        temp = temp[3:-1]
        # training_set.append(([temp, unit[-3:-2]]))
        test_set.append([' '.join(temp), unit[-3:-2]])
    else:
        temp = temp[1:-1]
        # training_set.append(([temp, unit[-3:-2]]))
        test_set.append([' '.join(temp), unit[-3:-2]])

print(train_set[0])
print(test_set[0])
print('\n')
print('All data sets are formatted!')

# label: str->int
# for samples in train_set:
#     print(samples)
#     break
print('Converting labels to Int()...\n')


def label2int(data):
    res = []
    temp_res = []
    for samples in data:
        sents = samples[0] + '.'
        temp_res = [sents, int(samples[1])]
        res.append(temp_res)

    return res


train_set = label2int(train_set)
test_set = label2int(test_set)
print(train_set[0])
print(test_set[0])
print('\nComplete!')
sents_train = []
labels_train = []
for items in train_set:
    sents_train.append(items[0])
    labels_train.append(items[1])

sents_test = []
labels_test = []
for stuff in test_set:
    sents_test.append(stuff[0])
    labels_test.append(stuff[1])


dataFrame_train = pd.DataFrame({'sents': sents_train, 'label': labels_train})
dataFrame_train.to_csv('train.csv', index=False, sep=',')

dataFrame_train = pd.DataFrame({'sents': sents_test, 'label': labels_test})
dataFrame_train.to_csv('test.csv', index=False, sep=',')
