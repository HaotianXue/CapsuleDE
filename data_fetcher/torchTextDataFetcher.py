"""
The implementaion of torchtext based data fetcher

Author: Haotian Xue
"""

import torchtext
from torchtext import data


class TorchTextSemEvalDataSet:

    """
    Sentence level of SemEval2020 task6 data set (including training, testing, etc)
    """

    def __init__(self, training_path, testing_path, w2v_path, batch_size, is_padding=False, max_len=150, is_gpu=False):
        if is_padding:
            self.TEXT = data.Field(sequential=True, batch_first=True, tokenize='spacy', fix_length=max_len)
            self.LABEL = data.Field(sequential=False, use_vocab=False, is_target=True, fix_length=max_len)
        else:
            self.TEXT = data.Field(sequential=True, batch_first=True, tokenize='spacy')
            self.LABEL = data.Field(sequential=False, use_vocab=False, is_target=True)
        self.train_set = torchtext.data.TabularDataset(path=training_path,
                                                       format='CSV',
                                                       fields=[('sents', self.TEXT), ('label', self.LABEL)],
                                                       skip_header=True)
        self.test_set = torchtext.data.TabularDataset(path=testing_path,
                                                      format='CSV',
                                                      fields=[('sents', self.TEXT), ('label', self.LABEL)],
                                                      skip_header=True)
        self.TEXT.build_vocab(self.train_set, self.test_set)
        self.TEXT.vocab.load_vectors(torchtext.vocab.Vectors(w2v_path))
        self.train_iter = data.BucketIterator(dataset=self.train_set,
                                              batch_size=batch_size,
                                              sort_key=lambda x: len(x.sents),
                                              shuffle=True,
                                              sort_within_batch=True)
        self.test_iter = torchtext.data.BucketIterator(dataset=self.test_set,
                                                       batch_size=batch_size,
                                                       sort_key=lambda x: len(x.sents),
                                                       sort_within_batch=True)

