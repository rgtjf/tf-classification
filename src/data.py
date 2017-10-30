# coding: utf-8
from __future__ import print_function

import os
import pickle
import codecs
import six
import numpy as np
np.random.seed(1234)

import data_utils
import config


def read_data(file_list):
    """
    load data from file list
    Args: file_list:
    Returns:
    """
    if type(file_list) != list:
        file_list = [file_list]

    examples = []
    for file in file_list:
        with codecs.open(file, 'r', encoding='utf8') as f:
            for line in f:
                items = line.strip().split('\t')
                label = items[0]
                sent = items[1].split()
                examples.append((sent, label))
    return examples


class Dataset(object):
    def __init__(self, file_list,
                 word_vocab,
                 max_sent_len,
                 num_class):
        """
        return the formatted matrix, which is used as the input to deep learning models
        Args: file_list:
              word_vocab:
        """
        self.examples = examples = read_data(file_list)

        y = []
        for example in examples:
            label = config.category2id[example[1]]
            one_hot_label = data_utils.onehot_vectorize(label, num_class)
            y.append(one_hot_label)

        sents_index = []
        sents_len = []
        for example in examples:
            sent = example[0]
            sent = data_utils.sent_to_index(sent, word_vocab)
            sents_index.append(sent)
            sents_len.append(min(len(sent), max_sent_len))

        x = data_utils.pad_2d_matrix(sents_index, max_sent_len)
        x_len = sents_len

        self.x = np.array(x, dtype=np.int32)  # [batch_size, sent_len]
        self.x_len = np.array(x_len, dtype=np.int32)  # [batch_size]
        self.y = np.array(y, dtype=np.float32)  # [batch_size, class_number]


    def batch_iter(self, batch_size, shuffle=False):
        """
        UPDATE_0: add Batch for yield
        To support different model with different data:
        - model_1 want data 1, 2, 3, 4;
        - model_2 want data 1, 2, 3, 4, 5;
        ===
        during training: add some data to be enough batch_size
        during test: add some data to be enough batch_size
        :param batch_size:
        :param shuffle:
        :return:
        """
        x = self.x
        x_len = self.x_len
        y = self.y
        assert len(x) == len(y)
        n_data = len(y)

        idx = np.arange(n_data)
        if shuffle:
            idx = np.random.permutation(n_data)

        for start_idx in range(0, n_data, batch_size):
            # end_idx = min(start_idx + batch_size, n_data)
            end_idx = start_idx + batch_size
            excerpt = idx[start_idx:end_idx]

            batch = data_utils.Batch()
            batch.add('sent', x[excerpt])
            batch.add('sent_len', x_len[excerpt])
            batch.add('label', y[excerpt])
            yield batch


class Task(object):

    def __init__(self, init=False):
        self.train_file = config.train_file
        self.dev_file = config.dev_file
        self.test_file = config.test_file
        self.word_embed_file = config.word_embed_file

        self.word_dim = config.word_dim
        self.max_len = config.max_sent_len
        self.num_class = config.num_class

        self.we_file = config.we_file
        self.w2i_file = config.w2i_file

        self.train_predict_file = None
        self.dev_predict_file = None
        self.test_predict_file = None

        if init:
            self.word_vocab = self.build_vocab()
            self.embed = data_utils.load_word_embedding(self.word_vocab, self.word_embed_file, self.word_dim)

            data_utils.save_params(self.word_vocab, self.w2i_file)
            data_utils.save_params(self.embed, self.we_file)
        else:
            self.embed = data_utils.load_params(self.we_file)
            self.word_vocab = data_utils.load_params(self.w2i_file)
            self.embed = self.embed.astype(np.float32)

        print("vocab size: %d" % len(self.word_vocab), "we shape: ", self.embed.shape)
        self.train_data = Dataset(self.train_file, self.word_vocab, self.max_len, self.num_class)
        self.dev_data = Dataset(self.dev_file, self.word_vocab, self.max_len, self.num_class)
        if self.test_file:
            self.test_data = Dataset(self.test_file, self.word_vocab, self.max_len, self.num_class)

    def build_vocab(self):
        """
            build sents is for build vocab
            during multi-lingual task, there are two kinds of sents
            :return: sents
            """
        if self.test_file is None:
            print('test_file is None')
            file_list = [self.train_file, self.dev_file]
        else:
            file_list = [self.train_file, self.dev_file, self.test_file]

        examples = read_data(file_list)
        sents = []
        for example in examples:
            sent = example[0]
            sents.append(sent)
        vocab = data_utils.build_word_vocab(sents)

        max_sent_len = 0
        avg_sent_len = 0
        for sent in sents:
            if len(sent) > max_sent_len:
                max_sent_len = len(sent)
            avg_sent_len += len(sent)
        avg_sent_len /= len(sents)
        print('task: max_sent_len: {}'.format(max_sent_len))
        print('task: avg_sent_len: {}'.format(avg_sent_len))
        return vocab


if __name__ == '__main__':
    pass