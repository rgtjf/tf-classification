# coding: utf-8
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf

import utils
import config
import evaluation
from data import Task
from models.NBoW import NBoWModel


FLAGS = tf.flags.FLAGS
tf.set_random_seed(1234)

# File Parameters
tf.flags.DEFINE_string('log_file', 'tf-classification.log', 'path of the log file')
# tf.flags.DEFINE_string('summary_dir', 'summary', 'path of summary_dir')
# tf.flags.DEFINE_string('description', __file__, 'commit_message')

# Task Parameters
tf.flags.DEFINE_string('model', 'nbow', 'given the model name')
tf.flags.DEFINE_integer('max_epoch', 10, 'max epoches')
tf.flags.DEFINE_integer('display_step', 100, 'display each step')

# Hyper Parameters
tf.flags.DEFINE_integer('batch_size', 256, 'batch size')
tf.flags.DEFINE_float('drop_keep_rate', 0.9, 'dropout_keep_rate')
tf.flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
tf.flags.DEFINE_float('lambda_l2', 0.0, 'lambda_l2')
tf.flags.DEFINE_float('clipper', 30, 'clipper')

FLAGS._parse_flags()

# Logger Part
logger = utils.get_logger(FLAGS.log_file)
logger.info(FLAGS.__flags)


def main():
    task = Task(init=True)
    FLAGS.we = task.embed
    model = NBoWModel(FLAGS)

    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True

    best_dev_result = 0.
    best_test_result = 0.

    with tf.Session(config=gpu_config) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        total_batch = 0
        for epoch in range(FLAGS.max_epoch):
            total_batch += 1
            for batch in task.train_data.batch_iter(FLAGS.batch_size, shuffle=True):
                results = model.train_model(sess, batch)
                step = results['global_step']
                loss = results['loss']

                if total_batch % FLAGS.display_step == 0:
                    print('batch_{} steps_{} cost_val: {:.5f}'.format(total_batch, step, loss))
                    logger.info('==>  Epoch {:02d}/{:02d}: '.format(epoch, total_batch))

            def do_eval(dataset):
                batch_size = 100  # for Simple
                preds, golds = [], []
                for batch in dataset.batch_iter(batch_size):
                    results =  model.test_model(sess, batch)
                    preds.append(results['predict_label'])
                    golds.append(np.argmax(batch.label, 1))
                preds = np.concatenate(preds, 0)
                golds = np.concatenate(golds, 0)
                predict_labels = [config.id2category[predict] for predict in preds]
                gold_labels = [config.id2category[gold] for gold in golds]
                acc = evaluation.Evalation_lst(predict_labels, gold_labels)
                return acc

            dev_acc = do_eval(task.dev_data)
            test_acc = do_eval(task.test_data)
            logger.info(
                'dev = {:.5f}, test = {:.5f}'.format(dev_acc, test_acc))

            if dev_acc > best_dev_result:
                best_dev_result = dev_acc
                best_test_result = test_acc
            logger.info(
                'dev = {:.5f}, test = {:.5f} best!!!!'.format(best_dev_result, best_test_result))

if __name__ == '__main__':
    main()