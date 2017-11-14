# coding: utf8
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
import config
import tf_utils


class LSTMModel(object):

    def __init__(self, FLAGS=None):
        self.FLAGS = FLAGS
        self.config = config

        self.seq_len = config.max_sent_len
        self.embed_size = config.word_dim
        self.num_class = config.num_class
        self.lstm_size = 100

        # Add PlaceHolder
        self.input_x = tf.placeholder(tf.int32, (None, self.seq_len))  # [batch_size, sent_len]
        self.input_x_len = tf.placeholder(tf.int32, (None,))
        self.input_y = tf.placeholder(tf.int32, (None, self.num_class))

        self.drop_keep_rate = tf.placeholder(tf.float32)
        self.learning_rate = tf.placeholder(tf.float32)

        # Add Word Embedding
        self.we = tf.Variable(FLAGS.we, name='emb')

        # Build the Computation Graph
        embedded_x = tf.nn.embedding_lookup(self.we, self.input_x)  # [batch_size, sent_len, emd_size]

        def BiLSTM(input_x, input_x_len, hidden_size, num_layers=1, dropout_rate=None, return_sequence=True):
            """
            BiLSTM Layer
            Args:
                input_x: [batch, sent_len, emb_size]
                input_x_len: [batch, ]
                hidden_size: int
                num_layers: int
                dropout_rate: float
                return_sequence: True/False
            Returns:
                if return_sequence=True:
                    outputs: [batch, sent_len, hidden_size*2]
                else:
                    output: [batch, hidden_size*2]
            """
            # cell = tf.contrib.rnn.GRUCell(hidden_size)
            cell_fw = tf.contrib.rnn.BasicLSTMCell(hidden_size)
            cell_bw = tf.contrib.rnn.BasicLSTMCell(hidden_size)

            if num_layers > 1:
                # Warning! Please consider that whether the cell to stack are the same
                cell_fw = tf.contrib.rnn.MultiRNNCell([cell_fw for _ in range(num_layers)])
                cell_bw = tf.contrib.rnn.MultiRNNCell([cell_bw for _ in range(num_layers)])

            if dropout_rate:
                cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=(1 - dropout_rate))
                cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=(1 - dropout_rate))

            b_outputs, b_states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, input_x,
                                                                  sequence_length=input_x_len, dtype=tf.float32)
            if return_sequence:
                outputs = tf.concat(b_outputs, axis=2)
            else:
                # states: [c, h]
                outputs = tf.concat([b_states[0][1], b_outputs[1][1]], axis=-1)
            return outputs

        with tf.variable_scope("bilstm") as s:
            lstm_x = BiLSTM(embedded_x, self.input_x_len, self.lstm_size,
                            return_sequence=True)

        avg_pooling = tf_utils.AvgPooling(embedded_x, self.input_x_len, self.seq_len)
        max_pooling = tf_utils.MaxPooling(lstm_x, self.input_x_len)

        logits = tf_utils.linear([max_pooling, avg_pooling], self.num_class, bias=True, scope='softmax')

        # Obtain the Predict, Loss, Train_op
        predict_prob = tf.nn.softmax(logits, name='predict_prob')
        predict_label = tf.cast(tf.argmax(logits, 1), tf.int32)

        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.input_y)
        loss = tf.reduce_mean(loss)

        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if v.get_shape().ndims > 1])
        reg_loss = loss + FLAGS.lambda_l2 * l2_loss

        # Build the loss
        global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        # optimizer = tf.train.AdagradOptimizer(self.learning_rate)

        if FLAGS.clipper:
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), FLAGS.clipper)
            train_op = optimizer.apply_gradients(list(zip(grads, tvars)))
        else:
            train_op = optimizer.minimize(loss, global_step=global_step)

        self.predict_prob = predict_prob
        self.predict_label = predict_label
        self.loss = loss
        self.reg_loss = reg_loss
        self.train_op = train_op
        self.global_step = global_step

    def train_model(self, sess, batch):
        feed_dict = {
            self.input_x: batch.sent,
            self.input_x_len: batch.sent_len,
            self.input_y: batch.label,
            self.drop_keep_rate: self.FLAGS.drop_keep_rate,
            self.learning_rate: 1e-3
        }
        to_return = {
            'train_op': self.train_op,
            'loss': self.loss,
            'global_step': self.global_step,
        }
        return sess.run(to_return, feed_dict)

    def test_model(self, sess, batch):
        feed_dict = {
            self.input_x: batch.sent,
            self.input_x_len: batch.sent_len,
            self.drop_keep_rate: 1.0
        }
        to_return = {
            'predict_label': self.predict_label,
            'predict_prob': self.predict_prob
        }
        return sess.run(to_return, feed_dict)