from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.layers import l1_regularizer, l2_regularizer

class LinkPredictionModel(object):

    def __init__(self, batch_size, prediction_pair,
                 num_hidden, num_layers, output_size):

        self._prediction_pair = prediction_pair
        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self._batch_size = batch_size
        self._output_size = output_size

        self.weight_regularizer = l2_regularizer(0.001)


        # Initialization of the input and target variables:
        self.inputs = tf.placeholder(tf.float32, [ self._batch_size, self._prediction_pair])
        self.targets = tf.placeholder(tf.float32,  [self._batch_size, self._output_size]) 


        # Bias and weight initializer
        initializer_weights = tf.variance_scaling_initializer()
        initializer_biases  = tf.constant_initializer(0.0)

        # Builing the stacked network. By adding the number of layers on top of eachother in a list.
        i = 0
        self.weights = {}
        self.biases = {}

        # First hidden layer
        with tf.variable_scope('Layer'+str(0)):
            self.weights[i] = tf.get_variable(
                name='weights',
                shape=[self._prediction_pair, self._num_hidden],
                initializer=initializer_weights,
                regularizer=self.weight_regularizer
            )
            self.biases[i] = tf.Variable(tf.zeros([self._num_hidden]), name='biases')
            i += 1



        # Looping over the amount of layers.
        for layer in range(1,self._num_layers):
            with tf.variable_scope('Layer'+str(layer)):
                self.weights[i] = tf.get_variable(
                    name='weights',
                    shape=[self._num_hidden, self._num_hidden],
                    initializer=initializer_weights,
                    regularizer=self.weight_regularizer
                )
                self.biases[i] = tf.Variable(tf.zeros([self._num_hidden]), name='biases')
                i += 1

        # Looping over the amount of layers.
        with tf.variable_scope('Final_Layer'):
            self.weights[i] = tf.get_variable(
                name='weights',
                shape=[self._num_hidden, self._output_size],
                initializer=initializer_weights,
                regularizer=self.weight_regularizer
            )
            self.biases[i] = tf.Variable(tf.zeros([self._output_size]), name='biases')


        # # Initialization of the softmax linear layer
        # with tf.variable_scope('softmax'):
        #     self.W = tf.get_variable('W', [self._num_hidden, self._output_size], initializer=initializer_weights)
        #     self.b = tf.get_variable('b', [self._output_size], initializer=initializer_biases)



    def _run_model(self, inputs):
        # Implement your model to return the logits per step of shape:
        #   [batch_size, vocab_size]
        
        for layer in range(0,self._num_layers):
            if layer == 0:
                self.logits = tf.layers.dense(inputs=inputs, units=128, activation=tf.nn.relu)
                # self.logits = tf.nn.relu(tf.matmul(inputs, self.weights[layer]) + self.biases[layer])
            else:
                self.logits = tf.layers.dense(inputs=self.logits, units=128, activation=tf.nn.relu)
                # self.logits = tf.nn.relu(tf.matmul(self.logits, self.weights[layer]) + self.biases[layer])
        # self.logits = tf.matmul(self.logits, self.weights[self._num_layers]) + self.biases[self._num_layers]
        self.logits = tf.layers.dense(inputs=self.logits, units=2, activation=tf.nn.sigmoid)
        return self.logits


    def _compute_loss(self, targets):
        # Cross-entropy loss, averaged over timestep and batch

        # self.loss = (tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        #         logits=self.logits, labels=targets, name='cross_entropy')))

        self.loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=targets, predictions=self.logits))
        return self.loss


    def _compute_accuracy(self, targets):
        # Implement the accuracy of predicting the last digit over the current batch
        # Operation comparing softmax prediction with true label.

        correct_prediction = tf.equal(tf.cast(self.logits,tf.float32), targets)

        # Operation calculating the accuracy of the predictions
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return self.accuracy