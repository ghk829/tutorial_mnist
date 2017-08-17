# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains model definitions."""
import math

import models
import tensorflow as tf
import utils

import tensorflow.contrib.slim as slim

class LogisticModel(models.BaseModel):
  """Logistic model with L2 regularization."""

  def create_model(self, model_input, num_classes=10, l2_penalty=1e-8, **unused_params):
    """Creates a logistic model.

    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""
    output = slim.fully_connected(
        model_input, num_classes, activation_fn=tf.nn.softmax,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    return {"predictions": output}

class MyModel(models.BaseModel):
  """Logistic model with L2 regularization."""
  def nielsen_net(inputs, is_training, scope='NielsenNet'):
    with tf.variable_scope(scope, 'NielsenNet'):
        # First Group: Convolution + Pooling 28x28x1 => 28x28x20 => 14x14x20
        net = slim.conv2d(inputs, 20, [5, 5], padding='SAME', scope='layer1-conv')
        net = slim.max_pool2d(net, 2, stride=2, scope='layer2-max-pool')

        # Second Group: Convolution + Pooling 14x14x20 => 10x10x40 => 5x5x40
        net = slim.conv2d(net, 40, [5, 5], padding='VALID', scope='layer3-conv')
        net = slim.max_pool2d(net, 2, stride=2, scope='layer4-max-pool')

        # Reshape: 5x5x40 => 1000x1
        net = tf.reshape(net, [-1, 5*5*40])

        # Fully Connected Layer: 1000x1 => 1000x1
        net = slim.fully_connected(net, 1000, scope='layer5')
        net = slim.dropout(net, is_training=is_training, scope='layer5-dropout')

        # Second Fully Connected: 1000x1 => 1000x1
        net = slim.fully_connected(net, 1000, scope='layer6')
        net = slim.dropout(net, is_training=is_training, scope='layer6-dropout')

        # Output Layer: 1000x1 => 10x1
        net = slim.fully_connected(net, 10, scope='output')
        net = slim.dropout(net, is_training=is_training, scope='output-dropout')

	return net
  def create_model(self, model_input, num_classes=10, l2_penalty=1e-8, **unused_params):
    """Creates a logistic model.

    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""
    net = self.nielsen_net(model_input,True, scope='NielsenNetTrain')
    output = slim.fully_connected(
        net, num_classes, activation_fn=tf.nn.softmax,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    return {"predictions": output}
