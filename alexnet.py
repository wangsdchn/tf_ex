#!usr/bin/env python
# -*- coding:utf-8 _*-
import tensorflow as tf

batch_size = 32
num_batches = 100


def printTensorShape(t):
    print(t.op.name, ' ', t.get_shape().as_list())


def inference(images):
    with tf.name_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0., shape=[64], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope)
        printTensorShape(conv1)

        lrn1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9, beta=0.75, name='lrn1')
        pool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')
        printTensorShape(pool1)

    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0., shape=[192], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope)
        printTensorShape(conv2)

        lrn2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9, beta=0.75, name='lrn2')
        pool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')
        printTensorShape(pool2)

    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 192, 384], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0., shape=[384], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope)
        printTensorShape(conv3)

    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 384, 256], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0., shape=[256], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope)
        printTensorShape(conv4)

    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 256, 256], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0., shape=[256], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope)
        printTensorShape(conv5)
        pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool5')
        printTensorShape(pool5)
    flattened = tf.reshape(pool5, shape=[-1, 6 * 6 * 256])

    with tf.name_scope('fc6') as scope:
        weight = tf.Variable(tf.truncated_normal([256 * 6 * 6, 4096], dtype=tf.float32, stddev=1e-1), name='weight')
        biases = tf.Variable(tf.constant(0., shape=[1], dtype=tf.float32), trainable=True, name='biased')
        fc6 = tf.nn.relu(tf.add(tf.matmul(flattened, weight), biases), name=scope)
        printTensorShape(fc6)
        dropout6 = tf.nn.dropout(fc6, keep_prob=0.5, name='dropout1')

    with tf.name_scope('fc7') as scope:
        weight = tf.Variable(tf.truncated_normal([4096, 4096], dtype=tf.float32, stddev=1e-1), name='weight')
        biases = tf.Variable(tf.constant(0., shape=[4096], dtype=tf.float32), trainable=True, name='biased')
        fc7 = tf.nn.relu(tf.add(tf.matmul(dropout6, weight), biases), name=scope)
        printTensorShape(fc7)
        dropout7 = tf.nn.dropout(fc7, keep_prob=0.5, name='dropout2')

    with tf.name_scope('fc8') as scope:
        weight = tf.Variable(tf.truncated_normal([4096, 2], dtype=tf.float32, stddev=1e-1), name='weight')
        biases = tf.Variable(tf.constant(0., shape=[2], dtype=tf.float32), trainable=True, name='biased')
        fc8 = tf.add(tf.matmul(dropout7, weight), biases)
        printTensorShape(fc8)
    return fc8


def losses(logits, labels):
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name+'/loss', loss)
    return loss


def trainning(loss, learning_rate):
    with tf.name_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
    return optimizer


def evaluation(logits, labels):
  with tf.variable_scope('accuracy') as scope:
      correct = tf.nn.in_top_k(logits, labels, 1)
      correct = tf.cast(correct, tf.float16)
      accuracy = tf.reduce_mean(correct)
      tf.summary.scalar(scope.name+'/accuracy', accuracy)
  return accuracy


if __name__ == '__main__':
    image_size = 224
    images = tf.Variable(tf.random_normal([batch_size,
                                           image_size,
                                           image_size, 3],
                                          dtype=tf.float32,
                                          stddev=1e-1))
    fc8, parameters = inference(images)
