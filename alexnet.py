#!usr/bin/env python
# -*- coding:utf-8 _*-
import tensorflow as tf
import numpy as np

batch_size = 32
num_batches = 100


def printTensorShape(t):
    print(t.op.name, ' ', t.get_shape().as_list())


def inference(images):
    with tf.name_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 96], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='VALID')
        biases = tf.Variable(tf.constant(0.1, shape=[96], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope)
        printTensorShape(conv1)

        lrn1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9, beta=0.75, name='lrn1')
        pool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')
        printTensorShape(pool1)

    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 96, 256], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.1, shape=[256], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope)
        printTensorShape(conv2)

        lrn2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9, beta=0.75, name='lrn2')
        pool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')
        printTensorShape(pool2)

    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 256, 384], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.1, shape=[384], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope)
        printTensorShape(conv3)

    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 384, 384], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.1, shape=[384], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope)
        printTensorShape(conv4)

    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 384, 256], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.1, shape=[256], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope)
        printTensorShape(conv5)
        pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool5')
        printTensorShape(pool5)
    flattened = tf.reshape(pool5, shape=[-1, 256*6*6])
    printTensorShape(flattened)
    with tf.name_scope('fc6') as scope:
        weight = tf.Variable(tf.truncated_normal([256 * 6 * 6, 4096], dtype=tf.float32, stddev=1e-1), name='weight')
        biases = tf.Variable(tf.constant(0.1, shape=[1], dtype=tf.float32), trainable=True, name='biased')
        fc6 = tf.nn.relu(tf.add(tf.matmul(flattened, weight), biases), name=scope)
        printTensorShape(fc6)
        dropout6 = tf.nn.dropout(fc6, keep_prob=0.5, name='dropout1')

    with tf.name_scope('fc7') as scope:
        weight = tf.Variable(tf.truncated_normal([4096, 4096], dtype=tf.float32, stddev=1e-1), name='weight')
        biases = tf.Variable(tf.constant(0.1, shape=[4096], dtype=tf.float32), trainable=True, name='biased')
        fc7 = tf.nn.relu(tf.add(tf.matmul(dropout6, weight), biases), name=scope)
        printTensorShape(fc7)
        dropout7 = tf.nn.dropout(fc7, keep_prob=0.5, name='dropout2')

    with tf.name_scope('fc8') as scope:
        weight = tf.Variable(tf.truncated_normal([4096, 2], dtype=tf.float32, stddev=1e-1), name='weight')
        biases = tf.Variable(tf.constant(0.1, shape=[2], dtype=tf.float32), trainable=True, name='biased')
        fc8 = tf.add(tf.matmul(dropout7, weight), biases)
        printTensorShape(fc8)
    return fc8


def losses(logits, labels):
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels,
                                                                       name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name + '/loss', loss)
    return loss


def trainning(loss, var_list, learning_rate):
    with tf.name_scope('optimizer'):
        gradients = tf.gradients(loss, var_list)
        gradients = list(zip(gradients, var_list))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.apply_gradients(grads_and_vars=gradients)
    return train_op


def evaluation(logits, labels):
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name + '/accuracy', accuracy)
    return accuracy


def load_initial_weights(weight_path, keep_layers, session):
    weights_dict = np.load(weight_path, encoding='bytes').item()
    for op_name in weights_dict:
        # Check if the layer is one of the layers that should be reinitialized
        # if op_name not in self.SKIP_LAYER:
        if op_name not in keep_layers:
            with tf.variable_scope(op_name, reuse=tf.AUTO_REUSE):
                # Loop over list of weights/biases and assign them to their corresponding tf variable
                for data in weights_dict[op_name]:
                    # Biases
                    if len(data.shape) == 1:
                        var = tf.Variable('biases', trainable=False)
                        session.run(var.assign(data))
                    # Weights
                    else:
                        var = tf.Variable('weights', trainable=False)
                        session.run(var.assign(data))

# 定义AlexNet神经网络结构模型
# 建立模型图
class AlexNet(object):

    # keep_prob:dropout概率,num_classes:数据类别数,skip_layer
    def __init__(self, x, keep_prob, num_classes, skip_layer, weights_path='DEFAULT'):

        self.X = x
        self.NUM_CLASSES = num_classes
        self.KEEP_PROB = keep_prob
        self.SKIP_LAYER = skip_layer
        if weights_path == 'DEFAULT':
            self.WEIGHTS_PATH = './bvlc_alexnet.npy'
        else:
            self.WEIGHTS_PATH = weights_path

        self.create()

    def create(self):
        # 第一层：卷积层-->最大池化层-->LRN
        conv1 = conv_layer(self.X, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
        self.conv1 = conv1
        pool1 = max_pool(conv1, 3, 3, 2, 2, padding='VALID', name='pool1')
        norm1 = lrn(pool1, 2, 2e-05, 0.75, name='norml')

        # 第二层：卷积层-->最大池化层-->LRN
        conv2 = conv_layer(norm1, 5, 5, 256, 1, 1, groups=2, name='conv2')
        self.conv2 = conv2
        pool2 = max_pool(conv2, 3, 3, 2, 2, padding='VALID', name='pool2')
        norm2 = lrn(pool2, 2, 2e-05, 0.75, name='norm2')

        # 第三层：卷积层
        conv3 = conv_layer(norm2, 3, 3, 384, 1, 1, name='conv3')
        self.conv3 = conv3

        # 第四层：卷积层
        conv4 = conv_layer(conv3, 3, 3, 384, 1, 1, groups=2, name='conv4')
        self.conv4 = conv4

        # 第五层：卷积层-->最大池化层
        conv5 = conv_layer(conv4, 3, 3, 256, 1, 1, groups=2, name='conv5')
        self.conv5 = conv5
        pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')

        # 第六层：全连接层
        flattened = tf.reshape(pool5, [-1, 6 * 6 * 256])
        fc6 = fc_layer(flattened, 6 * 6 * 256, 4096, name='fc6')
        dropout6 = dropout(fc6, self.KEEP_PROB)

        # 第七层：全连接层
        fc7 = fc_layer(dropout6, 4096, 4096, name='fc7')
        dropout7 = dropout(fc7, self.KEEP_PROB)

        # 第八层：全连接层，不带激活函数
        self.fc8 = fc_layer(dropout7, 4096, self.NUM_CLASSES, relu=False, name='fc8')

    # 加载神经网络预训练参数,将存储于self.WEIGHTS_PATH的预训练参数赋值给那些没有在self.SKIP_LAYER中指定的网络层的参数
    def load_initial_weights(self, session):
        # 下载权重文件
        weights_dict = np.load(self.WEIGHTS_PATH, encoding='bytes').item()

        for op_name in weights_dict:
            if op_name not in self.SKIP_LAYER:
                with tf.variable_scope(op_name, reuse=True):
                    for data in weights_dict[op_name]:
                        # 偏置项
                        if len(data.shape) == 1:
                            var = tf.get_variable('biases', trainable=False)
                            session.run(var.assign(data))
                        # 权重
                        else:
                            var = tf.get_variable('weights', trainable=False)
                            session.run(var.assign(data))


# 定义卷积层，当groups=1时，AlexNet网络不拆分；当groups=2时，AlexNet网络拆分成上下两个部分。
def conv_layer(x, filter_height, filter_width, num_filters, stride_y, stride_x, name, padding='SAME', groups=1):
    # 获得输入图像的通道数
    input_channels = int(x.get_shape()[-1])

    # 创建lambda表达式
    convovle = lambda i, k: tf.nn.conv2d(i, k, strides=[1, stride_y, stride_x, 1], padding=padding)

    with tf.variable_scope(name) as scope:
        # 创建卷积层所需的权重参数和偏置项参数
        weights = tf.get_variable("weights", shape=[filter_height, filter_width, input_channels / groups, num_filters])
        biases = tf.get_variable("biases", shape=[num_filters])

    if groups == 1:
        conv = convovle(x, weights)

    # 当groups不等于1时，拆分输入和权重
    else:
        input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
        weight_groups = tf.split(axis=3, num_or_size_splits=groups, value=weights)
        output_groups = [convovle(i, k) for i, k in zip(input_groups, weight_groups)]
        # 单独计算完后，再次根据深度连接两个网络
        conv = tf.concat(axis=3, values=output_groups)

    # 加上偏置项
    bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
    # 激活函数
    relu = tf.nn.relu(bias, name=scope.name)

    return relu


# 定义全连接层
def fc_layer(x, num_in, num_out, name, relu=True):
    with tf.variable_scope(name) as scope:
        # 创建权重参数和偏置项
        weights = tf.get_variable("weights", shape=[num_in, num_out], trainable=True)
        biases = tf.get_variable("biases", [num_out], trainable=True)

        # 计算
        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

        if relu == True:
            relu = tf.nn.relu(act)
            return relu
        else:
            return act


# 定义最大池化层
def max_pool(x, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1], strides=[1, stride_y, stride_x, 1],
                          padding=padding, name=name)


# 定义局部响应归一化LPN
def lrn(x, radius, alpha, beta, name, bias=1.0):
    return tf.nn.local_response_normalization(x, depth_radius=radius, alpha=alpha, beta=beta, bias=bias, name=name)


# 定义dropout
def dropout(x, keep_prob):
    return tf.nn.dropout(x, keep_prob)


if __name__ == '__main__':
    image_size = 224
    images = tf.Variable(tf.random_normal([batch_size,
                                           image_size,
                                           image_size, 3],
                                          dtype=tf.float32,
                                          stddev=1e-1))
    fc8, parameters = inference(images)
