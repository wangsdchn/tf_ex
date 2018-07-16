#!usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:WSD 
@file: googLeNet.py 
@time: 1:01 PM 
"""
import tensorflow as tf

slim = tf.contrib.slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)


def inception_v3_arg_scope(weight_dacay=0.0004, stddev=0.1, batch_norm_var_collection='moving_vars'):
    batch_norm_params = {
        'decay': 0.9997,
        'epsilon': 0.001,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        'variables_collections': {
            'beta': None,
            'gamma': None,
            'moving_mean': [batch_norm_var_collection],
            'moving_variance': [batch_norm_var_collection]
        }
    }

    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_regularizer=slim.l2_regularizer(weight_dacay)):
        with slim.arg_scope(
                [slim.conv2d],
                weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm,
                normalizer_params=batch_norm_params) as sc:
            return sc


def inception_v3_base(inputs, scope=None):  # input 299*299*3
    end_points = {} # 用来保存某些关键点供之后使用
    with tf.variable_scope(scope, 'InceptionV3', [inputs]):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='VALID'):
            net = slim.conv2d(input, 32, [3, 3], stride=2, scope='Conv2d_1a_3*3')   # (299-3)/2+1=149 149*194*32
            net = slim.conv2d(net, 32, [3, 3], scope='Conv2d_2a_3*3')               # (149-3)/1+1=147 147*147*32
            net = slim.conv2d(net, 64, [3, 3], padding='SAME', scope='Conv2d_3a_3*3')   # 147/1 147*147*64
            net = slim.max_pool2d(net, [3, 3], stride=2, scope='MaxPool_3a_3*3')        # (147-3)/2+1=73 73*73*64
            net = slim.conv2d(net, 80, [3, 3], scope='Conv2d_3b_3*3')                   # (73-3)/1+1=71 71*71*80
            net = slim.conv2d(net, 192, [3, 3], stride=2, scope='Conv2d_4a_3*3')        # (71-3)/2+1=35 35*35*192
            net = slim.max_pool2d(net, [3, 3], padding='SAVE', scope='MaxPool_5a_3*3')        # 35/1 35*35*192

    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAVE'):
        # 第一个模块组
        with tf.variable_scope('Mixed_5b'):
            with tf.variable_scopw('Branch_0'):
                branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1*1')
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0a_1*1')
                branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope='Conv2d_0b_5*5')
            with tf.variable_scope('Branch_3'):
                branch_2 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1*1')
                branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3*3')
                branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_3*3')
            with tf.variable_scope('Branch_3'):
                branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3*3')
                branch_3 = slim.conv2d(branch_3, 32, [1, 1], scope='Conv2d_0b_1*1')
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
        with tf.variable_scope('Mixed_5c'):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1*1')
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0a_1*1')
                branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope='Conv2d_0b_5*5')
            with tf.variable_scope('Branch_2'):
                branch_2 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_3*3')
                branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3*3')
                branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_3*3')
            with tf.variable_scope('Branch_3'):
                branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3*3')
                branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1*1')
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

        with tf.variable_scope('Mixed_5d'):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1*1')
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0a_1*1')
                branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope='Conv2d_0b_5*5')
            with tf.variable_scope('Branch_2'):
                branch_2 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1*1')
                branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3*3')
                branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='COnv2d_0c_3*3')
            with tf.varible_scope("Branch_3"):
                branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3*3')
                branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope="Conv2d_0b_1*1")
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

        # 第二个模块组
        with tf.variable_scope('Mixed_6a'):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(net, 384, [3, 3], stride=2, padding='VALID', scope='Conv2d_1a_3*3')
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_1a_1*1')
                branch_1 = slim.conv2d(branch_1, 96, [3, 3], scope='Conv2d_1b_3*3')
                branch_1 = slim.conv2d(branch_1, 96, [3, 3], stride=2, padding='VALID', scope='Conv2d_1c_3*3')
            with tf.varible_scope("Branch_3"):
                branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='maxPool_1a_3*3')
            net = tf.concat([branch_0, branch_1, branch_2], 3)

        with tf.variable_scope('Mixed_6b'):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_1a_1*1')
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_1a_1*1')
                branch_1 = slim.conv2d(branch_1, 128, [1, 7], scope='Conv2d_1b_1*7')
                branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_1b_7*1')
            with tf.variable_scope('Branch_2'):
                branch_2 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_1a_1*1')
                branch_2 = slim.conv2d(branch_2, 128, [7, 1], scope='Conv2d_1b_7*1')
                branch_2 = slim.conv2d(branch_2, 128, [1, 7], scope='COnv2d_1c_1*7')
                branch_2 = slim.conv2d(branch_2, 128, [7, 1], scope='COnv2d_1c_7*1')
                branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='COnv2d_1c_1*7')
            with tf.varible_scope("Branch_3"):
                branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_1a_3*3')
                branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope="Conv2d_1b_1*1")
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

        with tf.variable_scope('Mixed_6c'):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_1a_1*1')
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_1a_1*1')
                branch_1 = slim.conv2d(branch_1, 160, [1, 7], scope='Conv2d_1b_1*7')
                branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_1b_7*1')
            with tf.variable_scope('Branch_2'):
                branch_2 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_1a_1*1')
                branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_1b_7*1')
                branch_2 = slim.conv2d(branch_2, 160, [1, 7], scope='COnv2d_1c_1*7')
                branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='COnv2d_1c_7*1')
                branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='COnv2d_1c_1*7')
            with tf.varible_scope("Branch_3"):
                branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_1a_3*3')
                branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope="Conv2d_1b_1*1")
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

        with tf.variable_scope('Mixed_6d'):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_1a_1*1')
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_1a_1*1')
                branch_1 = slim.conv2d(branch_1, 160, [1, 7], scope='Conv2d_1b_1*7')
                branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_1b_7*1')
            with tf.variable_scope('Branch_2'):
                branch_2 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_1a_1*1')
                branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_1b_7*1')
                branch_2 = slim.conv2d(branch_2, 160, [1, 7], scope='COnv2d_1c_1*7')
                branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='COnv2d_1c_7*1')
                branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='COnv2d_1c_1*7')
            with tf.varible_scope("Branch_3"):
                branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_1a_3*3')
                branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope="Conv2d_1b_1*1")
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

        with tf.variable_scope('Mixed_6e'):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_1a_1*1')
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_1a_1*1')
                branch_1 = slim.conv2d(branch_1, 160, [1, 7], scope='Conv2d_1b_1*7')
                branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_1b_7*1')
            with tf.variable_scope('Branch_2'):
                branch_2 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_1a_1*1')
                branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_1b_7*1')
                branch_2 = slim.conv2d(branch_2, 160, [1, 7], scope='COnv2d_1c_1*7')
                branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='COnv2d_1c_7*1')
                branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='COnv2d_1c_1*7')
            with tf.varible_scope("Branch_3"):
                branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_1a_3*3')
                branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope="Conv2d_1b_1*1")
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

            # 保留
            end_points['Mixed_6e'] = net

        # 第三个模块
        with tf.variable_scope('Mixed_7a'):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_1a_1*1')
                branch_0 = slim.conv2d(branch_0, 320, [3, 3], stride=2, padding='VALID', scope='Conv2d_1a_1*1')
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_1a_1*1')
                branch_1 = slim.conv2d(branch_1, 192, [1, 7], scope='Conv2d_1b_1*7')
                branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_1b_7*1')
                branch_1 = slim.conv2d(branch_1, 192, [3, 3], stride=2, padding='VALID', scope='Conv2d_1a_1*1')
            with tf.varible_scope("Branch_2"):
                branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='maxPool_1a_3*3')
            net = tf.concat([branch_0, branch_1, branch_2], 3)

        with tf.variable_scope('Mixed_7b'):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(net, 320, [1, 1], scope='Conv2d_1a_1*1')
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(net, 384, [1, 1], scope='Conv2d_1a_1*1')
                branch_1 = tf.concat([slim.conv2d(branch_1, 384, [1, 3], scope='Conv2d_1b_1*3'),
                                      slim.conv2d(branch_1, 384, [3, 1], scope='Conv2d_1b_3*1')], 3)
            with tf.variable_scope('Branch_2'):
                branch_2 = slim.conv2d(net, 448, [1, 1], scope='Conv2d_1a_1*1')
                branch_2 = slim.conv2d(branch_2, 384, [3, 3], scope='Conv2d_1a_1*1')
                branch_2 = tf.concat([slim.conv2d(branch_2, 384, [1, 3], scope='Conv2d_1b_1*3'),
                                      slim.conv2d(branch_2, 384, [3, 1], scope='Conv2d_1b_3*1')], 3)
            with tf.varible_scope("Branch_3"):
                branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_1a_3*3')
                branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope="Conv2d_1b_1*1")
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

        with tf.variable_scope('Mixed_7c'):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(net, 320, [1, 1], scope='Conv2d_1a_1*1')
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(net, 384, [1, 1], scope='Conv2d_1a_1*1')
                branch_1 = tf.concat([slim.conv2d(branch_1, 384, [1, 3], scope='Conv2d_1b_1*3'),
                                      slim.conv2d(branch_1, 384, [3, 1], scope='Conv2d_1b_3*1')], 3)
            with tf.variable_scope('Branch_2'):
                branch_2 = slim.conv2d(net, 448, [1, 1], scope='Conv2d_1a_1*1')
                branch_2 = slim.conv2d(branch_2, 384, [3, 3], scope='Conv2d_1a_1*1')
                branch_2 = tf.concat([slim.conv2d(branch_2, 384, [1, 3], scope='Conv2d_1b_1*3'),
                                      slim.conv2d(branch_2, 384, [3, 1], scope='Conv2d_1b_3*1')], 3)
            with tf.varible_scope("Branch_3"):
                branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_1a_3*3')
                branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope="Conv2d_1b_1*1")
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

        return net, end_points

def inception_v3(inputs, num_classes=1000, is_training=True, dropout_keep_prob=0.8, prediction_fn=slim.softmax,
                 spatial_squeeze=True, reuse=None, scope='InceptionV3'):
    with tf.variable_scope(scope, "inceptionV3", [inputs, num_classes], reuse=reuse) as scope:
        # 前向计算
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            net, end_points = inception_v3_base(inputs, scope=scope)
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
            aux_logits = end_points['Mixed_6e']

            with tf.variable_scope('AuxLogits'):
                aux_logits = slim.avg_pool2d(aux_logits, [5, 5], stride=3, padding='VALID', scope='AvgPool_1a_5*5')
                aux_logits = slim.conv2d(aux_logits, 128, [1, 1], scope='conv2d_1b_1x1')

                aux_logits = slim.conv2d(aux_logits, 768, [5, 5], weights_initializer=trunc_normal(0.01),
                                         padding="VALID", scope='conv2d_2a_5x5')  # 输出1×1×768
                aux_logits = slim.conv2d(aux_logits, num_classes, [1, 1], activation_fn=None,
                                         weights_initializer=trunc_normal(0.01),
                                         padding="VALID", scope='conv2d_2b_1x1')  # 输出1×1×num_classes
                if spatial_squeeze:
                    aux_logits = tf.squeeze(aux_logits, [1, 2], name='SpatialSqueeze')  # 注2
                end_points["AuxLogits"] = aux_logits

            # 下面处理正常的分类预测逻辑logits
            with tf.variable_scope("Logits"):
                # 对Mixed_7e(z最后一层的输出)进行平均池化
                net = slim.avg_pool2d(net, [8, 8], padding="VALID", scope="AvgPool_1a_8x8")  # 1*1*2048
                # dropout
                net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_1b')
                end_points['prelogits'] = net
                logits = slim.conv2d(aux_logits, num_classes, [1, 1], activation_fn=None, normalizer_fn=None,
                                     scope='conv2d_1c_1x1')
                if spatial_squeeze:
                    logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')

                end_points['Logits'] = logits
                end_points['Predictions'] = prediction_fn(logits, scope='Predictions')  # softmax

            return logits, end_points
