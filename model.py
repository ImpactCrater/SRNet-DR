#! /usr/bin/python
# -*- coding: utf8 -*-

import time
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

# from tensorflow.python.ops import variable_scope as vs
# from tensorflow.python.ops import math_ops, init_ops, array_ops, nn
# from tensorflow.python.util import nest
# from tensorflow.contrib.rnn.python.ops import core_rnn_cell


def Generator(t_image, is_train=False, reuse=False):
    w_init = tf.random_normal_initializer(stddev=0.0001)
    b_init = None # tf.constant_initializer(value=0.0)
    n_channel = 128
    swish = lambda x: tf.nn.swish(x)
    with tf.compat.v1.variable_scope("Generator", reuse=reuse) as vs:
        n = InputLayer(t_image, name='in')
        n = Conv2d(n, n_channel, (3, 3), (1, 1), act=swish, padding='SAME', W_init=w_init, name='c0')
        n = GroupNormLayer(n, groups=16, act=None, name='gn0')

        # residual in residual blocks
        temp1 = n
        for j in range(8):
            temp0 = n
            for i in range(16):
                nn = Conv2d(n, n_channel * 2, (3, 3), (1, 1), act=swish, padding='SAME', W_init=w_init, b_init=b_init, name='res_c0/%s_%s' % (j, i))
                nn = Conv2d(nn, n_channel, (3, 3), (1, 1), act=swish, padding='SAME', W_init=w_init, b_init=b_init, name='res_c1/%s_%s' % (j, i))
                nn = ElementwiseLayer([n, nn], tf.add, name='res_add0/%s_%s' % (j, i))
                n = nn

            n = Conv2d(n, n_channel, (3, 3), (1, 1), act=swish, padding='SAME', W_init=w_init, b_init=b_init, name='res_c2/%s' % (j))
            n = ElementwiseLayer([temp0, n], tf.add, name='res_add1/%s' % (j))

        n = Conv2d(n, n_channel, (3, 3), (1, 1), act=swish, padding='SAME', W_init=w_init, b_init=b_init, name='res_c3')
        n = ElementwiseLayer([temp1, n], tf.add, name='res_add2')

        # residual in residual blocks end

        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='c1')
        n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=swish, name='pixelshufflerx2/1')

        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='c2')
        n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=swish, name='pixelshufflerx2/2')

        n = Conv2d(n, 3, (1, 1), (1, 1), act=tf.nn.tanh, padding='SAME', W_init=w_init, name='out')
        return n
