from keras import backend as K
import tensorflow as tf
from keras.engine.topology import Layer
import numpy as np
from keras.layers import Lambda

class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(** kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) == 2
        # W.shape = (time_steps, time_steps),16=f表示item的潜特征维度
        self.gamma_j = self.add_weight(name='gamma_j',
                                       shape=(input_shape[1], 8),
                                       initializer='uniform',
                                       trainable=True)
        self.Wa = self.add_weight(name='att_weight',
                                  shape=(8, 8),
                                  initializer='uniform',
                                  trainable=True)
        self.ba0 = self.add_weight(name='att_bias',
                                   shape=(8, ),
                                   initializer='uniform',
                                   trainable=True)
        self.ba1 = self.add_weight(name='att_bias1',
                                   shape=(1, ),
                                   initializer='uniform',
                                   trainable=True)
        self.h = self.add_weight(name='h',
                                 shape=(8, 1),
                                 initializer='uniform',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    # 除非你希望你写的层支持masking，否则你只需要关心call的第一个参数：输入张量。
    def call(self, inputs, mask=None):
        c = K.relu(K.dot(self.gamma_j,self.Wa)+self.ba0)
        s = K.dot(c, self.h) + self.ba1
        s = tf.reduce_sum(s, axis=1)
        user_score =inputs * s
        inf = tf.constant(value=-np.inf, name="numpy_inf")
        user_mask_inf = tf.where(tf.equal(inputs, tf.zeros_like(inputs)),
                                 tf.ones_like(inputs) * inf, inputs) - tf.constant(1.)
        user_score_mask = user_score + user_mask_inf
        print('user_score_mask',user_score_mask.shape)
        print('user_score',user_score)
        fenmu = tf.pow(tf.reduce_sum(tf.exp(user_score), axis=1), 1.0)
        fenmu = tf.expand_dims(fenmu, axis=1)
        user_score_soft = tf.exp(user_score) / fenmu
        print(user_score_soft.shape)
        # user_att_embedding = K.sum(K.dot(user_score_soft, self.gamma_j),axis =1)
        user_att_embedding=tf.matmul(user_score_soft,self.gamma_j)
        print('self.gamma_j',self.gamma_j.shape)
        print(user_att_embedding.shape)
        return user_att_embedding


    def compute_output_shape(self, input_shape):
        return input_shape[0], 8

