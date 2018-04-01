import tensorflow as tf
import numpy as np


class textCNN(object):

    def __init__(self,seq_len,voc_size,embed_dims,num_class,filter_sizes,num_channels,l2_reg=0.0):
        self.input_x=tf.placeholder(tf.int32,[None,seq_len],name='input_x')
        self.input_y=tf.placeholder(tf.float32,[None,num_class],name='labels')
        self.dropout_prob=tf.placeholder(tf.float32,name='dropout_prob')

        l2_loss=tf.constant(0.0)

        with tf.device('/cpu:0'),tf.name_scope('embeddings'):
            self.W=tf.Variable(tf.random_uniform([voc_size,embed_dims],-1,1),name='W')
            self.embeddings=tf.nn.embedding_lookup(self.W,self.input_x)
            self.ext_embeddings=tf.expand_dims(self.embeddings,-1)

        pooled_out=[]
        for i,sz in enumerate(filter_sizes):
            with tf.name_scope('conv-maxpool-%s'%sz):
                filter_shape = [sz, embed_dims, 1, num_channels]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                b = tf.Variable(tf.constant(0.1, shape=[num_channels]), name='b')
                conv = tf.nn.conv2d(self.ext_embeddings, W, strides=[1, 1, 1, 1], padding="VALID", name='conv')
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
                pool = tf.nn.max_pool(h, ksize=[1, seq_len - sz + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID',name='pool')
                pooled_out.append(pool)

        num_filters=num_channels*len(filter_sizes)
        self.h_pool=tf.concat(pooled_out,3)
        self.h_pool_flat=tf.reshape(self.h_pool,[-1,num_filters])

        with tf.name_scope('dropout'):
            self.drop=tf.nn.dropout(self.h_pool_flat,self.dropout_prob)

        with tf.name_scope('output'):
            W=tf.get_variable('W',shape=[num_filters,num_class],initializer=tf.contrib.layers.xavier_initializer())
            b=tf.constant(0.1,shape=[num_class],name='b')
            l2_loss+=tf.nn.l2_loss(W)
            l2_loss+=tf.nn.l2_loss(b)
            self.scores=tf.nn.xw_plus_b(self.drop,W,b)
            self.prediction=tf.argmax(self.scores,1,name='predict')

        with tf.name_scope('loss'):
            loss1=tf.nn.softmax_cross_entropy_with_logits(logits=self.scores,labels=self.input_y)
            self.loss=tf.reduce_mean(loss1)+l2_reg*l2_loss

        with tf.name_scope('accuracy'):
            correct_prediction=tf.equal(self.prediction,tf.argmax(self.input_y,1))
            self.accuracy=tf.reduce_mean(tf.cast(correct_prediction,'float'))

