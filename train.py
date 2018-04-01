import tensorflow as tf
import numpy as np
from data_preprocess import *
from cnn_model import textCNN
import os,time,datetime
from tensorflow.contrib import learn

ps_file="./data/rt-polarity.pos"
ng_file="./data/rt-polarity.neg"
embed_dim=128
filter_sizes=[3,4,5]
num_filters=128
dropout_prob=0.5
l2_reg=0.5
test_len=0.1
batch_size=64
num_epochs=200
eval_every=100
check_every=100
num_check=5


text,y=load_data(ps_file,ng_file)
max_len=max([len(s.split(" ")) for s in text])
vocab_pre_object=learn.preprocessing.VocabularyProcessor(max_len)
x=np.array(list(vocab_pre_object.fit_transform(text)))

np.random.seed(10)
shuffl_indices=np.random.permutation(np.arange(len(y)))
x_shuffled=x[shuffl_indices]
y_shuffled=y[shuffl_indices]

test_idx=-1*int(test_len*len(y))
x_train,x_test=x_shuffled[:test_idx],x_shuffled[test_idx:]
y_train,y_test=y_shuffled[:test_idx],y_shuffled[test_idx:]

with tf.Graph().as_default():
    sess=tf.Session()
    with sess.as_default():
        cnn=textCNN(x_train.shape[1],len(vocab_pre_object.vocabulary_),embed_dim,y_train.shape[1],filter_sizes,num_filters,l2_reg)
        global_step=tf.Variable(0,name='global_step',trainable=False)
        optimizer=tf.train.AdamOptimizer(1e-3)
        grads_and_vars=optimizer.compute_gradients(cnn.loss)
        train_step=optimizer.apply_gradients(grads_and_vars,global_step=global_step)

        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        train_summary_op = tf.summary.merge([loss_summary, acc_summary])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir,sess.graph)

        test_summary_op = tf.summary.merge([loss_summary, acc_summary])
        test_summary_dir = os.path.join(out_dir, "summaries", "dev")
        test_summary_writer = tf.summary.FileWriter(test_summary_dir, sess.graph)

        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=num_check)

        vocab_pre_object.save(os.path.join(out_dir, "vocab"))
        sess.run(tf.global_variables_initializer())

        def trainstep(x_batch,y_batch):
            feed_dict={
                cnn.input_x:x_batch,
                cnn.input_y:y_batch,
                cnn.dropout_prob:dropout_prob
            }


            _,step,summaries,loss,accuracy=sess.run([train_step,global_step,train_summary_op,cnn.loss,cnn.accuracy],feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)


        def eval(x_batch,y_batch,flag=None):
            feed_dict={
                cnn.input_x:x_batch,
                cnn.input_y:y_batch,
                cnn.dropout_prob:1.0
            }
            step,summaries,loss,accuracy=sess.run([global_step,test_summary_op,cnn.loss,cnn.accuracy],feed_dict)

            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if flag:
                flag.add_summary(summaries, step)


        batches=batch_iter(list(zip(x_train,y_train)),batch_size,num_epochs)

        for batch in batches:
            x_batch,y_batch=zip(*batch)
            trainstep(x_batch,y_batch)
            cur_step=tf.train.global_step(sess,global_step)
            if cur_step%eval_every==0:
                print("\nEvaluation:")
                eval(x_test,y_test,flag=test_summary_writer)
            if cur_step%check_every==0:
                path=saver.save(sess,checkpoint_prefix,global_step=cur_step)

