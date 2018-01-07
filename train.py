# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A deep MNIST classifier using convolutional layers.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""

import tensorflow as tf
from preprocess import *
from PIL import Image

NUM_POOLS = 5
BN_EPSILON = 0.001
batch_size = 50
learning_rate = 5e-3
epochs = 30000


def deepnn(x, phase_train, keep_prob):

	'''Convolution -> Batch Normalization -> Relu Layer'''
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([3, 3, 3, 64])
        b_conv1 = bias_variable([64])
        h_conv1 = conv_bn_relu(x, W_conv1, b_conv1, 64, phase_train)

    with tf.name_scope('dropout1'):
        h_conv1_drop = tf.nn.dropout(h_conv1, keep_prob)

    '''Convolution -> Batch Normalization -> Relu Layer'''
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([3, 3, 64, 128])
        b_conv2 = bias_variable([128])
        h_conv2 = conv_bn_relu(h_conv1_drop, W_conv2, b_conv2, 128, phase_train)

    '''Pooling layer - Downsamples image by 50%'''
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv2)

    with tf.name_scope('dropout2'):
        h_pool1_drop = tf.nn.dropout(h_pool1, keep_prob)
    
    '''Convolution -> Batch Normalization -> Relu Layer'''
    with tf.name_scope('conv3'):
        W_conv3 = weight_variable([3, 3, 128, 128])
        b_conv3 = bias_variable([128])
        h_conv3 = conv_bn_relu(h_pool1_drop, W_conv3, b_conv3, 128, phase_train)

    with tf.name_scope('dropout3'):
        h_conv3_drop = tf.nn.dropout(h_conv3, keep_prob)

    '''Convolution -> Pool -> Batch Normalization -> Relu Layer'''
    #Pooling downsamples by %
    with tf.name_scope('conv4'):
        W_conv4 = weight_variable([3, 3, 128, 128])
        b_conv4 = bias_variable([128])
        h_conv4 = conv2d(h_conv3_drop, W_conv4) + b_conv4
    with tf.name_scope('pool2'):
        h_pool2 = tf.nn.relu(batch_normalization_layer(max_pool_2x2(h_conv4), 128, phase_train))

    with tf.name_scope('dropout4'):
        h_pool2_drop = tf.nn.dropout(h_pool2, keep_prob)

    '''Convolution -> Batch Normalization -> Relu Layer'''
    with tf.name_scope('conv5'):
        W_conv5 = weight_variable([3, 3, 128, 128])
        b_conv5 = bias_variable([128])
        h_conv5 = conv_bn_relu(h_pool2_drop, W_conv5, b_conv5, 128, phase_train)

    with tf.name_scope('dropout5'):
        h_conv5_drop = tf.nn.dropout(h_conv5, keep_prob)

    '''Pooling layer - Downsamples by 50%'''
    with tf.name_scope('pool3'):
        h_pool3 = max_pool_2x2(h_conv5_drop)

    '''Convolution -> Batch Normalization -> Relu Layer'''
    with tf.name_scope('conv6'):
        W_conv6= weight_variable([3, 3, 128, 128])
        b_conv6 = bias_variable([128])
        h_conv6 = conv_bn_relu(h_pool3, W_conv6, b_conv6, 128, phase_train)

    with tf.name_scope('dropout6'):
        h_conv6_drop = tf.nn.dropout(h_conv6, keep_prob)

    '''Convolution -> Batch Normalization -> Relu Layer'''
    with tf.name_scope('conv7'):
        W_conv7 = weight_variable([1, 1, 128, 128])
        b_conv7 = bias_variable([128])
        h_conv7 = conv_bn_relu(h_conv6_drop, W_conv7, b_conv7, 128, phase_train)

    with tf.name_scope('dropout7'):
        h_conv7_drop = tf.nn.dropout(h_conv7, keep_prob)

    '''Pooling layer - Downsamples by %'''
    with tf.name_scope('pool4'):
        h_pool4 = max_pool_2x2(h_conv7_drop)

    with tf.name_scope('dropout8'):
        h_pool4_drop = tf.nn.dropout(h_pool4, keep_prob)

    '''Convolution -> Batch Normalization -> Relu Layer'''
    with tf.name_scope('conv8'):
        W_conv8 = weight_variable([3, 3, 128, 128])
        b_conv8 = bias_variable([128])
        h_conv8 = conv_bn_relu(h_pool4_drop, W_conv8, b_conv8, 128, phase_train)

    '''Pooling layer - downsamples by 50% '''
    with tf.name_scope('pool5'):
        h_pool5 = max_pool_2x2(h_conv8)

    '''Fully connected layer'''
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([(IMAGE_SIZE/(2**NUM_POOLS)) *(IMAGE_SIZE/(2**NUM_POOLS)) * 128, 1024])
        b_fc1 = bias_variable([1024])
        h_pool5_flat = tf.reshape(h_pool5, [-1, (IMAGE_SIZE/(2**NUM_POOLS)) *(IMAGE_SIZE/(2**NUM_POOLS)) * 128]) #Flatten the 4D output of pool5 into a 2D tensor 
        h_fc1 = tf.nn.relu(tf.matmul(h_pool5_flat, W_fc1) + b_fc1)

    with tf.name_scope('dropout10'):
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    '''Dot product layer'''
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([1024, 25])
        b_fc2 = bias_variable([25])
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    return y_conv, keep_prob, phase_train

'''Wrapper function for Convolution -> Batch Normalization -> ReLu Layers'''
def conv_bn_relu(input, weight, bias, out_channel, phase_train):
    return tf.nn.relu(batch_normalization_layer(conv2d(input, weight) + bias, out_channel, phase_train))

'''Batch Normalization function'''
def batch_normalization_layer(input_layer, dimension, phase_train):

    mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
    beta = tf.Variable(tf.constant(0.0, shape = [dimension]), name ='beta', trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape = [dimension]), name='gamma', trainable=True)
    
    bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, BN_EPSILON)
    return bn_layer

#Wrapper for Convlution function with input tensor x, filter size W, stride = 1 and SAME padding
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#Wrapper for Pooling function with 2x2 kernal, stride = 2 and SAME padding
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

#Creates a weight variable with shape = shape gaussian initialization
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

#Creates bias variable with shape = shape and constant initiialization 
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def main():
	#Load data using Preprocess.py
    train_x, vali_x, train_shapes, vali_shapes, train_y, vali_y = load_data()        
    print "Data Loaded!"
 
    global learning_rate

    # Create the model
    x = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, 3], name='input')
    y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES], name='labels')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    phase_train = tf.placeholder(tf.bool, name='phase_train' )
    learningrate = tf.placeholder(tf.float32, name='learning_rate')

    # Build the graph for the deep net
    y_conv, keep_prob, phase_train = deepnn(x, phase_train, keep_prob)

    #Loss function - Cross entropy
    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_conv)
        cross_entropy = tf.reduce_mean(cross_entropy)

    #Learning optimizer
    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(learningrate).minimize(cross_entropy)

    #Accuracy calculator
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)

    #Returns the predicted class
    with tf.name_scope('final_answer'):
        final_answer = tf.argmax(y_conv, 1)

    #Tensorboard Summary Writing
    step =  tf.Variable(0, trainable=False, name='step')
    ema1 = tf.train.ExponentialMovingAverage(0.0 , step)
    ema2 = tf.train.ExponentialMovingAverage(0.95, step)    
    val_op = tf.group(step.assign_add(1), ema1.apply([cross_entropy, accuracy]), ema2.apply([cross_entropy, accuracy]))

    training_loss = ema1.average(cross_entropy)
    vali_accuracy = ema1.average(accuracy)
    training_loss_avg = ema2.average(cross_entropy)
    vali_accuracy_avg = ema2.average(accuracy)
    
    tf.summary.scalar('learning_rate' , learningrate)
    tf.summary.scalar('training loss' , training_loss) 
    tf.summary.scalar('average training loss' , training_loss_avg)
    tf.summary.scalar('validation accuracy' , vali_accuracy)
    tf.summary.scalar('average validation accuracy', vali_accuracy_avg)

    merged = tf.summary.merge_all()

    #Saves GPU memory because tensorflow allocates entire gpu memory by default
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.5

    #Initializes saver object for saving the model for later use without training
    saver = tf.train.Saver(var_list=tf.trainable_variables())

    #Running tensorflow session
    with tf.Session(config = config) as sess:
    	#Initialize variables
        sess.run(tf.global_variables_initializer())
        #Initialize summary writer object
        summary_writer = tf.summary.FileWriter("./logs", sess.graph)

        #Start training
        for i in xrange(epochs):

        	#Show accuracy and write summaries at set number of epochs
            if i % show_steps == 0:
                vali_batch_x, vali_batch_y = batch(batch_size, vali_x, vali_shapes, vali_y, 'vali')  
                train_accuracy, summary, _  = sess.run([accuracy, merged, val_op], feed_dict=\
                	{x: vali_batch_x, y_: vali_batch_y, keep_prob: 1.0, phase_train:False, learningrate : learning_rate})

                summary_writer.add_summary(summary, i) 
                print('step %d, training accuracy %g' % (i, train_accuracy))

            #Training
            batch_x, batch_y = batch(batch_size, train_x, train_shapes, train_y)
            train_step.run(feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5, phase_train:True, learningrate:learning_rate})

        #Final test using unseen images
        test_x, test_shapes, test_labels = load_test_data()  
        test_batch_x, test_batch_y = test_batch(batch_size, test_x, test_shapes, test_labels)
 		#Save model
        save_path = saver.save(sess, "./model.ckpt")

main()
