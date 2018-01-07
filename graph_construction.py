import tensorflow as tf
import numpy as np

def weight_variable(name, shape):
    """

    """
    return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
    
def bias_variable(shape):
    """

    """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    """

    """
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    """

    """
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def classifier(learning_rate=0.0001):
    """
    
    """
    # Parameters
    training_epochs = 15
    batch_size = 50
    display_step = 1
    
    keep_prob = tf.placeholder(tf.float32)
    
    x = tf.placeholder(tf.float32, [None, 64, 256, 1])
    y = tf.placeholder(tf.int32, [None, 2])
      
    # Model
    W_conv1 = weight_variable('W1', [3, 3, 1, 64])
    b_conv1 = bias_variable([64])
        
    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    
    W_conv2 = weight_variable('W2', [3, 3, 64, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_conv3 = weight_variable('W3', [3, 3, 64, 64])
    b_conv3 = bias_variable([64])

    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

    W_conv4 = weight_variable('W4', [3, 3, 64, 64])
    b_conv4 = bias_variable([64])
    
    h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)

    W_conv5 = weight_variable('W5', [3, 3, 64, 64])
    b_conv5 = bias_variable([64])
    
    h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5) + b_conv5)
    
    W_fc1 = weight_variable('W6', [16 * 64 * 64, 512])
    b_fc1 = bias_variable([512])

    h_pool2_flat = tf.reshape(h_conv5, [-1, 16 * 64 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    W_fc2 = weight_variable('W7', [512, 512])
    b_fc2 = bias_variable([512])
    
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    
    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)
   
    W_fc5 = weight_variable('W8', [512, 2])
    b_fc5 = bias_variable([2])
    
    y_conv = tf.matmul(h_fc2_drop, W_fc5) + b_fc5
    
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv))
    
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    return training_epochs, batch_size, display_step, x, y, keep_prob, y_conv, train_op, accuracy

def classifier_trivial(learning_rate=0.0001, use_dropout=True):
    """
    
    """
  # Parameters
    training_epochs = 200
    batch_size = 100
    display_step = 1
    
    x = tf.placeholder(tf.float32, [None, 64, 256])
    y = tf.placeholder(tf.float32, [None, 2])
    a = 64*256
    h = tf.reshape(x, [-1, a])
    
    W = tf.Variable(tf.zeros([a, 2]))
    b = tf.Variable(tf.zeros([2]))
    
    y_ = tf.matmul(h, W) + b
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_)
    
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    return training_epochs, batch_size, display_step, x, y, y_, train_op, accuracy