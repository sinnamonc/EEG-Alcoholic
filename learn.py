import tensorflow as tf
import numpy as np
import time
from graph_construction import classifier_trivial
from graph_construction import classifier
from data_preprocessing import get_iterators
import matplotlib.pyplot as plt
from IPython import display

def acc_plot(train_acc, val_acc, training_epochs, display_step):
    """Clear the current figure and display the training and validation accuracy.
    
    :input train_acc: the training accuracy array
    :input val_acc: the validation accuracy array
    :input training_epochs: the total number of epochs that will be trained over
    :input display_step: the plot will only contain the accuracy values every display_step epochs
    """
    plt.ion()
    # Set up plot
    plt.plot(train_acc, 'C1')
    plt.plot(val_acc, 'C2')
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.ylim(0, 110)
    plt.xlabel('epoch / {}'.format(display_step))
    x_max = np.ceil(training_epochs / display_step)
    plt.xlim(0, x_max)
    plt.legend(['train', 'validation'], loc='lower right')
    display.display(plt.gcf())
    display.clear_output(wait=True)
    plt.draw()

def train_model():
    """
    
    """
    (training_epochs, batch_size, display_step, x, y, keep_prob,
     y_, train_step, accuracy) = classifier(learning_rate=0.0001)

    train_iterator, val_iterator, test_iterator = get_iterators(batch_size)
    
    next_train_batch = train_iterator.get_next()
    next_val_batch = val_iterator.get_next()
    next_test_batch = test_iterator.get_next()
    
    # Initialize the variables
    init = tf.global_variables_initializer()

    train_acc = []
    val_acc = []

    with tf.Session() as sess:
        # Run the initializer
        sess.run(init)

        # Training cycle
        for i in range(training_epochs):
            sess.run(train_iterator.initializer)
            while True:
                try:
                    batch = sess.run(next_train_batch)
                    sess.run(train_step, feed_dict={x: batch['feature'],
                                                    y: batch['label'],
                                                    keep_prob: 0.3})
                except tf.errors.OutOfRangeError:
                    break
            if i % display_step == 0:
                # Evaluate accuracy on a batch of training data
                sess.run(train_iterator.initializer)
                batch = sess.run(next_train_batch)
                train_acc.append(100*sess.run(accuracy, feed_dict={x: batch['feature'],
                                                                   y: batch['label'],
                                                                   keep_prob: 1.0}))
                # Evaluate accuracy on a batch of validation data
                sess.run(val_iterator.initializer)
                batch = sess.run(next_val_batch)
                val_acc.append(100*sess.run(accuracy, feed_dict={x: batch['feature'],
                                                                 y: batch['label'],
                                                                 keep_prob: 1.0}))
                acc_plot(train_acc, val_acc, training_epochs, display_step)
        print("Training Complete.")
        
        test_acc = []
        sess.run(test_iterator.initializer)
        while True:
            try:
                batch = sess.run(next_test_batch)
                test_acc.append(100*sess.run(accuracy, feed_dict={x: batch['feature'],
                                                                  y: batch['label'],
                                                                  keep_prob: 1.0}))
            except tf.errors.OutOfRangeError:
                break
        mean_test_acc = np.mean(test_acc)
        print('Test accuracy = {}%'.format(mean_test_acc))