import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("D:\ML Datasets\MNIST", one_hot=True)

X_train, Y_train = mnist.train.next_batch(5000)
X_test, Y_test = mnist.train.next_batch(200)

x_train = tf.placeholder(tf.float32, [None, 784])
x_test = tf.placeholder(tf.float32, [784])

distance = tf.reduce_sum(tf.abs(tf.add(x_train, tf.negative(x_test))), reduction_indices=1)
pred = tf.arg_min(distance, 0)

accuracy = 0.
init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	
	for i in range(len(X_test)):
		#get nearest neighbor
		nearest_neighbor_index = sess.run(pred, feed_dict={x_train:X_train, x_test:X_test[i, :]})
		print("Test {}, Prediction: {}, True Class: {}".format(i,np.argmax(Y_train[nearest_neighbor_index]), np.argmax(Y_test[i])))
		if np.argmax(Y_train[nearest_neighbor_index]) == np.argmax(Y_test[i]):
			accuracy += 1.0/len(X_test)
		print("Done!")
		print(accuracy)



