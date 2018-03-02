import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST/",one_hot=True)


def add_layer(input_,in_size,out_size,activation_function=None):
	weights =tf.Variable(tf.random_normal([in_size,out_size]))
	biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)
	y = tf.matmul(input_,weights) + biases
	if activation_function is None:
		outputs = y
	else:
		outputs = activation_function(y)

	return outputs

def compute_accuracy(v_x_axes,v_y_axes):
	global output_layer
	y_prediction =sess.run(output_layer,feed_dict={x_axes:v_x_axes})
	correct_prediction =tf.equal(tf.argmax(y_prediction,1),tf.argmax(v_y_axes,1))
	accuracy =tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
	result = sess.run(accuracy,feed_dict={x_axes:v_x_axes,y_axes:v_y_axes})
	return result


# define placeholder for input of Network
x_axes = tf.placeholder(tf.float32,[None,784])
y_axes = tf.placeholder(tf.float32,[None,10])

# add output layer
output_layer = add_layer(x_axes,784,10,activation_function=tf.nn.softmax)

# Error Cross Entropy Loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_axes*tf.log(output_layer),reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess =tf.Session()
sess.run(tf.initialize_all_variables())

for i in range(1000):
	batch_x_axes,batch_y_axes=mnist.train.next_batch(100)
	sess.run(train_step,feed_dict={x_axes:batch_x_axes,y_axes:batch_y_axes})
	if i % 50 ==0:
		print(compute_accuracy(mnist.test.images,mnist.test.labels))










