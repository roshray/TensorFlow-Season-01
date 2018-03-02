# Activation function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(input_,in_size,out_size,activation_function=None):
	weights =tf.Variable(tf.random_normal([in_size,out_size]))
	biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)
	y = tf.matmul(input_,weights) + biases
	if activation_function is None:
		outputs = y
	else:
		outputs = activation_function(y)

	return outputs


x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data)-0.5 + noise

#plt.scatter(x_data, y_data)
#plt.show()



# define placeholder

x_axes = tf.placeholder(tf.float32,[None,1])
y_axes = tf.placeholder(tf.float32,[None,1])

# Add Hidden Layer

hidden_layer_one = add_layer(x_axes,1,10,activation_function=tf.nn.relu)

# Output Layer

output_layer_one = add_layer(hidden_layer_one,10,1,activation_function=None)


#ERROR LOSS

loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_axes-output_layer_one),reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)


init =tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

#visualize the result


fig = plt.figure()
axes =fig.add_subplot(1,1,1)
axes.scatter(x_data, y_data)
plt.ion()
plt.show()


for i in range(1000):
	sess.run(train_step,feed_dict={x_axes:x_data,y_axes:y_data})

	if i%50 ==0:
		#print(sess.run(loss,feed_dict={x_axes:x_data,y_axes:y_data}))
		# Visualize the result
		try:
			axes.lines.remove(lines[0])
		except Exception:

			pass

		output_layer_value = sess.run(output_layer_one,feed_dict={x_axes:x_data})
		# plot the prediction

		lines =axes.plot(x_data,output_layer_value,'r-',lw=5)
		plt.pause(1)

