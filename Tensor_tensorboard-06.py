# Activation function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(input_,in_size,out_size,n_layer,activation_function=None):

	layer_name ="layer%s"% n_layer

	with tf.name_scope("layers"):
		with tf.name_scope("weights"):
			weights =tf.Variable(tf.random_normal([in_size,out_size]),name="W")
			tf.summary.histogram(layer_name + "/weights",weights)
		with tf.name_scope("Baises"):
			biases = tf.Variable(tf.zeros([1,out_size]) + 0.1, name="B")
			tf.summary.histogram(layer_name + "/biases",biases)

		with tf.name_scope("Inputs"):
		
			y = tf.matmul(input_,weights) + biases
			tf.summary.histogram(layer_name + "/Inputss",y)

		#with tf.name_scope("Activations_functions"):
		if activation_function is None:
			outputs = y
		else:
			outputs = activation_function(y)
			tf.summary.histogram(layer_name + "/outputs",outputs)

		return outputs


x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data)-0.5 + noise

#plt.scatter(x_data, y_data)
#plt.show()



# define placeholder

with tf.name_scope("input_"):

	x_axes = tf.placeholder(tf.float32,[None,1],name="x_inputs")
	y_axes = tf.placeholder(tf.float32,[None,1],name="y_inputs")

# Add Hidden Layer

hidden_layer_one = add_layer(x_axes,1,10,n_layer=1,activation_function=tf.nn.relu)

# Output Layer

output_layer_one = add_layer(hidden_layer_one,10,1,n_layer=2,activation_function=None)


#ERROR LOSS
with tf.name_scope("LOSS"):
	loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_axes-output_layer_one),reduction_indices=[1]),name="L")

	tf.summary.scalar("loss",loss)
with tf.name_scope("Train"):
	train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)



sess = tf.Session()

merged=tf.summary.merge_all()
writer = tf.summary.FileWriter("logs/",sess.graph)
sess.run(tf.global_variables_initializer())

#visualize the result

'''
fig = plt.figure()
axes =fig.add_subplot(1,1,1)
axes.scatter(x_data, y_data)
plt.ion()
plt.show()
'''

for i in range(1000):
	sess.run(train_step,feed_dict={x_axes:x_data,y_axes:y_data})

	if i%50 ==0:
		result=sess.run(merged,feed_dict={x_axes:x_data,y_axes:y_data})
		writer.add_summary(result,i)

		'''
		# Visualize the result
		try:
			axes.lines.remove(lines[0])
		except Exception:

			pass

		output_layer_value = sess.run(output_layer_one,feed_dict={x_axes:x_data})
		# plot the prediction

		lines =axes.plot(x_data,output_layer_value,'r-',lw=5)
		plt.pause(1)

'''
