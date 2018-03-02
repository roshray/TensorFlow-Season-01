
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

import tensorflow as tf
import numpy as np



X_data = np.random.randn(100).astype(np.float32)
y_data = X_data *0.1 + 0.3

# Create tensorflow structure start
weights = tf.Variable(tf.random_uniform([1],-1,1))
biases = tf.Variable(tf.zeros([1]))

y = weights * X_data + biases

loss = tf.reduce_mean(tf.square(y-y_data))

train = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

init = tf.initialize_all_variables()
#create tensorflow structure end


sess =tf.Session()
sess.run(init)


for step in range(201):
	sess.run(train)
	if step % 20 ==0:
		print(step,sess.run(weights),sess.run(biases))