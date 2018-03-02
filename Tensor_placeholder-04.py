import tensorflow as tf

input_one =tf.placeholder(tf.float32)
input_two = tf.placeholder(tf.float32)


output = tf.multiply(input_one, input_two)

with tf.Session() as sess:
	print(sess.run(output, feed_dict={input_one:[7],input_two:[9]}))
