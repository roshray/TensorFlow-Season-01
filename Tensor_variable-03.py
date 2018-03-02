import tensorflow as tf

var = tf.Variable(0,name="counter")

power = tf.constant(1)

new_value = tf.add(var,power)

update = tf.assign(var,new_value)

init = tf.initialize_all_variables()

with tf.Session() as sess:
	sess.run(init)
	for step in range(5):
		sess.run(update)
		print(sess.run(var))