# Tensor flow all about Session 
# Matrix

import tensorflow as tf


matrix_one = tf.constant([[1,2]],shape=[2,2])
matrix_two = tf.constant([[7,6]],shape=[2,2])

product = tf.matmul(matrix_one, matrix_two)

sess = tf.Session()
print(sess.run(product))
sess.close()

