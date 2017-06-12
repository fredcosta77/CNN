import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)
import pickle
import sys

n_classes = 10
batch_size = 128

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)

checkpoint = sys.argv[1]
checkpoint_filename = "checkpoints/model-" + str(checkpoint) + ".ckpt"

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
	#                        size of window         movement of window
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')



def convolutional_neural_network(x, view_layers = False):
	weights = {'W_conv1':tf.Variable(tf.random_normal([5,5,1,32])),
			   'W_conv2':tf.Variable(tf.random_normal([5,5,32,64])),
			   'W_fc':tf.Variable(tf.random_normal([7*7*64,1024])),
			   'out':tf.Variable(tf.random_normal([1024, n_classes]))}

	biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
			  'b_conv2':tf.Variable(tf.random_normal([64])),
			  'b_fc':tf.Variable(tf.random_normal([1024])),
			  'out':tf.Variable(tf.random_normal([n_classes]))}

	x = tf.reshape(x, shape=[-1, 28, 28, 1])

	conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
	conv1 = maxpool2d(conv1)
    
	conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
	conv2 = maxpool2d(conv2)

	fc = tf.reshape(conv2,[-1, 7*7*64])
	fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
	fc = tf.nn.dropout(fc, keep_rate)

	output = tf.matmul(fc, weights['out'])+biases['out']
	if view_layers: return (conv1, conv2, fc, output)
	else: return output

def use_neural_network(x, view_layers = False):
	if view_layers:
		conv1, conv2, fc, output = convolutional_neural_network(x, True)
	else: output = convolutional_neural_network(x)
	saver = tf.train.Saver()

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		try:
			saver.restore(sess, checkpoint_filename)
		if view_layers:
			return (conv1.eval(), conv2.eval(), fc.eval(), output.eval())
		else: return output.eval()

print("running on first image of test set")
img = mnist.test.images[0]
conv1, conv2, fc, output = use_neural_network(img,True)
#output = use_neural_network(img, False)
print("Conv1: ", conv1)
print("Conv2: ", conv2)
print("FC : ",fc)
print("Output: ", output)

f = open("input.pickle", 'wb')
pickle.dump(img, f)
f.close()

f = open("conv1.pickle", 'wb')
pickle.dump(conv1, f)
f.close()

f = open("conv2.pickle", 'wb')
pickle.dump(conv2, f)
f.close()

f = open("fc.pickle", 'wb')
pickle.dump(fc, f)
f.close()

f = open("output.pickle", 'wb')
pickle.dump(output, f)
f.close()
