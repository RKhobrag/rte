import data
import embeddings
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

num_units = 300
num_classes = 2
batch_size = 50
time_steps_p = 49
time_steps_h = 27
embedding_dim = 300
def conditional_encoding(premise, hypothesis, label):
	with tf.variable_scope("te") as scope:
		out_weights = tf.Variable(tf.random_normal([num_units, num_classes]))
		out_bias = tf.Variable(tf.random_normal([num_classes]))
		

		x_p = tf.placeholder("float", [batch_size, time_steps_p, embedding_dim])	
		x_h = tf.placeholder("float", [batch_size, time_steps_h, embedding_dim])
		y = tf.placeholder(tf.int64, [batch_size])


		cell_p = rnn.LSTMCell(num_units, state_is_tuple=True)
		outputs_p, state_p = tf.nn.dynamic_rnn(cell=cell_p, inputs=x_p, dtype="float32")

		scope.reuse_variables()
		cell_h = rnn.LSTMCell(num_units, state_is_tuple=True)
		outputs_h, _ = tf.nn.dynamic_rnn(cell=cell_h, inputs=x_h,  dtype="float32")

		# apply attention on final represntation of hypothesis
		# w_y = tf.Variable(tf.random_normal([time_steps_p, time_steps_p]))
		# w_h = tf.Variable(tf.random_normal([time_steps_p, time_steps_p]))
		# w_p = tf.Variable(tf.random_normal([time_steps_p, time_steps_p]))
		# w_x = tf.Variable(tf.random_normal([time_steps_p, time_steps_p]))
		# e_l = tf.Variable(tf.random_normal([time_steps_p]))

	# outputs_p = tf.reshape(outputs_p, [time_steps_p, -1])
	# m = tf.tanh(tf.matmul(w_y, outputs_p) + np.dot(tf.matmul(w_h, outputs_h[:,-1]), np.ones(1, time_steps_p)))
	# alpha = tf.nn.softmax(logits=m)
	# r = tf.matmul(outputs, tf.transpose(alpha))
	# h_new = tf.tanh(tf.matmul(w_p, r)+tf.matmul(w_x,outputs))

	outputs_h = outputs_h[:,-1]
	# context_rep = tf.reshape(outputs, [-1, num_units])
	pred = tf.matmul(outputs_h, out_weights) + out_bias
	logits = tf.reshape(pred, [batch_size, num_classes])

	losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
	mask = tf.sequence_mask(time_steps_h)
	losses = tf.boolean_mask(losses, mask)
	loss = tf.reduce_mean(losses)

	optimizer = tf.train.AdamOptimizer()
	train_op = optimizer.minimize(loss)

	prediction = tf.equal(tf.argmax(logits, 1), y)
	accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

	init = tf.global_variables_initializer()
	with tf.Session() as sess:
		sess.run(init)
		i = 1
		prev_i = 0

		while(batch_size*i<premise.shape[0]):
			x_train_p_batch = premise[prev_i:batch_size*i,:,:]
			x_train_h_batch = hypothesis[prev_i:batch_size*i,:,:]
			y_train_batch = label[prev_i:batch_size*i]


			sess.run(train_op, feed_dict={x_p: x_train_p_batch, x_h: x_train_h_batch, y: y_train_batch})
			
			acc = sess.run(accuracy, feed_dict={x_p: premise[:batch_size,:,:], x_h: hypothesis[:batch_size,:,:], y: label[:batch_size]})
			print("Accuracy: ", acc)
			print("\n")
			prev_i = batch_size*i
			i+=1



def main():
	premise, p_max, hypothesis, h_max, l, _ = data.load_train_data()
	print(premise.shape)
	print(hypothesis.shape)
	
	global time_steps_p
	global time_steps_h
	time_steps_p = p_max
	time_steps_h = h_max

		
	label = []
	for x in l:
		if x == 'entailment':
			label.append(0)
		else:
			label.append(1)	
	label = np.array(label)

	e = embeddings.embeddings('GoogleNews-vectors-negative300.bin')
	premise = e.embed(premise)
	hypothesis = e.embed(hypothesis)

	# premise = data.pad(premise)
	# hypothesis = data.pad(hypothesis)

	# premise = premise.reshape(1000,49,embedding_dim)
	# print(premise.shape)
	# # print(label.shape)
	conditional_encoding(premise, hypothesis, label)

	# print(premise.shape)
	# print(hypothesis.shape)

if __name__=="__main__":
	main()
