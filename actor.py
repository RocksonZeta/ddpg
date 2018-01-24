import tensorflow as tf 
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
import numpy as np
import math
import layers


# Hyper Parameters
LAYER1_SIZE = 400
LAYER2_SIZE = 300
LEARNING_RATE = 1e-4
TAU = 0.001
BATCH_SIZE = 64

class ActorNetwork:
	"""docstring for ActorNetwork"""
	def __init__(self,sess,state_dim,action_dim):

		self.sess = sess
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.state_input = tf.placeholder(tf.float32,[None,state_dim])
		self.target_state_input = tf.placeholder(tf.float32,[None,state_dim])
		# self.action_input = tf.placeholder(tf.float32,[None,action_dim])
		self.is_training  = tf.placeholder(tf.bool)
		self.target_is_training  = tf.placeholder(tf.bool)

		self.actor_output  = self.create_network(self.state_input,self.is_training , "actor_eval")
		self.target_actor_output  = self.create_network(self.target_state_input,self.target_is_training,"actor_target")
		self.eval_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "actor_eval")
		self.target_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "actor_target")
		# define training rules
		self.create_update_target()
		self.create_training_method()

		# self.sess.run(tf.global_variables_initializer())

		# self.update_target()
		#self.load_network()

	def create_training_method(self):
		self.q_gradient_input = tf.placeholder("float",[None,self.action_dim])
		self.parameters_gradients = tf.gradients(self.actor_output,self.eval_params,-self.q_gradient_input)
		self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(zip(self.parameters_gradients,self.eval_params))


	def create_network(self,input , training, variable_scope):
		with tf.variable_scope(variable_scope):
			x = input
			with tf.variable_scope(variable_scope+"_bn0"):
				x = layers.batch_norm(x,training ,variable_scope+"_bn0",tf.nn.relu)
			x = tf.layers.dense(input , LAYER1_SIZE ,kernel_initializer=tf.random_normal_initializer(stddev=0.1))
			with tf.variable_scope(variable_scope+"_bn1"):
				x = layers.batch_norm(x,training ,variable_scope+"_bn1",tf.nn.relu)
			x = tf.layers.dense(x , LAYER2_SIZE ,kernel_initializer=tf.random_normal_initializer(stddev=0.1))
			with tf.variable_scope(variable_scope+"_bn2"):
				x = layers.batch_norm(x,training ,variable_scope+"_bn2",tf.nn.relu)
			x = tf.layers.dense(x , self.action_dim , activation=tf.nn.tanh,kernel_initializer=tf.random_normal_initializer(stddev=0.1))
		return x
	def create_update_target(self):
		ema = tf.train.ExponentialMovingAverage(decay=1-TAU)
		self.target_update = ema.apply(self.eval_params)
		target_net = [ema.average(x) for x in self.eval_params]
		self.assign_target = [tf.assign(r,v) for r,v in zip(self.target_params,target_net)]
		# a = np.random.uniform()
		# self.target_update = [tf.assign(r,a*r+(1-a)*v) for r,v in zip(self.target_params,self.eval_params)]

	def update_target(self):
		# self.sess.run(self.target_update)
		self.sess.run([self.target_update , self.assign_target])
	# def assign(self):
	# 	self.sess.run(self.assign_direct)

	def train(self,q_gradient_batch,state_batch):
		self.sess.run(self.optimizer,feed_dict={
			self.q_gradient_input:q_gradient_batch,
			self.state_input:state_batch,
			self.is_training: True
			})

	def actions(self,state_batch):
		return self.sess.run(self.actor_output,feed_dict={
			self.state_input:state_batch,
			self.is_training: True
			})

	def action(self,state):
		return self.sess.run(self.actor_output,feed_dict={
			self.state_input:[state],
			self.is_training: False
			})[0]


	def target_actions(self,state_batch):
		return self.sess.run(self.target_actor_output,feed_dict={
			self.target_state_input: state_batch,
			self.target_is_training: True
			})

	# f fan-in size
	def variable(self,shape,f):
		return tf.Variable(tf.random_uniform(shape,-1/math.sqrt(f),1/math.sqrt(f)))


	# def batch_norm(self,x,training_phase,scope_bn,activation=None):
	# 	return tf.cond(training_phase, 
	# 	lambda: tf.contrib.layers.batch_norm(x, activation_fn=activation, center=True, scale=True,
	# 	updates_collections=None,is_training=True, reuse=None,scope=scope_bn,decay=0.9, epsilon=1e-5),
	# 	lambda: tf.contrib.layers.batch_norm(x, activation_fn =activation, center=True, scale=True,
	# 	updates_collections=None,is_training=False, reuse=True,scope=scope_bn,decay=0.9, epsilon=1e-5))
		
'''
	def load_network(self):
		self.saver = tf.train.Saver()
		checkpoint = tf.train.get_checkpoint_state("saved_actor_networks")
		if checkpoint and checkpoint.model_checkpoint_path:
			self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
			print "Successfully loaded:", checkpoint.model_checkpoint_path
		else:
			print "Could not find old network weights"
	def save_network(self,time_step):
		print 'save actor-network...',time_step
		self.saver.save(self.sess, 'saved_actor_networks/' + 'actor-network', global_step = time_step)

'''

		
