
import tensorflow as tf 
import numpy as np
import math


LAYER1_SIZE = 400
LAYER2_SIZE = 300
LEARNING_RATE = 1e-3
TAU = 0.001
L2 = 0.01

class CriticNetwork:
	'''
	Q(s,a|θ)
	'''
	def __init__(self,sess,state_dim,action_dim):
		self.time_step = 0
		self.sess = sess
		# create q network
		self.state_input = tf.placeholder(tf.float32 , [None,state_dim])
		self.action_input = tf.placeholder(tf.float32 , [None,action_dim])
		self.target_state_input = tf.placeholder(tf.float32 , [None,state_dim])
		self.target_action_input = tf.placeholder(tf.float32 , [None,action_dim])

		self.q_value = self.create_network(self.state_input , self.action_input,'critic_eval')
		self.target_q_value = self.create_network(self.target_state_input , self.target_action_input,'critic_target')
		self.eval_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES , 'critic_eval')
		self.target_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES , 'critic_target')

		self.create_training_method()
		self.create_update_target()

		# initialization 
		# self.sess.run(tf.global_variables_initializer())
			
		# self.update_target()
	
	def create_training_method(self):
		self.y_input = tf.placeholder(tf.float32,[None,1])
		# Define training optimizer
		weight_decay = tf.add_n([L2 * tf.nn.l2_loss(var) for var in self.eval_params])
		self.cost = tf.reduce_mean(tf.square(self.y_input - self.q_value)) + weight_decay
		# self.cost = tf.reduce_mean(tf.square(self.y_input - self.q_value))
		self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.cost,var_list=self.eval_params)
		self.action_gradients = tf.gradients(self.q_value,self.action_input)

	def create_network(self,state_input,action_input,variable_scope):
		'''
		q(s,q)->scalar,how good the action(create by actor) in the state is
		'''
		with tf.variable_scope(variable_scope):
			x = tf.layers.dense(state_input , LAYER1_SIZE , tf.nn.relu , kernel_initializer=tf.random_normal_initializer(stddev=0.1))
			x = tf.layers.dense(x , LAYER2_SIZE , kernel_initializer=tf.random_normal_initializer(stddev=0.1)) + tf.layers.dense(action_input,LAYER2_SIZE ,kernel_initializer=tf.random_normal_initializer(stddev=0.1))
			x = tf.nn.relu(x)
			q_value = tf.layers.dense(x ,1 ,kernel_initializer=tf.random_normal_initializer(stddev=0.1) )
		return q_value
	def create_update_target(self):
		self.assign_direct = [tf.assign(r,v) for r,v in zip(self.target_params,self.eval_params)]
		ema = tf.train.ExponentialMovingAverage(decay=1-TAU)
		self.target_update = ema.apply(self.eval_params)
		target_net = [ema.average(x) for x in self.eval_params]
		self.assign_target = [tf.assign(r,v) for r,v in zip(self.target_params,target_net)]
		# a = np.random.uniform()
		# self.target_update = [tf.assign(r,a*r+(1-a)*v) for r,v in zip(self.target_params,self.eval_params)]
	def update_target(self):
		# self.sess.run(self.target_update)
		self.sess.run([self.target_update , self.assign_target])

	def train(self,y_batch,state_batch,action_batch):
		self.time_step += 1
		_,cost = self.sess.run([self.optimizer,self.cost],feed_dict={
			self.y_input:y_batch,
			self.state_input:state_batch,
			self.action_input:action_batch
			})
		if self.time_step %100 ==  0 :
			print("cost:",cost)

	def gradients(self,state_batch,action_batch):
		return self.sess.run(self.action_gradients,feed_dict={
			self.state_input:state_batch,
			self.action_input:action_batch
			})[0]

	def target_q(self,state_batch,action_batch):
		return self.sess.run(self.target_q_value,feed_dict={
			self.target_state_input:state_batch,
			self.target_action_input:action_batch
			})

	def q_value(self,state_batch,action_batch):
		return self.sess.run(self.q_value,feed_dict={
			self.state_input:state_batch,
			self.action_input:action_batch})

	# f fan-in size
	def variable(self,shape,f):
		return tf.Variable(tf.random_uniform(shape,-1/math.sqrt(f),1/math.sqrt(f)))
'''
	def load_network(self):
		self.saver = tf.train.Saver()
		checkpoint = tf.train.get_checkpoint_state("saved_critic_networks")
		if checkpoint and checkpoint.model_checkpoint_path:
			self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
			print "Successfully loaded:", checkpoint.model_checkpoint_path
		else:
			print "Could not find old network weights"

	def save_network(self,time_step):
		print 'save critic-network...',time_step
		self.saver.save(self.sess, 'saved_critic_networks/' + 'critic-network', global_step = time_step)
'''
		