import numpy as np
import gym


class NormBox(object):
	'''
	compress: box->[low,high] ,eg.[-100,10000]->[-1,1]; 10000 |-> 1.
	uncompress: [-1,1] -> box
	'''
	def __init__(self,box ,low = -1. , high=1.):
		self.box = box
		self.low = low
		self.high = high
		if np.any(box.high<1e10):
			self.center = (box.high +box.low)/(high-low)
			self.len = (box.high - box.low)/(high-low)
		else:
			self.center = np.zeros_like(box.high)
			self.len = np.ones_like(box.high)
	def compress(self,x):
		return (x - self.center)/self.len
	def uncompress(self,x):
		return x * self.len + self.center

def normEnv(env):
	""" crate a new environment class with actions and states normalized to [-1,1] """
	# action_sp = env.action_space
	# observation_sp = env.observation_space
	# print(type(acsp))
	# print(obsp.__dict__)
	
	# if not type(acsp)==gym.spaces.box.Box:
	# 	raise RuntimeError('Environment with continous action space (i.e. Box) required.')
	# if not type(obsp)==gym.spaces.box.Box:
	# 	raise RuntimeError('Environment with continous observation space (i.e. Box) required.')

	env_type = type(env)

	class NormEnv(env_type):
		def __init__(self):
			self.__dict__.update(env.__dict__)
			self.action_is_box = False
			self.observation_is_box = False
			if type(env.action_space) == gym.spaces.box.Box:
				self.action_is_box = True
				self.action_space_nb = NormBox(env.action_space)
				self.action_space = gym.spaces.Box(
					self.action_space_nb.compress(env.action_space.low),
					self.action_space_nb.compress(env.action_space.high))
			if type(env.observation_space) == gym.spaces.box.Box:
				self.observation_is_box = True
				self.observation_space_nb = NormBox(env.observation_space)
				self.observation_space = gym.spaces.Box(
					self.observation_space_nb.compress(env.observation_space.low),
					self.observation_space_nb.compress(env.observation_space.high))


		def step(self,action):
			if self.action_is_box :
				action = np.clip(self.action_space_nb.uncompress(action) , env.action_space.low , env.action_space.high)
			ob,reward,done,info = env_type.step(self,action)
			if self.observation_is_box:
				ob = self.observation_space_nb.compress(ob)
			return ob, reward, done, info
	return NormEnv()
if '__main__' == __name__:
	env = gym.make('CartPole-v0')
	e = normEnv(env)
	print(e.action_space.__dict__)
	print(e.action_space)
	# e.reset()
	# for i in range(100):
	# 	e.step(e.action_space.sample())
	# 	e.render()
