import collections
import random

class ReplayBuffer():
	def __init__(self ,capacity):
		self.deque = collections.deque()
		self._size = 0
		self._capacity = capacity
	
	@property
	def capacity(self):
		return self._capacity
	@property
	def size(self):
		return self._size

	def add(self,state, action, reward, next_state, done):
		experience = (state, action, reward, next_state, done)
		self.deque.append(experience)
		if self.size < self.capacity :
			self._size+=1
		else:
			self.deque.popleft()
	def sample(self, n):
		return random.sample(self.deque , n)
	def reset(self):
		self.deque = collections.deque()
		self._size = 0