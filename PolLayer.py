import numpy as np
import random

MAX_INITIAL_WEIGHT = -1
MIN_INITIAL_WEIGHT = 1

MAX_INITIAL_BIAS = -1
MIN_INITIAL_BIAS = 1

MAX_INITIAL_EXP = 1
MIN_INITIAL_EXP = 1

def norm(x):
	return 1 if x>= 0 else -1

def np_pow(a, b):
	#return np.sign(a) * (np.abs(a)) ** ((n*10**gd(n))%(10**gd(n)))
	return np.sign(a) * (np.abs(a) ** b)

# 1 -> bias
# 2 -> weight
# 3 -> exps
def r(t):
	if t == 1:
		return random.uniform(MIN_INITIAL_BIAS, MAX_INITIAL_BIAS)
	elif t == 2:
		return random.uniform(MIN_INITIAL_WEIGHT, MAX_INITIAL_WEIGHT)
	else:
		return random.uniform(MIN_INITIAL_EXP, MAX_INITIAL_EXP)

def sigmoid(x, deriv=False):
	return 1/(1+np.exp(-x)) if not deriv else sigmoid(x)*(1-sigmoid(x))

def none(x, deriv=False):
	return x if not deriv else 1

class PolLayer:

	def __init__(self, n_neurons, activation=none, dtype=np.float64):
		self.n_neurons = n_neurons
		self.func = activation
		self.dtype = dtype
		self.neurons = np.array([0 for i in range(n_neurons)], dtype=dtype)
		self.biases = np.array([r(1) for _ in range(self.n_neurons)], dtype=dtype)

	def log(self, next_layer):
		self.weights = np.array([[r(2) for _ in range(self.n_neurons)] for n in range(next_layer.n_neurons)], dtype=self.dtype)
		self.exps = np.array([[r(3) for _ in range(self.n_neurons)] for n in range(next_layer.n_neurons)], dtype=self.dtype)

	def feed(self, prev):
		if isinstance(prev, ExpLayer):
			for i in range(self.n_neurons):
				self.neurons[i] = prev.activation(i, bias=self.biases[i])
		else:
			self.neurons = prev

	def activation(self, n_index, bias=0):
		return self.func(np.dot(np_pow(self.neurons, self.exps[n_index]), self.weights[n_index])+bias)

	def get_z(self, n_index, bias=0):
		return np.dot(np_pow(self.neurons, self.exps[n_index]), self.weights[n_index])+bias