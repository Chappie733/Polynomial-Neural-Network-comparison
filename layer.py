import numpy as np

INIT_BIASES = (-1,1)
INIT_WEIGHTS = (-1,1)

def none(x, deriv=False):
	return x if not deriv else 1

class Layer:

	def __init__(self, n_neurons, activation=none):
		self.n_neurons = n_neurons
		self.activation = activation
		self.neurons = np.zeros(shape=(n_neurons,))
		self.biases = np.random.uniform(low=INIT_BIASES[0], high=INIT_BIASES[1], size=(n_neurons))

	def log(self, layer):
		self.weights = np.random.uniform(low=INIT_WEIGHTS[0], high=INIT_WEIGHTS[1], size=(layer.n_neurons, self.n_neurons))

	def get_val(self, n_index, bias=0):
		return self.activation(np.dot(self.neurons,self.weights[n_index])+bias)

	def get_z(self, n_index, bias=0):
		return np.dot(self.neurons,self.weights[n_index])+bias

	def feed(self, prev):
		if isinstance(prev, Layer):
			for i in range(self.n_neurons):
				self.neurons[i] = prev.get_val(i, bias=self.biases[i])
		else:
			try:
				if prev.shape != self.neurons.shape:
					raise Exception("Expected data of shape " + str(self.neurons.shape) + " but received data with shape " + str(prev.shape))
				self.neurons = prev
			except AttributeError:
				raise Exception("the \"feed\" function only allows Layer objects or numpy arrays as parameters!")
