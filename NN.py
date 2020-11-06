import numpy as np
from layer import *

class NeuralNetwork:

	def __init__(self, layers, lr=0.01):
		for i in range(len(layers)-1):
			layers[i].log(layers[i+1])
		self.layers = layers
		self.lr = lr

	def fit(self, x, y, epochs=5, c=5):
		
		loss = []

		for epoch in range(1, epochs+1):
			predicted = self.predict(x)
			cost = self.get_cost(predicted, y)
			if c != 0:
				self.lr -= self.lr/(c*epoch)
			loss.append(cost)
			print("Epoch: " + str(epoch) + ", cost: " + str(cost))

			for L in range(len(self.layers)-1):
				for a in range(self.layers[L+1].n_neurons):
					for b in range(self.layers[L].n_neurons):
						self.layers[L].weights[a][b] -= self.lr*pow(self.dC_w(L, b, a, y), -1)
					self.layers[L+1].biases[a] -= self.lr*pow(self.dC_b(L, a, y), -1)


		return loss

	# derivative of cost with respect to a weight from node a (in layer L) to node b (in layer L+1)
	def dC_w(self, L, a, b, y):
		v = self.layers[L].neurons[a]
		zb = self.layers[L].get_z(b, bias=self.layers[L+1].biases[b])
		return v*self.layers[L].activation(zb, deriv=True)*self.dC_a(L+1, b, y)

	def dC_b(self, L, a, y):
		zb = self.layers[L].get_z(a, bias=self.layers[L+1].biases[a])
		return self.layers[L].activation(zb, deriv=True)*self.dC_a(L+1, a, y)

	# derivative of cost with respect to a node
	def dC_a(self, L, a, y):
		if L == len(self.layers)-1:
			return 2*(self.layers[-1].neurons[a]-y[a])
		sigma = 0
		for i in range(self.layers[L+1].n_neurons):
			sigma += self.da_a(L, a, i)*self.dC_a(L+1, i, y)
		return sigma
	
	# derivative of neuron (b in layer L+1) with respect to an another neuron (a in layer L)
	def da_a(self, L, a, b):
		w = self.layers[L].weights[b][a]
		zb = self.layers[L].get_z(b, bias=self.layers[L+1].biases[b])
		return w*self.layers[L].activation(zb, deriv=True)

	def get_cost(self, pred, y):
		return np.sum((pred-y)**2)

	def predict(self, x):
		self.layers[0].feed(x)
		for i in range(1, len(self.layers)):
			self.layers[i].feed(self.layers[i-1])
		return self.layers[-1].neurons

	def __str__(self):
		res = ''
		for i, layer in enumerate(self.layers):
			res += "Layer #" + str(i) + ": " + str(layer.n_neurons) + " neurons, activation = " + str(layer.activation) + "\n"
		return res