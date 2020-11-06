import numpy as np
from PolLayer import *

class PolynomialNeuralNetwork:

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
						w_grad = self.dCo_w(L, b, a, y)
						self.layers[L].weights[a][b] -= self.lr*pow(w_grad, -1)

						n_grad = self.dCo_n(L, b, a, y)
						self.layers[L].exps[a][b] -= self.lr*pow(n_grad, -1)

					b_grad = self.dCo_b(L, a, y)
					self.layers[L+1].biases[a] -= self.lr*pow(b_grad, -1)

		return loss

	# derivative of cost with respect to bias at layer L+1 on neuron a with output y
	def dCo_b(self, L, a, y):
		return self.layers[L].func(self.layers[L].get_z(a, bias=self.layers[L+1].biases[a]), deriv=True)*self.dCo_a(L+1, a, y)

	# derivative of cost with respect to exponent at layer L on edge from a to b (b's in layer L+1), with output y
	def dCo_n(self, L, a, b, y):
		w = self.layers[L].weights[b][a]
		n = self.layers[L].exps[b][a]
		a = self.layers[L].neurons[a] # even if it overrides the argument it doesn't matter because that info isn't used after this line
		dZb_n = w*(np_pow(a,n))*np.log(abs(a))*norm(a) # np.log(x) is actually ln(x)
		db_zb = self.layers[L].func(self.layers[L].get_z(b, bias=self.layers[L+1].biases[b]), deriv=True)
		return dZb_n*db_zb*self.dCo_a(L+1, b, y)

	# derivative of cost with respect to activation a at layer L, with output y (recursive function)
	def dCo_a(self, L, a, y):
		if L == len(self.layers)-1: # if on the last layer
			return self.dCo_p(a, y)
		sigma = 0
		for i in range(self.layers[L+1].n_neurons): # for each neuron in the next layer
			sigma += self.da_a(L, a, i)*self.dCo_a(L+1, i, y)
		return sigma

	# derivative of neuron (b) with respect to an another neuron (a) in layer L
	def da_a(self, L, a, b):
		w = self.layers[L].weights[b][a]
		n = self.layers[L].exps[b][a]
		a = self.layers[L].neurons[a] # even if it overrides the argument it doesn't matter because that lost info isn't used after this line
		db_zb = self.layers[L].func(self.layers[L].get_z(b, bias=self.layers[L+1].biases[b]), deriv=True)
		return w*n*np_pow(a,n-1)*db_zb

	# derivative of cost with respect to weight in layer L on edge from a to b with output y
	def dCo_w(self, L, a, b, y):
		n = self.layers[L].exps[b][a]
		a = self.layers[L].neurons[a]
		db_zb = self.layers[L].func(self.layers[L].get_z(b, bias=self.layers[L+1].biases[b]), deriv=True)
		return np_pow(a,n)*db_zb*self.dCo_a(L+1, b, y)

	# derivative of cost with respect to final prediction on neuron i
	def dCo_p(self, a, y):
		return 2*(self.layers[-1].neurons[a]-y[a])

	def get_cost(self, x, y):
		return np.sum((x-y)**2)

	def predict(self, x):
		self.layers[0].feed(x)
		for L in range(1, len(self.layers)):
			self.layers[L].feed(self.layers[L-1])
		return self.layers[-1].neurons

	def add_layer(self, layer):
		self.layers[-2].log(layer)
		self.layers.append(layer)

	def __str__(self):
		res = ''
		for i, layer in enumerate(self.layers):
			res += "Layer #" + str(i) + ": neurons: " + str(layer.n_neurons) + ", activation: " + str(layer.func) + ", dtype: " + str(layer.dtype) + "\n"
		return res
