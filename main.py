import numpy as np
from layer import *
from PolLayer import *
from NN import *
from polNN import *
import random
import time
import matplotlib.pyplot as plt
import sys

try:
	gen = int(sys.argv[1])
except IndexError:
	gen = 5

w = np.random.uniform(-2,2, size=(3,))
b = np.random.uniform(-2,2, size=(3,))
n = np.random.uniform(-2,2, size=(3,))
x = np.array([5,4,-2])

y = np_pow(x, n)*w+b
epochs = 500

for i in range(1, gen+1):
	model = PolynomialNeuralNetwork([PolLayer(3), PolLayer(3)], lr=0.1)
	loss = model.fit(x,y, epochs=epochs, c=0)

	plt.plot([i for i in range(1,epochs+1)], loss, color='black')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.show()
	plt.clf()
