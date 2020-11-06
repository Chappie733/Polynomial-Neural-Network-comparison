import numpy as np
from layer import *
from expLayer import *
from NN import *
from expNN import *
import random
import time
import matplotlib.pyplot as plt

k = 5
epochs = 100
steps = 100
start = 100
INPUT_SIZE = (3,)
gen = 40

def get_disc(a,b):
	return 100-(np.abs(np.sum(a-b))/np.sum(np.maximum(a,b)))*100

def get_dist(a,b):
	return np.sum((a-b)**2)

def get_avg(x):
	return np.sum(x)/len(x)

# higher weights and biases required -> exp model is worse

w = np.random.uniform(-1,1, size=INPUT_SIZE)
b = np.random.uniform(-1,1, size=INPUT_SIZE)
n = np.random.uniform(-1,1, size=INPUT_SIZE)

for p in range(1, gen+1):
	x = np.random.uniform(low=-k, high=k, size=INPUT_SIZE)
	y = np_pow(x, n)*w+b

	model = NeuralNetwork([Layer(INPUT_SIZE[0]), Layer(INPUT_SIZE[0])], lr=0.01)
	model.fit(x,y, epochs=epochs)

	exp_model = ExponentialNeuralNetwork([ExpLayer(INPUT_SIZE[0]), ExpLayer(INPUT_SIZE[0])], lr=0.01)
	exp_model.layers[0].weights = np.copy(model.layers[0].weights)
	exp_model.layers[0].biases = np.copy(model.layers[0].biases)

	exp_loss = exp_model.fit(x,y, epochs=epochs)

#	std_losses = []
#	exp_losses = []
	losses_ratio = []	
	input_intervals = [] # absolute distance between training inputs and current inputs
	# TODO: start from the extremes of the training inputs

	for i in range(start, start+steps+1):
		x += k
		y = np_pow(x, n)*w+b

		std_pred = model.predict(x)
		exp_pred = exp_model.predict(x)

		input_intervals.append(i*k*INPUT_SIZE[0])
#		std_losses.append(get_dist(std_pred, y))
#		exp_losses.append(get_dist(exp_pred, y))
		# >0 -> std loss higher, <0 -> exp loss higher
		losses_ratio.append(get_dist(std_pred, y)/get_dist(exp_pred,y)-1)

#	plt.plot(input_intervals, std_losses)
#	plt.plot(input_intervals, exp_losses)
	plt.plot(input_intervals, losses_ratio)
	plt.xlabel("Diff")
	plt.ylabel("Loss ratio")
	plt.savefig('C:\\Users\\Luni\\Desktop\\exponential net\\interesting data\\' + str(p), bbox_inches='tight')
#	plt.show()
	plt.clf()