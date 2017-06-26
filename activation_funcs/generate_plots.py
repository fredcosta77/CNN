import numpy as np
import matplotlib.pyplot as plt
import math

def threshold(x):
	if x >= 0: return 1
	else: return 0
	
def logistic(x): return 1 / (1 + math.e ** (-5 * x))
	
tanh = math.tanh
	
def relu(x): return max(0,x)
	
def leaky_relu(x):
	if x < 0: return .1 * x
	else: return x
	
def generate_plot(function, xlimits, ylimits):
	xs = [i for i in np.arange(xlimits[0],xlimits[1],.1)]
	ys = [function(i) for i in xs]
	plt.plot(xs, ys)
	plt.title(function.__name__)
	plt.ylim(ylimits)
	plt.xlim(xlimits)
	plt.savefig("./" + function.__name__ + ".png")
	plt.clf()

if __name__ == "__main__":

	generate_plot(threshold, (-2, 2), (-2, 2))
	generate_plot(logistic, (-2, 2), (-2, 2))
	generate_plot(tanh, (-2, 2), (-2, 2))
	generate_plot(relu, (-2, 2), (-2, 2))
	generate_plot(leaky_relu, (-2, 2), (-2, 2))	
