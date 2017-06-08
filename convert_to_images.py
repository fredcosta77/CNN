import numpy as np
import matplotlib.pyplot as plt
import pickle

for i in [("input", (28,28)), ("conv1", ()), ("conv2", ()), ("fc", ()), ("output",())]:
	try:
		fname = i[0].replace('.pickle','')
		f = open(fname + ".pickle", "rb")
		data = pickle.load(f)
		plt.imshow(np.reshape(data,i[1]))
		plt.savefig(fname + ".png")	
	except:
		pass
