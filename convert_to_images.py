import numpy as np
import matplotlib.pyplot as plt
import pickle

pickledir = "pickles/"
imagesdir = "images/"

def get_data(n):
		f = open(pickledir + n + ".pickle", "rb")
		data = pickle.load(f)
		f.close()
		return data

name = "input"
data = get_data(name)
plt.imshow(np.reshape(data,(28,28)))
plt.savefig(imagesdir + name + ".png")
		
name = "conv1"
data = get_data(name)
for i in range(32):
	plt.imshow(data[0,:,:,i])
	plt.savefig(imagesdir + name + "-" + str(i) + ".png")
