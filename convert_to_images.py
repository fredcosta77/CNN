import numpy as np
import matplotlib.pyplot as plt
import pickle

pickledir = "pickles/epoch_10/"
imagesdir = "images/epoch_10/"

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
	
name = "conv2"
data = get_data(name)
for i in range(64):
	plt.imshow(data[0,:,:,i])
	plt.savefig(imagesdir + name + "-" + str(i) + ".png")
	
name = "fc"
data = get_data(name)
plt.imshow(data.reshape(32,32))
plt.savefig(imagesdir + name + ".png")

name = "output"
data = get_data(name)
plt.imshow(data)
plt.savefig(imagesdir + name + ".png")
