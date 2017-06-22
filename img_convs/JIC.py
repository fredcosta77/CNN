__doc__ = '''This module uses the pillow library to increase dataset size for image data through affine transformations.'''
from PIL import Image, ImageFilter
import os

mean_blur_kernel = [1,1,1,1,1,1,1,1,1]
gaussian_blur_kernel = [1,2,1,2,4,2,1,2,1]
edge_detection_kernel = [-1,-1,-1,-1,8,-1,-1,-1,-1]
sharpen_kernel = [0,-1,0,-1,5,-1,0,-1,0]
identity_kernel = [0,0,0,0,1,0,0,0,0]

def batch_apply(directory, transformation, new_directory = "./"):
	if directory[-1] != '/': directory += '/'
	for fn in os.listdir(directory):
		try:
			filename, filetype = fn.split(".")
			transformation(directory + fn, new_directory + filename + "_BATCHED." + filetype)
		except Exception as e:
			print(e)
	
def apply_kernel(image_fn, kernel, new_image_fn = "output.png"):
	f = open(image_fn, "rb")
	im = Image.open(image_fn)
	im = im.filter(ImageFilter.Kernel((3,3), kernel))
	im.save(new_image_fn)

def mean_blur(image, new_image_fn = "output.png"): return apply_kernel(image, mean_blur_kernel, new_image_fn)
	
def gaussian_blur(image, new_image_fn = "output.png"): return apply_kernel(image, gaussian_blur_kernel, new_image_fn)

def edge_detection(image, new_image_fn = "output.png"): return apply_kernel(image, edge_detection_kernel, new_image_fn)
	
def sharpen(image, new_image_fn = "output.png"): return apply_kernel(image, sharpen_kernel, new_image_fn)
	
def identity(image, new_image_fn = "output.png"): return apply_kernel(image, identity_kernel, new_image_fn)

