import PIL.Image as Image
import sys

try:
	epoch = sys.argv[1]
except:
	epoch = 10
imagesdir = "images/epoch_" + str(epoch) +"/"

w = 800
h = 600

conv1_combined = Image.new("RGB", (8*w, 4*h))
name = "conv1"
cw = 0
ch = 0
for i in range(32):
	cur_imgfn = imagesdir + name + "-" + str(i) + ".png"
	f = open(cur_imgfn, "rb")
	im = Image.open(f)
	im.load()
	f.close()
	conv1_combined.paste(im, (cw, ch))
	cw += w
	if (i+1) % 8 == 0: 
		cw = 0
		ch += h

f = open(imagesdir + name + ".png", "wb")
conv1_combined.save(f)
f.close()
	
conv2_combined = Image.new("RGB", (8*w, 8*h))
name = "conv2"
cw = 0
ch = 0
for i in range(64):
	cur_imgfn = imagesdir + name + "-" + str(i) + ".png"
	f = open(cur_imgfn, "rb")
	im = Image.open(f)
	im.load()
	f.close()
	conv2_combined.paste(im, (cw, ch))
	cw += w
	if (i+1) % 8 == 0: 
		cw = 0
		ch += h

f = open(imagesdir + name + ".png", "wb")
conv2_combined.save(f)
f.close()
