"""
testing
"""
import pdb
import numpy as np
import scipy as sp
import cv2
from matplotlib import pyplot as plt
import sys
import math
from scipy import ndimage
from transforms import *
from colors import *
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
import pygame
import glob
import colorcorrect.algorithm as cca
from skimage import color

def ColorMatch(RGB):
	return RGBColorMatch(RGB[0],RGB[1],RGB[2])

def draw(img):
	fig = plt.figure()
	plt.imshow(img)
	plt.show()
	return fig

sources = glob.glob("ColorLabels/*.data")
print "analyzing files : "+ str(sources) 
imgs = []

cumulative = dict()
cumulative2 = dict()

for source in sources:
	s = source.split('/')
	filename = s[len(s)-1].split('.')[0]
	if filename == "rs9":
		filename+='.png'
	else:
		filename+='.JPG'
	print filename

	img = sp.ndimage.imread('images/'+filename)
	#img = cca.luminance_weighted_gray_world(img)
	img_lab = color.rgb2lab(img)
	plt.figure()
	plt.imshow(img)
	plt.show()
	fig = plt.figure()
	ax = fig.add_subplot(121, projection='3d',title="RGB")
	ax2 = fig.add_subplot(122, projection='3d',title="LAB")
	f = open(source,'r')
	lines = f.readlines()
	print lines[0:4]
	for line in lines:
		line = line.replace('\n','')
		#pdb.set_trace()
		if line == '':
			continue
		[x,y,c] = line.split(',')
		x = int(x)
		y = int(y)
		red = img[y][x][0]
		green = img[y][x][1]
		blue = img[y][x][2]
		[l,a,b] = img[y][x]
		ax.scatter(xs=red,ys=green,zs=blue,c=c,marker='x')
		ax2.scatter(xs=a,ys=b,zs=l,c=c,marker='x')
		if not c in cumulative.keys():
			cumulative[c] = [(red,green,blue)]
			cumulative2[c] = [(a,b,l)]
		else:
			cumulative[c].append((red,green,blue))
			cumulative2[c].append((a,b,l))
		print x,y,c, l, a, b
	ax.set_xlabel('RED')
	ax.set_ylabel('GREEN')
	ax.set_zlabel('BLUE')
	ax2.set_xlabel('A')
	ax2.set_ylabel('B')
	ax2.set_zlabel('L')
	plt.show()

print 'displaying cumulative colors'
fig=plt.figure()
ax = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')
for c in cumulative.keys():
	for (r,b,g) in cumulative[c]:
		ax.scatter(xs=r,ys=g,zs=b,c=c,marker='x')
	for (a,b,l) in cumulative2[c]:
		ax2.scatter(xs=a,ys=b,zs=l,c=c,marker='x')
ax.set_xlabel('RED')
ax.set_ylabel('GREEN')
ax.set_zlabel('BLUE')
ax2.set_xlabel('A')
ax2.set_ylabel('B')
ax2.set_zlabel('L')
plt.show()
