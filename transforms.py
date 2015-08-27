"""
Implement Matrix transformations
"""

import numpy as np
import scipy
from matplotlib import pyplot as plt
import math

pi = math.pi

def translate(img, dx, dy):
	img = np.matrix(img) + np.transpose([dx, dy])
	return img

def rotate(img, angle):
	angle = angle*pi/180.0
	img = img*np.matrix([[math.cos(angle), math.sin(angle)],[-math.sin(angle), math.cos(angle)]])
	return img

def paint(mat, width, height):
	img = np.zeros((height, width))
	mat = mat.tolist()
	for coord in mat:
		x = coord[0]
		y = coord[1]
		img[y][x] = 255
	return img