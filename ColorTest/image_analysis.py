"""
testing
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys
import math
from scipy import ndimage
from transforms import *
from ../colors import *
import matplotlib.image as mpimg
import pygame

def ColorMatch(RGB):
	return RGBColorMatch(RGB[0],RGB[1],RGB[2])

def draw(img):
	plt.imshow(img)
	plt.show()

source = 'clr_test1.JPG'

img_RGB = cv2.blur(cv2.cvtColor((cv2.imread('images/'+source,1)), cv2.COLOR_BGR2RGB), (2,2))

img_LAB = cv2.blur(cv2.cvtColor((cv2.imread('images/'+source,1)), cv2.COLOR_BGR2LAB), (2,2))

img = mpimg.imread('images/'+source)

brown = [(240, 230), (236,245), (237,280)]
black = [(237,266)]


print "calling RGB matcher\n"
print ColorMatch(img_RGB[brown[0][0]][brown[0][1]])