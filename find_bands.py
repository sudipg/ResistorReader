"""
Test Algorithm to get color bands from the resistor
"""
import numpy as np
from scipy import ndimage, misc, signal
import pdb
import math
import pickle
from matplotlib import pyplot as plt
import copy
import cv2

def filterAndThreshold(img, threshold = 130):
    """
    Take the ROI as an image and return the binary thresholded version of the ROI
    steps implmented: 1. Equivalent of Gaussian high pass filter, 2. thresholding to make a binary image to use later for finding the axis, etc
    """
    lowPass = ndimage.gaussian_filter(img,10)
    highPass = img - lowPass
    highPass = ndimage.gaussian_filter(highPass,10)
    #highPassThresholded = map(lambda x: np.array([255 if y>threshold else 0 for y in x]),highPass)
    shape=highPass.shape
    highPass = highPass.reshape(-1)
    highPassThresholded = np.array(map(lambda x: 255 if x>threshold else 0, highPass)).reshape(shape)
    return highPassThresholded


def find_bands(ROI, line_of_best_fit,clf=None):
	"""
	given an ROI of a image with the resistor in focus
	input: ROI as ndimage
		   clf (a classifier with a 'predict' funcion)
		   line_of_best_fit , array of x and y values as [[x1, ... , xn],[y1, ... , yn]]
	returns: list of possible color band sequences
	"""
	if clf == None:
		try: 
			pf = open('classifier.pdata','r')
			clf = pickle.load(pf)
			pf.close()
			print "found classifier"
		except Excption as e:
			print type(e)
			print "Error : please specify a trained classifier"
			return None



	ROI_grey = cv2.cvtColor(ROI, cv2.COLOR_RGB2GRAY)
	print "found grayscale"
	plt.imshow(ROI_grey)
	plt.show()
	plt.cla()


	BW = filterAndThreshold(ROI_grey,120)
	plt.imshow(BW)
	plt.show()
	plt.cla()

	pdb.set_trace()

	possible_sequences = []
	plt.cla()
	fig = plt.imshow(ROI)
	start_ortho = get_orthogonal_line(line_of_best_fit, (line_of_best_fit[0][0], line_of_best_fit[1][0]), 20)
	ortho_line = copy.deepcopy(start_ortho)
	i = 0
	x_step = 3
	print 
	while i < len(line_of_best_fit[0]):
		ortho_line = advance_line(line_of_best_fit, ortho_line, x_step)
		plt.plot(ortho_line[0], ortho_line[1], 'b*')
		i+=x_step
	plt.show()
	plt.cla()


def get_slope(line):
	y = line[1]
	x = line[0]
	y2 = y[1]
	y1 = y[0]
	x2 = x[1]
	x1 = x[0] 
	return (y2-y1)/(x2-x1)

def get_orthogonal_slope(line):
	return -1/get_slope(line)

def get_orthogonal_line(line, point,num_pixels):
	x1, y1 = point
	ortho_slope = get_orthogonal_slope(line)
	ortho_line_x = [x1]
	ortho_line_y = [y1]
	for x in xrange(1,num_pixels//2):
		ortho_line_x = [x1-x]+ortho_line_x+[x1+x]
		y = ortho_slope*x
		ortho_line_y = [y1-y]+ortho_line_y+[y1+y]
	return [ortho_line_x, ortho_line_y]

def advance_line(line_axis, ortho_line, x_amount):
	y_amount = x_amount*get_slope(line_axis)
	return [map(lambda x: x+x_amount, ortho_line[0]), map(lambda y: y+y_amount, ortho_line[1])]

def get_sample_bracket(line, binary_img):
	"""
	takes a binary image (filter and threshold), line of best fit
	return rectangular matrix of RGB values for the sampling hot spots
	"""
	pass


def main():
	try:
		ROI = misc.imread('current_ROI.jpg')
		lf = open('current_lobf.pdata' ,'r')
		line_of_best_fit = pickle.load(lf)
		lf.close()
	except Excption as e:
		print type(e)
		return None
	find_bands(ROI, line_of_best_fit)

if __name__ == '__main__':
    main()