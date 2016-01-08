"""
Test Algorithm to get color bands from the resistor
"""
import numpy as np
from scipy import ndimage, misc
import pdb
import math


def find_bands(ROI,clf, line_of_best_fit):
	"""
	given an ROI of a image with the resistor in focus
	input: ROI as ndimage
		   clf (a classifier with a 'predict' funcion)
		   line_of_best_fit , array of x and y values as [[x1, ... , xn],[y1, ... , yn]]
	"""
	if clf == None:
		print "Error : please specilfy a trained classifier"
		return None

