"""
image quantization for identifying the value of the resistor (reducing dimensionality)
"""
import sys
import copy
import numpy as np
from scipy import misc, ndimage
from sklearn.cluster import KMeans

def quantize_kmeans(img,num_colors=15, verbose=False):
	height = len(img)
	width = len(img[0])
	# set up KMeans estimator
	est = KMeans(n_clusters = num_colors,n_jobs=-1)
	pixels = np.reshape(img, (-1,3))
	est.fit(pixels)
	assign_means = lambda x: est.cluster_centers_[x]
	labels = copy.deepcopy(est.labels_)
	img_quantized = np.array(map(assign_means, labels)).reshape(height, width,3)
	if verbose:
		misc.imshow(img_quantized)
	return img_quantized