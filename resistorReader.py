# -*- coding: utf-8 -*-
"""
Find the region of interest in the sample image using a template from file.

@author: sudipguha and Charles (XiaRui) Zhang
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys
import math
from scipy import ndimage, misc
from transforms import *
from image_quantization import *
from find_bands import *
import pdb

if len(sys.argv)>1 and sys.argv[1] == '-s':
    imgSource = sys.argv[2]
    templateSource = sys.argv[3]
    print 'selected sources:\r\n'
    print 'template is '+templateSource+' and the test image is '+imgSource;
elif len(sys.argv)>1 and sys.argv[1] == '-d':
    imgSource = 'test_res.png'
    templateSource = 'r10t.png'
else:
    imgSource = raw_input('Please enter the template picture name : ')
    templateSource = raw_input('Please enter the template picture name : ')

# Keep images in the global frame to have better interactive debug access.
img = cv2.imread('images/'+imgSource,0)
template = cv2.imread('images/'+templateSource,0)
matches = []

def main():
    """
    ---------------------------------------------------------------------------

    TODO: Consider using command line arguments for image names?
    Something along the lines of sys.argv?

    ---------------------------------------------------------------------------
    """
    blurAmt = max(len(img)//100,14)
    imgBlurred = cv2.blur(img, (blurAmt,blurAmt))
    templateBlurred = cv2.blur(template, (4,4))

    # now draw them on top of the image
    matchedKeypointsX, matchedKeypointsY, img3 = findMatches(templateBlurred, imgBlurred)

    plt.clf()
    plt.subplot(311)
    plt.imshow(img3, cmap = 'gray')
    #pdb.set_trace()
    lowerBoundX, upperBoundX, lowerBoundY, upperBoundY = findBoxAroundNthPercentile(matchedKeypointsX, matchedKeypointsY, 0.5, blurAmt*15)
    plt.subplot(312)
    ROI = img[lowerBoundY:upperBoundY, lowerBoundX:upperBoundX]
    plt.imshow(ROI, cmap = 'gray')
    plt.subplot(313)
    highPassThresholded = filterAndThreshold(ROI)
    plt.imshow(highPassThresholded, cmap = 'gray')
    a,b,x_range,y_range, shape = getLineOfBestFit(highPassThresholded)
    plt.plot(x_range,y_range,'ro')
    [ox_range,oy_range] = get_orthogonal_line([x_range, y_range], (x_range[0],y_range[0]),20)
    plt.plot(ox_range,oy_range,'ro')
    plt.show()

    #findBestAngle(highPassThresholded, shape)
    img_c_cv = cv2.cvtColor(cv2.imread('images/'+imgSource), cv2.COLOR_BGR2RGB)
    img_c = misc.imread('images/'+imgSource)
    img_c = ndimage.interpolation.rotate(img_c, -90)
    ROI_c = img_c_cv[lowerBoundY:upperBoundY, lowerBoundX:upperBoundX,:]
    #pdb.set_trace()
    misc.imsave('current_ROI.jpg', ROI_c)
    lf = open('current_lobf.pdata', 'w')
    pickle.dump([x_range,y_range], lf)
    lf.close()
    print find_bands(ROI_c, line_of_best_fit=[x_range,y_range])
    #ROI_q = quantize_kmeans(ROI_c,num_colors=15)
    #misc.imsave('images/'+imgSource.split('.')[0]+'_q.'+imgSource.split('.')[1], ROI_q)

def findBestAngle(img, shape):
    """
    Take a filtered and thresholded ROI as an image (img)
    return a,b such that y = a*x + b is the best description of the axis of the resistor
    """

    # first find center of mass, and construct band for correlation.
    a = b = 0
    xPoints = []
    yPoints = []

    height = len(img)
    width = len(img[0])

    for y in xrange(len(img)):
        for x in xrange(len(img[y])):
            if img[y][x] > 0:
                xPoints.append(x)
                yPoints.append(y)
    xCenter = sum(xPoints)/len(xPoints)
    yCenter = sum(yPoints)/len(yPoints)
    original = np.transpose(np.matrix([xPoints, yPoints]))
    stripWidth = 0.15*(len(img))

    compImg = np.array([np.array([0 for j in range(len(img[0]))]) for i in range(len(img))])

    line = [0 for j in range(len(img[0])/6)] + [255 for j in range(2*len(img[0])/3)] + [0 for j in range(len(img[0])/6)]
    if len(line) != len(img[0]):
        line = line + [0 for j in range(len(img[0])-len(line))]
    line = np.array(line)
    for i in range(int(stripWidth//2)):
        compImg[yCenter+i] = line
        compImg[yCenter-i] = line
    compImg[yCenter] = line
    xPoints = []
    yPoints = []
    for y in xrange(len(compImg)):
        for x in xrange(len(compImg[y])):
            if compImg[y][x] > 0:
                xPoints.append(x)
                yPoints.append(y)

    compImgCenterX = sum(xPoints)/len(xPoints)
    compImgCenterY = sum(yPoints)/len(yPoints) 

    compImg = np.transpose(np.matrix([xPoints, yPoints]))
    # print compImg
    compImg = translate(compImg, xCenter - compImgCenterX, yCenter - compImgCenterY)
    # compImg = rotate(compImg, 90, xCenter, yCenter)

    bestAngle = 0
    highCost = None
    costs = dict()

    for angle in [x*0.5 for x in range(180)]:
        compImg = rotate(compImg, angle, xCenter, yCenter)
        newCost = costNaive(compImg, original, width, height)
        compImg = rotate(compImg, -angle, xCenter, yCenter)
        costs[angle] = newCost
        if highCost == None or highCost < newCost:
            highCost = newCost
            bestAngle = angle 

    print costs

    compImg = rotate(compImg, bestAngle, xCenter, yCenter)

    compImg = paint(compImg, len(img[0]), len(img))

    plt.clf()
    plt.figure()
    plt.subplot(211)
    plt.imshow(compImg,cmap='gray')
    plt.plot(xCenter,yCenter,'ro')
    plt.ylim(0,len(img))
    plt.xlim(0,len(img[0]))
    plt.subplot(212)
    plt.imshow(img,cmap='gray')
    plt.ylim(0,len(img))
    plt.xlim(0,len(img[0]))
    plt.show()

def costNaive(img, compImg, w, h):
    """Calculate the correlation of two matrices."""
    im1 = paint(img, w, h)
    im2 = paint(compImg, w, h)
    conv = np.multiply(im1,im2)
    cost = np.sum(conv)
    return cost

def getLineOfBestFit(img):
    """
    Analyze given image using filters and perform a best fit matching
    return a, b such that the line of best fit for that image is y = a*x + b
    """
    a = b = 0
    xPoints = []
    yPoints = []
    for y in xrange(len(img)):
        for x in xrange(len(img[y])):
            if img[y][x] > 0:
                xPoints.append(x)
                yPoints.append(y)

    
    shape = np.transpose(np.matrix([xPoints, yPoints]))

    #least sqaures regression also gives the same result.
    [a,b] = np.polyfit(xPoints,yPoints,1)
    x_range = [sorted(xPoints)[0]+x for x in range(int(sorted(xPoints)[len(xPoints)-1] - sorted(xPoints)[0]))]

    y_range = [b + a*x for x in x_range]
    return a,b,x_range,y_range, shape
    
def findMatches(template, img):
    """Return coordinates for all key points in 'img' that match 'template'."""
    orb = cv2.ORB_create()
    kpTemplate, desTemplate = orb.detectAndCompute(template, None)
    kpImg, desImg = orb.detectAndCompute(img, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.match(desTemplate, desImg)
    matches = sorted(matches, key = lambda x:x.distance)

    matchedKeypoints = set()

    img3 = img
    img3 = cv2.drawMatches(template,kpTemplate,img,kpImg,matches,img3,flags=2)

    for match in matches:
        matchedKeypoints.add(kpImg[match.trainIdx].pt)

    matchedKeypoints = list(matchedKeypoints)
    matchedKeypointsX = [match[0] for match in matchedKeypoints]
    matchedKeypointsY = [match[1] for match in matchedKeypoints]

    return matchedKeypointsX, matchedKeypointsY, img3
    
def plotMatches(template, img):
    x, y, img3 = findMatches(template, img)

    plt.clf()
    plt.imshow(img3)
    plt.show()

def findLowerAndUpperPercentile(arr, lower, upper):
    """Return a 2-tuple with the lower and upper percentiles of 'arr'.

    >>> findLowerAndUpperPercentile([1, 2, 3], 0.25, 0.75)
    (1, 3)
    """
    assert 0 < lower < 1 and 0 < upper < 1
    l_index, u_index = int(len(arr) * lower), int(len(arr) * upper)
    arr.sort()
    return (arr[l_index], arr[u_index])

def findBoxAroundNthPercentile(keypointsX, keypointsY, percentile, border):
    """
    Given the X and Y coordinates of all keypoints, this returns 4 values.

    These 4 values, represent the lower and upper bound on X and Y coordinates 
    such that PERCENTILE percent of all key points are inside that box.

    Also adds some padding to the box in the form of BORDER
    """
    
    if (percentile < 0 or percentile > 1):
        print("findBoxAroundNthPercentile Error: Percentile not valid.\n")
        print("Percentile is: " + str(percentile))
    else:
        lowerPercentile = 0.5 - percentile / 2
        upperPercentile = 0.5 + percentile / 2
        lowerBoundX, upperBoundX = findLowerAndUpperPercentile(keypointsX, lowerPercentile, upperPercentile)
        lowerBoundY, upperBoundY = findLowerAndUpperPercentile(keypointsY, lowerPercentile, upperPercentile)
        #TODO: Perform bound checking before returning
        return lowerBoundX - border, upperBoundX + border, lowerBoundY - border, upperBoundY + border    

def discreteFourierTransform(img):
    """Given an image in the form of IMG, returns the DFT "img"
    NOTE: the magnitude ONLY is returned and in dB scale
    """
    dft = np.fft.fft2(img)
    dft = np.fft.fftshift(dft)
    dft  = 20 * np.log(np.abs(dft))
    return dft

if __name__ == '__main__':
    main()

