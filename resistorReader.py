# -*- coding: utf-8 -*-
"""
------------------------------------------------------------------------------
Created on Sat Jul 11 20:45:07 2015

Find the region of interest in the sample image using a template from file.

@author: sudipguha and Charles (XiaRui) Zhang
-------------------------------------------------------------------------------
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys
import math
from scipy import ndimage
from transforms import *

"""
-------------------------------------------------------------------------------
Keeping images in the global frame to have better interactive debug access.
List of important debug objects:
- template image
- test image
- list of matches and keypoints

-------------------------------------------------------------------------------
"""


if len(sys.argv)>1 and sys.argv[1] == '-s':
    imgSource = sys.argv[2]
    templateSource = sys.argv[3]
    print 'selected sources:\r\n'
    print 'template is '+imgSource+' and the test image is '+templateSource;
elif len(sys.argv)>1 and sys.argv[1] == '-d':
    imgSource = 'test_res.png'
    templateSource = 'r10t.png'
else:
    imgSource = raw_input('Please enter the template picture name : ')
    templateSource = raw_input('Please enter the template picture name : ')
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

    imgBlurred = cv2.blur(img, (14,14))
    templateBlurred = cv2.blur(template, (5,5))

    # now draw them on top of the image
    matchedKeypointsX, matchedKeypointsY, img3 = findMatches(templateBlurred, imgBlurred)

    plt.clf()
    plt.subplot(311)
    plt.imshow(img3, cmap = 'gray')
    lowerBoundX, upperBoundX, lowerBoundY, upperBoundY = findBoxAroundNthPercentile(matchedKeypointsX, matchedKeypointsY, 0.5, 50)
    plt.subplot(312)
    ROI = img[lowerBoundY:upperBoundY, lowerBoundX:upperBoundX]
    plt.imshow(ROI, cmap = 'gray')
    plt.subplot(313)
    highPassThresholded = filterAndThreshold(ROI)
    plt.imshow(highPassThresholded, cmap = 'gray')
    a,b,x_range,y_range, shape = getLineOfBestFit(highPassThresholded)
    plt.plot(x_range,y_range,'ro')
    plt.show()
    findBestAngle(highPassThresholded, shape)

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
    """
    takes 2 matrices 
    returns integer denoting the correlation
    """
    im1 = paint(img, w, h)
    im2 = paint(compImg, w, h)
    conv = np.multiply(im1,im2)
    cost = np.sum(conv)
    return cost

def filterAndThreshold(img):
    """
    Take the ROI as an image and return the binary thresholded version of the ROI
    steps implmented: 1. Equivalent of Gaussian high pass filter, 2. thresholding to make a binary image to use later for finding the axis, etc
    """
    lowPass = ndimage.gaussian_filter(img,10)
    highPass = img - lowPass
    highPass = ndimage.gaussian_filter(highPass,10)
    highPassThresholded = map(lambda x: np.array([255 if y>130 else 0 for y in x]),highPass)
    return highPassThresholded


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
    """
    ---------------------------------------------------------------------------
    Finds keypoints that matches the TEMPLATE in IMG.
    
    Assumes template and img are 2D arrays (i.e. Pictures)
    
    RETURNS the coordinates of all matched keypoints in IMG.

    The return format is: xCoordinates, yCoordinates.
    ---------------------------------------------------------------------------
    """

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

def findLowerAndUpperPercentile(arr, lowerPercentile, upperPercentile):
    """
    ---------------------------------------------------------------------------

    TODO: Algorithmitically, this may not be ideal.
    Currently, it takes O(n log(n)) time.
    Maybe use better algorithm? For now, shouldn't be too bad?

    Given an array ARR, it returns 2 values in the array. 

    The first value's percentile in the ARR is = lowerPercentile.

    The second value's percentile in the ARR is = upperPercentile

    ---------------------------------------------------------------------------
    """

    if (lowerPercentile > 1 or upperPercentile > 1 or lowerPercentile < 0 or upperPercentile < 0):
        print("findLowerAndUpperPercentile Error: One (or more) percentile is (are) invalid.\n")
        print("lowerPercentile: " + str(lowerPercentile) + "upperPercentile: " + str(upperPercentile))
    else:
        temp = sorted(arr, key = lambda x:x)
        return temp[int(len(temp) * lowerPercentile)], temp[int(len(temp) * upperPercentile)]

def findBoxAroundNthPercentile(keypointsX, keypointsY, percentile, border):
    """
    ---------------------------------------------------------------------------

    Given the X and Y coordinates of all keypoints, this returns 4 values.

    These 4 values, represent the lower and upper bound on X and Y coordinates 
    such that PERCENTILE percent of all key points are inside that box.

    Also adds some padding to the box in the form of BORDER

    ---------------------------------------------------------------------------
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
    """
    ---------------------------------------------------------------------------
    
    Given an image in the form of IMG, returns the DFT "img"
    NOTE: the magnitude ONLY is returned and in dB scale

    ---------------------------------------------------------------------------
    """
    dft = np.fft.fft2(img)
    dft = np.fft.fftshift(dft)
    dft  = 20 * np.log(np.abs(dft))
    return dft


if __name__ == '__main__':
    main()

