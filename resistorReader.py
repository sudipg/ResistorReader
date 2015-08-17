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


"""
-------------------------------------------------------------------------------
Keeping images in the global frame to have better interactive debug access.
List of important debug objects:
- template image
- test image
- list of matches and keypoints

-------------------------------------------------------------------------------
"""
img = cv2.imread('images/test_res.png',0)
template = cv2.imread('images/rt10.png',0)
matches = []

def main():
    """
    ---------------------------------------------------------------------------

    TODO: Consider using command line arguments for image names?
    Something along the lines of sys.argv?

    ---------------------------------------------------------------------------
    """

    imgBlurred = cv2.blur(img, (14,14))
    templateBlurred = cv2.blur(img, (5,5))
    # create ORB object for detecting features
    orb = cv2.ORB_create()
    kpT, desT = orb.detectAndCompute(templateBlurred, None)
    kpI, desI = orb.detectAndCompute(imgBlurred, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.match(desT, desI)
    matches = sorted(matches, key = lambda x:x.distance)


    # now draw them on top of the image
    matchedKeypointsX, matchedKeypointsY, img3 = findMatches(templateBlurred, imgBlurred)

    # x_range = [sorted(matched_keypoints_x)[0]+x for x in range(int(sorted(matched_keypoints_x)[len(matched_keypoints_x)-1] - sorted(matched_keypoints_x)[0]))]

    # y_range = [line_of_BF[1] + line_of_BF[0]*x for x in x_range]

    plt.clf()
    lowerBoundX, upperBoundX, lowerBoundY, upperBoundY = findBoxAroundNthPercentile(matchedKeypointsX, matchedKeypointsY, 0.50, 40)
    
    ROI = img [lowerBoundY:upperBoundY, lowerBoundX:upperBoundX]
    plt.clf()
    plt.imshow(ROI, cmap = 'gray')
    plt.show()
    
    bands = cv2.imread('images/Bands.png', 0)
    bands = cv2.blur(bands, (2,2))
    plotMatches(bands, ROI)

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
    img3 = cv2.drawMatches(template,kpTemplate,img,kpImg,matches[:15],img3,flags=2)

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

    # medianX = median(keypointsX)
    # medianY = median(keypointsY)
    # distanceX = computeOneDimensionalDistance(keypointsX, medianX)
    # distanceY = computeOneDimensionalDistance(keypointsY, medianY)
    # lowerBoundX = findNthPercentile(distanceX, 1 - percentile)
    # upperBoundX = findNthPercentile(distanceX, percentile)
    # lowerBoundY = findNthPercentile(distanceY, 1 - percentile)
    # upperBoundY = findNthPercentile(distanceY, percentile)

def discreteFourierTransform(img):
    """
    ---------------------------------------------------------------------------
    
    Given an image in the form of IMG, returns the DFT "img"

    ---------------------------------------------------------------------------
    """
    dft = np.fft.fft2(img)
    dft = np.fft.fftshift(dft)
    dft  = 20 * np.log(np.abs(dft))
    return dft


if __name__ == '__main__':
    main()

