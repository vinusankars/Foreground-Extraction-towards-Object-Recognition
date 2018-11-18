#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 09:30:23 2018

@author: vinusankars
Reference: https://www.pyimagesearch.com/2014/07/28/a-slic-superpixel-tutorial-using-python/
"""
# import the necessary packages
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import cv2 as cv

#-----------
#Function to group pixels to superpixels
#Input:
#    Image and number of segments
#Arguments passed:
#    test - the relative path of the input image
#    n - number of segments to be made in the super pixel image
#Returns:
#    returns segments of shape as input image with labels of superpixel of each pixel
#Plots the image with superpixels partitioned
#-----------

def spixel(img, n = 500):    
    #Call slic to get segments
    #Segments label each pixel to one of the n superpixels
    #Sigma used is 5
    segments = slic(img, n_segments = n, sigma = 5)
    
    #For plotting the image result
    bound = mark_boundaries(img, segments)
    plt.imshow(bound)
    plt.title('Superpixel segments')
    plt.axis("off")    
    plt.show()

    #Return segments
    return segments, bound