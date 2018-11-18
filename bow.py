#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 20:04:47 2018

@author: vinusankars
"""

import cv2 as cv
import numpy as np

#BOW bag of words vector representation
#Input images
#Argument passed: img1, img2
#Returns error

def bow(img1, img2):
    #Keypoint detection
    sift = cv.xfeatures2d.SIFT_create()
    bf = cv.BFMatcher(cv.NORM_L1,crossCheck=False)

    img1 = cv.resize(img1, (400,400)).astype('uint8')
    img2 = cv.resize(img2, (400,400)).astype('uint8')

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    #Keypoint matching
    try:
        temp = bf.match(des1, des2)
    except:
        return np.inf
    
    match = sorted(temp, key = lambda x:x.distance)
    error = 0
    
    #Calculating error
    for m in range(128):
        try:
            i1 = match[m].queryIdx
            i2 = match[m].trainIdx        
            error += np.linalg.norm(des1[i1] - des2[i2])
        except:
            continue
    
    #Return error
    return error