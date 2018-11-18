#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 20:25:47 2018

@author: vinusankars
"""

import cv2 as cv
import numpy as np
from getGistNeighbors import getGistNeighbors as ggn

#-----------
#Function to geometric prior
#Input:
#    Image 
#Arguments passed:
#    g files
#    test - the image array
#    kg - no. of g files
#    ka - no. of a files
#Returns:
#    returns geometric prior
#-----------

def geometricPrior(img, kg=100):
    files = ggn(img, 10*kg)[:kg]
    pgf = np.zeros((400, 400))#+0.1
    count = 0

    for file in files:
        i1 = cv.imread('train/data/'+file)
        i2 = cv.imread('train/annotations/'+(file.split('.'))[0]+'.png')
        
        i1 = cv.resize(i1, (400,400))
        i2 = cv.resize(i2, (400,400))

        count += 1

        for x in range(400):
            for y in range(400):
                if list(i2[x][y]) != [0,0,0]:
                    pgf[x][y] += 1

        if (count)%10 == 0:
            print("Selected {0:d}/{1:d}".format(count, kg))

    pgf = pgf/kg*100
    pgf = cv.GaussianBlur(pgf, (41,41), 7)

    return pgf, count