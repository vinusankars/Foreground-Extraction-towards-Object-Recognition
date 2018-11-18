#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 01:04:25 2018

@author: vinusankars
"""

import cv2 as cv 
from gist import gist
from createGabor import createGabor
from prefilt import prefilt
from gistGabor import gistGabor

img = cv.imread('2007_009654.jpg')
g = gist(img)

print(g)
print('Shape is ', g.shape)