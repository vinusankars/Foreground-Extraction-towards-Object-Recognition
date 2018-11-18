#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 01:04:25 2018

@author: vinusankars
"""

import numpy as np 
import cv2 as cv 
from bow import bow
import os

def getBowNeighbors(img, ka=1000):
	path = 'train/data/'
	files = os.listdir(path)
	dic = {}

	for i in range(ka):
		if (i+1)%100 == 0:
			print('Done {0:d}/{1:d}'.format(i+1, ka))
		x = np.random.randint(len(files)-2)
		I = cv.imread(path+files[x])
		e = bow(img, I)
		dic[files[x]] = e

	files = sorted(dic, key=dic.__getitem__)
	return files