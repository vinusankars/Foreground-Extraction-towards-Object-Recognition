#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 01:04:25 2018

@author: vinusankars
"""

import numpy as np
import cv2 as cv
from globalDescriptor import globalDescriptor

def getGists(path, train_ids):
	gfeatures = np.zeros((960, len(train_ids)), dtype='uint8')
	print("Length of train set is ", len(train_ids))
	
	for i in range(len(train_ids)):
		x= train_ids[i]
		if i%50 == 0:
			print(i)
		img = cv.imread(path+x)
		img = cv.resize(img, (100,100))
		g = globalDescriptor(img)
		gfeatures[:, i] = g[:, 0]

	return gfeatures