#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 01:04:25 2018

@author: vinusankars
"""

import cv2 as cv
import numpy as np
from createGabor import createGabor
from prefilt import prefilt
from gistGabor import gistGabor

def gist(I):
	I = cv.resize(I, (100, 100))
	if len(I.shape) == 2:
		I = cv.cvtColor(I, cv.COLOR_GRAY2BGR)

	m = I.shape[0]*I.shape[1]*I.shape[2]
	m = 240*320/m

	if m<1:
		I = cv.resize(I, None, fx=m, fy=m, interpolation = cv.INTER_LINEAR)

	G = createGabor([8, 8, 4], I.shape[0], I.shape[1])
	output = prefilt(I.astype('float'), 4)
	G = gistGabor(output, 4, G)[:, 0]

	return G