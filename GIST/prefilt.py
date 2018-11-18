#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 01:04:25 2018

@author: vinusankars
"""

import numpy as np
import cv2 as cv

def prefilt(img, fc = 4):
	w = 5
	s1 = fc/np.log(2)**0.5

	#Padding images to reduce boundary artefacts
	img = np.log(img+1)
	img = cv.copyMakeBorder(img, w, w, w, w, cv.BORDER_REFLECT)
	sn, sm, c, N = list(img.shape)+list([1]*(4-len(img.shape)))
	n = max([sn, sm])
	n += n%2
	img = cv.copyMakeBorder(img, 0, n-sn, 0, n-sm, cv.BORDER_REFLECT)

	#Filter
	fx, fy = np.meshgrid(np.linspace(-n/2, n/2-1, n), np.linspace(-n/2, n/2-1, n))
	gf = np.fft.fftshift(np.exp(-(fx**2+fy**2)/s1**2))
	ff = np.zeros((gf.shape[0], gf.shape[0], c))

	for i in range(c):
		ff[:, :, i] = gf

	#Whitening
	gf = ff
	output = img - np.fft.ifft2(np.fft.fft2(img)*gf).real
	del img
	
	#Local contrast normalization
	localstd = np.sqrt(np.abs(np.fft.ifft2(np.fft.fft2(np.mean(output, 2)**2)*gf[:,:,0])))
	ff = np.zeros((localstd.shape[0], localstd.shape[1], c))

	for i in range(c):
		ff[:, :, i] = localstd

	localstd = ff
	output = output/(0.2+localstd)

	#Crop output to have same as input size
	output = output[w:sn-w, w:sm-w, :]

	return output