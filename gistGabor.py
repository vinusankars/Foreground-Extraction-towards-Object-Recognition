#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 01:04:25 2018

@author: vinusankars
"""

import numpy as np
import cv2 as cv

def downN(x, N):
	nx = np.floor(np.linspace(0, x.shape[0], N+1)).astype('int')
	ny = np.floor(np.linspace(0, x.shape[1], N+1)).astype('int')
	y = np.zeros((N, N, x.shape[2]))
	
	for i in range(N):
		for j in range(N):
			#print(nx, ny)
			v = np.mean(np.mean(x[nx[i]:nx[i+1], ny[j]:ny[j+1], :], 0), 0)
			#print(v, y.shape, x[nx[i]:nx[i+1], ny[j]:ny[j+1], :].shape)
			y[i, j, :] = v

	return y

def gistGabor(img, w, G):
	if len(img.shape) == 2:
		img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

	nrows, ncols, c = list(img.shape)
	N = c

	n, n, nfilters = list(G.shape)
	W = w*w
	g = np.zeros((W*nfilters, N))

	img = np.fft.fft2(img)
	k = 0

	for i in range(nfilters):
		X = G[:, :, i]
		ff = np.zeros((n, n, N))
		for j in range(N):
			ff[:, :, j] = X
		X = ff
		ig = np.abs(np.fft.ifft2(img*X))
		v = downN(ig, w)
		g[k:k+W, :] = v.reshape((W, N))
		k += W

	if c == 3:
		g = g.reshape((g.shape[0]*3, int(g.shape[1]/3)))

	return g