#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 01:04:25 2018

@author: vinusankars
"""

import numpy as np

def createGabor(nops, n, m): #Number of Orientations Per Scale
	nscales = len(nops)
	nfilters = sum(nops)

	llen = sum([len(range(nops[i])) for i in range(nscales)])
	param = np.zeros((llen, 4))
	l = 0

	for i in range(nscales):
		for j in range(nops[i]):
			param[l, :] = [0.35, 0.3/(1.85**(i-1)), 16*nops[i]**2/32**2, np.pi/nops[i]*(j-1)]
			l += 1

	#Frequencies
	fx, fy = np.meshgrid(np.linspace(-m/2, m/2-1, m), np.linspace(-n/2, n/2-1, n))
	fr = np.fft.fftshift(np.sqrt(fx**2 + fy**2))
	t = np.fft.fftshift(np.angle(fx + fy*1.0j))

	#Transfer functions
	G = np.zeros((n, m, nfilters))
	for i in range(nfilters):
		tr = t + param[i, 3]
		tr += 2*np.pi*(tr<-np.pi).astype('int') - 2*np.pi*(tr>np.pi).astype('int')
		G[:, :, i] = np.exp(-10*param[i, 0]*(fr/n/param[i,1]-1)**2 - 2*param[i, 2]*np.pi*tr**2)

	return G	