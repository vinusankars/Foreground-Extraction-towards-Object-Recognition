#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 01:04:25 2018

@author: vinusankars
"""

import cv2 as cv
import numpy as np 
from time import time
from skimage.segmentation import mark_boundaries

clusters, cluster_pixels = 0, 0

def slic(I, K, M):
	global clusters, cluster_pixels    
	img = cv.cvtColor(I, cv.COLOR_BGR2Lab)

	h = img.shape[0]
	w = img.shape[1]
	N = h*w
	S = int(np.sqrt(N/K))

	clusters = []
	label = dict()
	dis = np.full((h, w), np.inf)
	itr = 10
	cluster_pixels = {}

	x = S/2
	y = S/2
	cluster = 1

	while x < h:
		while y < w:
			x, y = int(x), int(y)
			clusters.append([cluster, x, y, img[x][y][0] 
							,img[x][y][1], img[x][y][2]])
			cluster += 1
			y += S
		y = S/2
		x += S 

	for i in range(len(clusters)):
		id = clusters[i][0]
		ih = clusters[i][1]
		iw = clusters[i][2]
		il = clusters[i][3]
		ia = clusters[i][4]
		ib = clusters[i][5]

		for x in range(-1, 2):

			if iw+1 >= w:
				iw -= 2
			if ih+1 >= h:
				ih -= 2
			            
			grad = sum(img[iw+1][ih+1]) - sum(img[iw][ih])

			for y in range(-1, 2):
				h_ = ih + x
				w_ = iw + y

				if w_+1 >= w:
					w_ -= 2
				if h_+1 >= h:
					h_ -= 2

				ngrad = sum(img[w_+1][h_+1]) - sum(img[w_][h_])

				if grad > ngrad:
					clusters[i] = [id, h_, w_, img[h_][w_][0] 
								  ,img[h_][w_][1], img[h_][w_][2]]
					grad = ngrad

	for j in  range(itr):
		print('Iteration {0:d}/{1:d}'.format(j+1,itr))             
		for i in range(len(clusters)):
			id = clusters[i][0]
			ih = clusters[i][1]
			iw = clusters[i][2]
			il = int(clusters[i][3])
			ia = int(clusters[i][4])
			ib = int(clusters[i][5])

			for x in range(ih-2*S, ih+2*S):
				if x < 0 or x >= h:
					continue

				for y in range(iw-2*S, iw+2*S):
					if y < 0 or y >= w:
						continue

					l, a, b = img[x][y]
					dc = (l-il)**2 + (a-ia)**2 + (b-ib)**2
					dc = dc**0.5
					ds = (x-ih)**2 + (y-iw)**2
					ds = ds**0.5
					d = (dc/M)**2 + (ds/S)**2
					d = d**0.5

					if d < dis[x][y]:
						if (x, y) not in label:
							label[(x, y)] = clusters[i]
							if id not in cluster_pixels:
								cluster_pixels[id] = []
							cluster_pixels[id].append((x, y))

						else:
							cluster_pixels[label[(x, y)][0]].remove((x, y))
							label[(x, y)] = clusters[i]
							if id not in cluster_pixels:
								cluster_pixels[id] = []
							cluster_pixels[id].append((x, y))
						dis[x][y] = d

		for i in range(len(clusters)):
			sh, sw, num = 0, 0, 0
			for p in cluster_pixels[clusters[i][0]]:
				sh += p[0]
				sw += p[1]
				num += 1
				h_ = int(sh/num)
				w_ = int(sw/num)
				clusters[i] = [clusters[i][0], h_, w_, img[h_][w_][0]
							  ,img[h_][w_][1], img[h_][w_][2]]
                
		img1 = np.copy(I)
		segments = np.zeros((400,400), dtype='int')
        
		for i in cluster_pixels:
			x = np.random.randint(255)
			y = np.random.randint(255)
			z = np.random.randint(255)
			for k in cluster_pixels[i]:
				img1[k[0], k[1]] = (np.array([x,y,z])).astype('uint8')
				segments[k[0], k[1]] = i-1
                
		bound = mark_boundaries(I, segments)
		bound = (bound/np.max(bound)*255).astype('uint8')
		cv.imwrite('segment_{0:d}.png'.format(j+1), bound)
		cv.imwrite('patch_{0:d}.png'.format(j+1), img1)

img = cv.imread('2007_009654.jpg')
img = cv.resize(img, (400, 400))
start = time()
slic(img, 100, 25)
print('Total time taken in seconds is ', time()-start)