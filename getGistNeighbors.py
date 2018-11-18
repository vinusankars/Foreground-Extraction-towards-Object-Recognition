#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 01:04:25 2018

@author: vinusankars
"""

from gist import gist
import cv2 as cv
import os
import numpy as np

def getGistNeighbors(img, kg=1000):
	
	if 'train_gist.txt' not in os.listdir(os.getcwd()):
		with open('train_gist.txt', 'a+') as f:
			path = 'train/data/'
			print('Total train set ', len(os.listdir(path)))
			for j, i in enumerate(os.listdir(path)):
				if (j+1)%100 == 0:
					print(j+1)
				img = cv.imread(path+i)
				g = ' '.join(map(str, gist(img)))
				f.write(i+' '+g+'\n')

	f = open('train_gist.txt', 'r')
	load = f.read().split('\n')
	f.close()

	dic = {}
	g = gist(img)

	print('Length of train data is ', len(load))
	for j, i in enumerate(load):
		if (j+1)%5000 == 0:
			print(j+1)
		data = i.split()
		try:
			g_ = np.array(list(map(float, data[1:])))
			dic[data[0]] = np.linalg.norm(g-g_)
		except:
			continue

	files = sorted(dic, key=dic.__getitem__)
	return files[:kg]