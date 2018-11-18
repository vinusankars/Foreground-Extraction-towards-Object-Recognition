import cv2 as cv
from sklearn.cluster import KMeans
import numpy as np
import os
import time

def getDesc(n=20):
	path = 'train/data/'
	D = []

	for i in os.listdir(path)[:n]:
		img = cv.imread(path+i)
		_, des = surf.detectAndCompute(img, None)
		
		if len(des) < 250:
			des = np.array(list(des)+[0]*(250-len(des)))

		des = des[:250].reshape(-1)
		D.append(des)

	return D

start = time.time()
surf = cv.xfeatures2d.SURF_create()
km = KMeans(n_clusters = 2)
desc = getDesc()
print('Fitting')
km.fit(desc)
print(time.time()-start)