import cv2 as cv
from sklearn.cluster import KMeans
import os

def getDesc(n=100):
	path = 'train/data/'

	for i in os.listdir(path)[:n]:
		img = cv.imread(path+i)
		_, des = surf.detectAndCompute(img, None)
		
		if len(des) < 250:
			des = np.array(list(des)+[0]*(250-len(des)))

		des = des[:250]
		D.append(des)

	return D

surf = cv.xfeatures2d.SURF_create()
km = KMeans(n_clusters = 50)
desc = getDesc()
km.fit(desc)