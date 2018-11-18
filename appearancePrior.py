#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 01:04:25 2018

@author: vinusankars
"""

import cv2 as cv 
import numpy as np
from sklearn.cluster import KMeans
from getBowNeighbors import getBowNeighbors as gbn

def appearancePrior(img, ka=100):	
    sift = cv.xfeatures2d.SIFT_create()
    img = cv.resize(img, (400,400))
    files = gbn(img, 10*ka)[: ka]
    count = 0

    #Prior
    pfw = np.zeros((400,400)) + 0.1    
    B, F = np.array([[0]*128]), np.array([[0]*128])

    for file in files:
        i1 = cv.imread('train/data/'+file)
        i2 = cv.imread('train/annotations/'+(file.split('.'))[0]+'.png')
        
        i1 = cv.resize(i1, (400,400))
        i2 = cv.resize(i2, (400,400))

        count += 1
        kp, des = sift.detectAndCompute(i1, None)
        des = des[:100]
        
        for k in range(len(des)):
            x, y = kp[k].pt
            y, x = int(x), int(y)
            
            #Partition bg and fg pixel probs
            if list(i2[x, y]) == [0,0,0]:
                B = np.append(B, [des[k]], 0)
            else:
                F = np.append(F, [des[k]], 0)

        if (count)%10 == 0:
            print("Selected {0:d}/{1:d}".format(count, ka))

    #Cluster descriptors
    X = np.concatenate((B, F))
    kmeans = KMeans(n_clusters=10, random_state=0).fit(X)
    
    #Cluster test pixels
    kp, des = sift.detectAndCompute(img, None)
    for k in range(len(kp)):
        x, y = kp[k].pt
        y, x = int(x), int(y)
        
        #Find pixel class
        cls = kmeans.predict([des[k]])[0]
        bcls = kmeans.predict(B)
        fcls = kmeans.predict(F)
        
        #Count bg and fg pixels
        wb = bcls.tolist().count(cls)
        wf = fcls.tolist().count(cls)
        
        #Calculate pixel prior
        pfw[x, y] = wf/(wf + wb)*1000
    
    #Normalize prior
    pfw = cv.GaussianBlur(pfw, (101,101), 17)
    
    #Return prior
    return pfw, count