#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 11:55:45 2018

@author: vinusankars
"""

import cv2 as cv
from appearancePrior import appearancePrior as ap
from geometricPrior import geometricPrior as gp
from spixel import spixel as sp
import matplotlib.pyplot as plt
import numpy as np

heat = 0
pgf, pfw = 0, 0

def fg_extract(img, ka=100, kg=100, threshold=20):
    global heat, pgf, pfw
    img = cv.resize(img, (400,400))
    
    #Displaying the test image
    plt.title('Test Image')
    plt.axis("off")
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.show()

    #Displaying the superpixel segment image
    print('\nSuperpixel-ing...')
    seg, bound = sp(img, n=100)

    #Defining hyperparameters for prior calculations
    kg = ka
    ka = kg

    #obtaining appearance priors
    print('\nGetting appearance prior...')
    pfw, ca = ap(img, ka)
    plt.imshow(pfw)
    plt.axis("off")
    plt.title('Appearance prior')
    plt.show()

    #obtaining geometric priors
    print('\nGetting geometric prior...')
    pgf, cg = gp(img, kg)
    plt.imshow(pgf)
    plt.axis("off")
    plt.title('Geometric prior')
    plt.show()

    #setting parameters for energy map
    heat = np.zeros(seg.shape)
    M = np.max(seg)+1
    seg_dic = {}
    seg_count = {}
    x_c = {}
    y_c = {}
    gamma = 0.5
    threshold = threshold

    #getting energy map
    print('\nGetting energy map...')
    for i in range(M):
        seg_dic[i] = 0
        seg_count[i] = 0
        x_c[i] = 0
        y_c[i] = 0
    
    #adding superpixel information
    for x in range(seg.shape[0]):
        for y in range(seg.shape[1]):
            seg_dic[seg[x, y]] += pfw[x, y]*pgf[x, y]**gamma
            seg_count[seg[x, y]] += 1
            x_c[seg[x, y]] += x
            y_c[seg[x, y]] += y

    #Quantizing the energy map
    for i in range(len(seg_dic)):
        seg_dic[i] = seg_dic[i]/seg_count[i]
        x_c[i] = x_c[i]/seg_count[i]
        y_c[i] = x_c[i]/seg_count[i]
    
    #generate energy map
    for x in range(heat.shape[0]):
        for y in range(heat.shape[1]):
            heat[x, y] = seg_dic[seg[x, y]]   

    #display energy map
    heat = cv.normalize(heat, heat, 0, 255, cv.NORM_MINMAX).astype('uint8')  
    plt.imshow(heat)
    plt.axis("off")
    plt.title('Energy map')
    plt.show() 

    #extract foreground from enrgy map
    print('\nExtracting foreground...')
    fg = np.zeros(img.shape).astype('uint8')

    for x in range(heat.shape[0]):
        for y in range(heat.shape[1]):
            if heat[x, y] > threshold:
                fg[x, y] = ((img[x, y] + [0, 255, 0])/2).astype('uint8')
            else:
                fg[x, y] = img[x, y]

    #display forground        
    plt.imshow(fg)
    plt.axis("off")
    plt.title('Extracted foreground')
    plt.show()

#run main
if __name__ == "__main__":
	img = cv.imread('2009_004317.jpg')
	threshold = int(input('Enter threshold value: '))
	fg_extract(img, 20, 20, threshold)