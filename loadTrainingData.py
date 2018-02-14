#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 1/19/18 4:33 PM 

@author: Hantian Liu
"""

import numpy as np
import cv2, os
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pdb


def loadTrainingData():
	rpixels=np.load('rpixels_train.npy')
	imgs=np.load('imgs_train.npy')
	dist=np.load('dist_train.npy')

	data = np.zeros([1, 3])
	data_other = np.zeros([1, 3])
	for i in range(len(imgs)):
		pic=imgs[i]
		pic=pic/(1-0)*(255-0)+0
		pic = cv2.cvtColor(pic, cv2.COLOR_BGR2HSV)#2YCR_CB)

		r = pic[:, :, 0]
		g = pic[:, :, 1]
		b = pic[:, :, 2]

		g = g / (1 - 0) * (255 - 0) + 0

		is_red = rpixels[i]

		red = r[is_red]
		#pdb.set_trace()
		green = g[is_red]
		blue = b[is_red]
		red = np.array([red])
		green = np.array([green])
		blue = np.array([blue])
		rg = np.append(red.transpose(), green.transpose(), axis = 1)
		rgb = np.append(rg, blue.transpose(), axis = 1)
		data = np.append(data, rgb, axis = 0)
		'''
		not_red=~(is_red)
		nred = r[not_red]
		ngreen = g[not_red]
		nblue = b[not_red]
		nred = np.array([nred])
		ngreen = np.array([ngreen])
		nblue = np.array([nblue])
		nrg = np.append(nred.transpose(), ngreen.transpose(), axis = 1)
		nrgb = np.append(nrg, nblue.transpose(), axis = 1)
		data_other = np.append(data_other, nrgb, axis = 0)
		#pdb.set_trace()
		'''

	data = data[1:, :]
	#data_other = data_other[1:,:]
	#Pred= len(data)/(len(data)+len(data_other))
	print(len(data))
	#print(len(data_other))
	#print(Pred)
	return data#, data_other, Pred


def loadTestData():
	rpixels=np.load('rpixels_test.npy')
	imgs=np.load('imgs_test.npy')
	dist=np.load('dist_test.npy')

	imgs_hsv=[]
	data = np.zeros([1, 3])
	data_other = np.zeros([1, 3])
	for i in range(len(imgs)):
		pic=imgs[i]
		pic = pic / (1 - 0) * (255 - 0) + 0
		pic = cv2.cvtColor(pic, cv2.COLOR_BGR2HSV)#2YCR_CB)
		pic[:, :, 1] = pic[:, :, 1] / (1 - 0) * (255 - 0) + 0
		imgs_hsv.append(pic)

		r = pic[:, :, 0]
		g = pic[:, :, 1]
		b = pic[:, :, 2]

		is_red = rpixels[i]

		red = r[is_red]
		green = g[is_red]
		blue = b[is_red]
		red = np.array([red])
		green = np.array([green])
		blue = np.array([blue])
		rg = np.append(red.transpose(), green.transpose(), axis = 1)
		rgb = np.append(rg, blue.transpose(), axis = 1)
		data = np.append(data, rgb, axis = 0)

	data = data[1:, :]
	#print(len(data))
	return data, rpixels, imgs, imgs_hsv, dist


if __name__ == '__main__':
	#loadTrainingData()
	loadTestData()
