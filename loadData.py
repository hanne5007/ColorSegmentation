#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 1/21/18 2:51 PM 

@author: Hantian Liu
"""

import numpy as np
import cv2, os
from PIL import Image
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pdb

def loadData():
	imfolder = "./allimages"
	folder = "./labeled_data/RedBarrel"

	redpixels=[]
	imgs=[]
	distances=[]
	for filename in os.listdir(imfolder):
		if filename == ".DS_Store":
			continue
		im_path = os.path.join(imfolder, filename)
		pic = plt.imread(im_path)
		#pic = cv2.cvtColor(pic, cv2.COLOR_BGR2HSV)#2YCR_CB)
		imgs.append(pic)

		name = filename.split(".")
		npyname = name[0] + '.' + (name[1] != 'png') * (name[1] + '.') + 'npy'
		npy_path = os.path.join(folder, npyname)
		is_red = np.load(npy_path)
		redpixels.append(is_red)

		dist=float(name[0])
		distances.append(dist)
		#pdb.set_trace()

	#print(np.shape(redpixels))
	#print(np.shape(imgs))
	#print(np.shape(distances))
	#pdb.set_trace()
	return redpixels, imgs, distances


if __name__ == '__main__':
	loadData()