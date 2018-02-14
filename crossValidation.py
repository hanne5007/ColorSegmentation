#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 1/19/18 6:02 PM 

@author: Hantian Liu
"""

import numpy as np
from loadData import loadData
import pdb

def crossValidation(redpixels, imgs, distances):
	n=len(redpixels)
	chosen=np.floor(n/4)
	chosen=chosen.astype('int64')

	totest=np.random.choice(n,size=chosen, replace=False)

	rpixels_test=[]
	rpixels_train=[]
	imgs_test=[]
	imgs_train=[]
	dist_test=[]
	dist_train=[]
	for i in range(0, n):
		if any(totest==i):
			rpixels_test.append(redpixels[i])
			imgs_test.append(imgs[i])
			dist_test.append(distances[i])
		else:
			rpixels_train.append(redpixels[i])
			imgs_train.append(imgs[i])
			dist_train.append(distances[i])
	#pdb.set_trace()

	'''
	ind_totest=ind_totest.astype('int64')
	ind_totrain=ind_totrain.astype('int64')
	rpixels_test=redpixels[ind_totest]
	imgs_test=imgs[ind_totest]
	dist_test=distances[ind_totest]
	rpixels_train=redpixels[ind_totrain]
	imgs_train=imgs[ind_totrain]
	dist_train=distances[ind_totrain]
	'''
	np.save('rpixels_test.npy', rpixels_test)
	np.save('imgs_test.npy', imgs_test)
	np.save('dist_test.npy', dist_test)
	np.save('rpixels_train.npy', rpixels_train)
	np.save('imgs_train.npy', imgs_train)
	np.save('dist_train.npy', dist_train)
	return