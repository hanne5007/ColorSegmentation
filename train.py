#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 1/24/18 9:47 PM 

@author: Hantian Liu
"""
from EM import EM
import numpy as np
from loadTrainingData import loadTrainingData, loadTestData


############################
## MODIFY THESE VARIABLES ##
############################
# number of cluster of GMM model
k = 4

# EM convergence
epsilon = 0.001
############################


if __name__ == '__main__':
	data = loadTrainingData()
	'''
	fig=plt.figure()
	ax = fig.add_subplot(111, projection = '3d')
	ax.scatter(data[:,0],data[:,1],data[:,2])
	plt.show()
	'''

	alpha, sigma, mu, r = EM(k, data, epsilon)
	np.save('alpha.npy', alpha)
	np.save('sigma.npy', sigma)
	np.save('mu.npy', mu)

	'''
	alpha = np.load('alpha.npy')
	sigma = np.load('sigma.npy', )
	mu = np.load('mu.npy')
	
	data, redpixels, imgs_rgb, imgs, distances = loadTestData()
	for i in range(len(imgs)):
		pic = imgs[i]
		is_red = redpixels[i]
	
		p = pred(pic, mu, sigma, alpha)
	
		# check=p[is_red]
		pdb.set_trace()
	
		fig = plt.figure()
		ax1 = fig.add_subplot(311)
		ax1.imshow(pic)
		ax2 = fig.add_subplot(312)
		ax2.imshow(is_red)
		ax3 = fig.add_subplot(313)
		ax3.imshow(p > np.percentile(p, 98))
		plt.show()
	pdb.set_trace()
	'''