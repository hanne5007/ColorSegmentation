#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 1/19/18 3:31 PM 

@author: Hantian Liu
"""

import numpy as np
import cv2, os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from detect import detect_rb, simple_detect_rb, refine_mask, pred
import pdb

model_folder='./models/HSVk=4'

############################
## MODIFY THESE VARIABLES ##
############################
test_folder = "./test"
############################

def myAlgorithm(image):
	"""

	:param img: h*w*3 RGB
	:return: x centroid, y centroid, d distance
	"""
	# load saved GMM model of red barrel
	alpha = np.load(os.path.join(model_folder, 'alpha.npy'))
	sigma = np.load(os.path.join(model_folder, 'sigma.npy'))
	mu = np.load(os.path.join(model_folder, 'mu.npy'))

	# prediction on new test image and get mask of red barrels
	pic=image.copy()
	pic = pic / (1 - 0) * (255 - 0) + 0
	pic = cv2.cvtColor(pic, cv2.COLOR_BGR2HSV)
	pic[:, :, 1] = pic[:, :, 1] / (1 - 0) * (255 - 0) + 0
	p = pred(pic, mu, sigma, alpha)
	mask =(p>-14)
	'''
	mask = (p > max(-17.9999, np.percentile(p, 97)))

	print('97+: '+str(np.percentile(p, 97)))
	print('95+: '+str(np.percentile(p, 95)))
	fig = plt.figure()
	ax1 = fig.add_subplot(131)
	ax1.imshow((p>-14))
	ax3 = fig.add_subplot(132)
	ax3.imshow(image)
	ax4 = fig.add_subplot(133)
	ax4.imshow((p>-13))
	plt.show()
	'''

	# plot bounding box and calculate distance
	restart, bbox, distances, h, w = detect_rb(mask, pic, True)

	# try increase threshold
	thres=-15
	while restart and thres>=-18:
		mask = (p > thres)
		restart, bbox, distances, h, w = detect_rb(mask, pic, False)
		thres=thres-1

	# try stronger dilation
	if restart:
		mask =(p>-14)
		refined_mask=refine_mask(mask, 15)
		restart, bbox, distances, h, w = detect_rb(refined_mask, pic, False)

	return bbox, distances, h, w


def displayResult(folder):

	for filename in os.listdir(folder):
		if filename == ".DS_Store":
			continue
		# read one test image
		pic = plt.imread(os.path.join(folder, filename))

		# Your computations here!
		bbox, distances, h, w = myAlgorithm(pic)
		
		# Display results:
		# (1) Segmented image
		# (2) Barrel bounding box
		# (3) Distance of barrel
		print(filename + ' has '+str(len(bbox))+ ' red barrels. ')
		if len(bbox)==0:
			print('So sad...\n')

		img = cv2.imread(os.path.join(folder, filename))
		for j in range(len(bbox)):
			sides = bbox[j]
			img = cv2.rectangle(img, (sides[1], sides[0]), (sides[3], sides[2]), (0, 255, 0), 2)
			cv2.putText(img, 'd = {0}m'.format(str(distances[j])), (sides[1], sides[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, \
						1, (0, 255, 0), 2, cv2.LINE_AA)
			cv2.circle(img, (int(sides[1] + w[j] / 2), int(sides[0] + h[j] / 2)), 4, (0, 255, 255), thickness = 6)
			print(' Barrel '+str(j+1))
			print('  CentroidX: '+ str(sides[1] + w[j] / 2))
			print('  CentroidY: ' + str(sides[0] + h[j] / 2))
			print('  Distance: '+ str(distances[j]))
		print(' ')
		cv2.imshow('image', img)
		cv2.imwrite('new'+filename, img)
		key = cv2.waitKey(0)
		cv2.destroyAllWindows()

	return

if __name__ == '__main__':
	displayResult(test_folder)