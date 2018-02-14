#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 1/24/18 4:58 PM 

@author: Hantian Liu
"""

import numpy as np
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.morphology import label, closing, square
from skimage.measure import regionprops
from skimage.color import label2rgb
from loadTrainingData import loadTestData
from EM import pdf
import pdb


def pred(img, mu, sigma, alpha):
	"""

	:param img: h*w*3
	:param mu: k*3
	:param sigma: k*3*3 diag
	:param alpha: k*1
	:return: p h*w
	"""
	h, w, z = np.shape(img)
	n = h * w
	data = np.zeros([n, 3])
	data[:, 0] = img[:, :, 0].flatten()
	data[:, 1] = img[:, :, 1].flatten()
	data[:, 2] = img[:, :, 2].flatten()
	k, dim = np.shape(mu)

	alphamat=np.tile(alpha.transpose(), (n,1))
	px = np.zeros([k, n])
	for j in range(0, k):
		px[j,:] = pdf(data, sigma[j], mu[j, :])
	px = px.transpose()

	p=px*alphamat
	p=np.sum(p, axis=1)
	p = np.reshape(np.log(p), [h, w])
	return p

def refine_mask(mask, n):
	"""

	:param mask: h*w mask of red labels
	:param n: scalar for dilation iteration
	:return: detected h*w refined mask
	"""
	mask = mask.astype(np.uint8)
	#blurred=cv2.medianBlur(mask, ksize=3)
	kernel = np.ones((2, 2), np.uint8)
	erosed = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations = 2)
	closing= cv2.morphologyEx(erosed, cv2.MORPH_CLOSE, kernel)
	detected = cv2.dilate(closing, kernel, iterations = n)

	fig = plt.figure()
	ax1 = fig.add_subplot(221)
	ax1.imshow(mask)
	ax4 = fig.add_subplot(222)
	ax4.imshow(erosed)
	ax2 = fig.add_subplot(223)
	ax2.imshow(closing)
	ax3 = fig.add_subplot(224)
	ax3.imshow(detected)
	plt.show()
	return detected

def detect_rb(mask, img, is_first):
	"""

	:param mask: h*w mask of red labels
	:param img: h*w*3
	:param is_first: boolean for first detection
	:return: restart boolean for restart to choose right region,
		bbox m*4 each row composed of the [minimum row, minimum column, maximum row, maximum column] of bounding box,
		distances m*1 estimated distance , (m is number of red barrels detected)
		height m*1 bbox height,
		width m*1 bbox width
	"""
	detected=refine_mask(mask, 5)
	mask_red=label(detected)

	# label image regions
	#image_label_overlay = label2rgb(mask_red, image = img.astype(np.float32))
	#fig, ax = plt.subplots(ncols = 1, nrows = 1, figsize = (6, 6))
	#ax.imshow(image_label_overlay)
	#ax.imshow(img)

	coeff = 512.29804576
	if is_first:
		restart = False
	distances=[]
	bbox=[]
	height = []
	width = []
	for region in regionprops(mask_red):
		# skip small images
		if region.area < 1100:
			#print('area too small:')
			#print(region.area)
			#print('')
			continue
		# draw rectangle around segmented coins
		minr, minc, maxr, maxc = region.bbox
		h=abs(maxr-minr)
		w=abs(maxc-minc)

		'''
		print('bbox h: ' + str(h))
		print('bbox w: ' + str(w))
		print('bbox area: ' + str(h * w))
		print('pixel area: ' + str(region.area))
		print('est: ' + str(region.area / (w * h)))
		print('h/w: ' + str(h / w))
		'''
		# choose the reasonable region via h/w ratio and area/bbox_area ratio
		if 1.15 < h / w < 2:
			#pdb.set_trace()
			if (region.area /(w*h))<0.5 and is_first==False:
				#restart=True
				#print('nonononononononooooooooooo')
				continue
			area=w*h
			pred_dist=coeff*area**(-0.5)

			distances.append(pred_dist)
			bbox.append(np.asarray([minr, minc, maxr, maxc]))
			height.append(h)
			width.append(w)
			'''
			rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, \
									  fill = False, edgecolor = 'red', linewidth = 2)
			ax.add_patch(rect)
			print(minr)
			print(minc)
			print(region.centroid)
			ax.scatter(minc+w/2, minr+h/2, c='r')
			plt.show()
			'''
	#if restart==True and len(distances)>0:
	if len(distances)==0:
		restart=True
	else:
		restart=False

	return restart, bbox, distances, height, width

def simple_detect_rb(mask, img, is_first):
	"""
	('simple' for this function does not require mask to be refined)

	:param mask: h*w mask of red labels
	:param img: h*w*3
	:param is_first: boolean for first detection
	:return: restart boolean for restart to choose right region,
		bbox m*4 each row composed of the [minimum row, minimum column, maximum row, maximum column] of bounding box,
		distances m*1 estimated distance , (m is number of red barrels detected)
		height m*1 bbox height,
		width m*1 bbox width
	"""
	mask_red=label(mask)

	#fig, ax = plt.subplots(ncols = 1, nrows = 1, figsize = (6, 6))
	#ax.imshow(img)

	coeff = 512.29804576
	if is_first:
		restart = False
	distances=[]
	bbox=[]
	height=[]
	width=[]
	for region in regionprops(mask_red):
		# skip small images
		if region.area < 1100:
			#print('area too small:')
			#print(region.area)
			#print('')
			continue

		# draw rectangle around segmented coins
		minr, minc, maxr, maxc = region.bbox
		h=abs(maxr-minr)
		w=abs(maxc-minc)
		'''
		print('bbox h: ' + str(h))
		print('bbox w: ' + str(w))
		print('bbox area: ' + str(h * w))
		print('pixel area: ' + str(region.area))
		print('est: ' + str(region.area / (w * h)))
		print('h/w: ' + str(h / w))
		'''
		if 1.15 < h / w < 2:
			if (region.area /(w*h))<0.5 and is_first==False:
				continue
			area=w*h
			pred_dist=coeff*area**(-0.5)

			distances.append(pred_dist)
			bbox.append(np.asarray([minr, minc, maxr, maxc]))
			height.append(h)
			width.append(w)

	if len(distances)==0:
		restart=True
	else:
		restart=False
	return restart, bbox, distances, height, width



if __name__ == '__main__':
	alpha = np.load('alpha.npy')
	sigma = np.load('sigma.npy', )
	mu = np.load('mu.npy')

	data, redpixels, imgs, imgs_hsv, distances = loadTestData()

	imgs = np.load('imgs_test.npy')

	for i in range(len(imgs)):
	#for i in [5,6]:
		pic = imgs_hsv[i]
		is_red = redpixels[i]

		p = pred(pic, mu, sigma, alpha)
		#mask=(p>max(-17.9999, np.percentile(p, 97)))
		mask=(p>-14)
		#print(np.percentile(p,97))
		#plt.figure()
		#plt.imshow(mask)
		#plt.show()
		
		restart, bbox, distances, h, w=detect_rb(mask, imgs[i], True)

		if restart:
			mask = (p > max(-17.9999, np.percentile(p, 95)))
			restart, bbox, distances, h, w = detect_rb(mask, imgs[i], False)

		if len(bbox)==0:
			restart, bbox, distances, h, w = simple_detect_rb(mask, imgs[i], True)
			if restart:
				mask = (p > max(-17.9999, np.percentile(p, 95)))
				restart, bbox, distances, h, w = detect_rb(mask, imgs[i], False)


		img=imgs[i].copy()
		img=cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
		for j in range(len(bbox)):
			sides=bbox[j]
			img = cv2.rectangle(img, (sides[1], sides[0]), (sides[3], sides[2]), (0, 255, 0), 2)
			cv2.putText(img, 'd = {0}m'.format(str(distances[j])), (sides[1], sides[0]-5), cv2.FONT_HERSHEY_SIMPLEX,\
						1, (0, 255, 0), 2, cv2.LINE_AA)
			cv2.circle(img, (int(sides[1]+w[j]/2), int(sides[0]+h[j]/2)), 4, (0,255,255), thickness = 6)
		cv2.imwrite(str(i)+'.jpg', img)
		cv2.imshow('image', img)
		k = cv2.waitKey(0) & 0xFF
		cv2.destroyAllWindows()

