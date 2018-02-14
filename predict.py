#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 1/23/18 8:02 PM 

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
	mask = mask.astype(np.uint8)
	# mask=cv2.medianBlur(mask, ksize=3)
	kernel = np.ones((2, 2), np.uint8)
	blurred = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations = 2)
	detected = cv2.dilate(blurred, kernel, iterations = n)

	fig = plt.figure()
	ax1 = fig.add_subplot(311)
	ax1.imshow(mask)
	ax2 = fig.add_subplot(312)
	ax2.imshow(blurred)
	ax3 = fig.add_subplot(313)
	ax3.imshow(detected)
	plt.show()
	return detected

def detect_rb(mask, img):
	K = np.matrix([[413.8794311523438, 0.0, 667.1858329372201], [0.0, 415.8518917846679, 500.6681745269121],
				   [0.0, 0.0, 1.0]])

	detected=refine_mask(mask, 5)
	mask_red=label(detected)
	props=regionprops(mask_red)
	#print(props[0].centroid)
	#print(props[0].bbox)

	# label image regions
	image_label_overlay = label2rgb(mask_red, image = img.astype(np.float32))

	fig, ax = plt.subplots(ncols = 1, nrows = 1, figsize = (6, 6))
	ax.imshow(image_label_overlay)

	restart = False
	for region in regionprops(mask_red):

		# skip small images
		if region.area < 3000:
			print('area too small:')
			print(region.area)
			print('')
			continue

		# draw rectangle around segmented coins
		minr, minc, maxr, maxc = region.bbox
		h=abs(maxr-minr)
		w=abs(maxc-minc)

		print('bbox h: '+str(h))
		print('bbox w: '+str(w))
		print('bbox area: '+str(h*w))
		print('pixel area: '+str(region.area))
		print('est: '+ str(region.area/(w*h)))
		print('h/w: ' +str(h/w))

		if 1.15 < h / w < 2:# and region.area /(w*h)>0.79:
		#	counter=counter+1
			rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,\
									  fill = False, edgecolor = 'red', linewidth = 2)
			ax.add_patch(rect)
			print(minr)
			print(minc)
			print(region.centroid)
			ax.scatter(minc+w/2, minr+h/2, c='r')

			if region.area /(w*h)<0.8:
				restart=True
	plt.show()

	return restart
'''
#cv2.imshow('mask', mask)
#cv2.imshow('detected', detected.astype(np.float32))

cv2.imshow('detected', detected.astype(np.float32))
k = cv2.waitKey(0) & 0xFF
cv2.destroyAllWindows()

# remove the regions that are clearly not rectangular shape
_, contours, _ = cv2.findContours(detected, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
areas = []
mask = np.zeros((img_hsv.shape[0], img_hsv.shape[1]), np.uint8)
for contour in contours:
	area = cv2.contourArea(contour)
	areas.append(areas)
	_, _, w, h = cv2.boundingRect(contour)
	print(w)
	print(h)
	rect_area = w * h
	extent = float(area) / rect_area
	if extent > 0.7:
		pts = contour.reshape((-1, 1, 2))
		cv2.fillPoly(mask, [pts], 255)
	#pdb.set_trace()

mask = cv2.dilate(mask, kernel, iterations = 3)
# choose the regions with proper height/width ratio as barrels
_, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
areas = []
bl_x, bl_y, tr_x, tr_y, dist = [], [], [], [], []
for contour in contours:
	area = cv2.contourArea(contour)
	areas.append(areas)
	x, y, w, h = cv2.boundingRect(contour)
	rect_area = w * h
	extent = float(area) / rect_area
	if extent > 0.8 and 1.15 < h / w < 2:
		bl_x.append(int(x))
		bl_y.append(int(y + h))
		tr_x.append(int(x + w))
		tr_y.append(int(y))
		pts = np.array([[x - w / 2, y - h / 2, 1], [x - w / 2, y + h / 2, 1], [x + w / 2, y - h / 2, 1],
						[x + w / 2, y + h / 2, 1]])
		pos = np.linalg.inv(K) * pts.T
		# calculate the distance with camera parameters
		distance = (0.93 / np.linalg.norm(pos[:, 0] - pos[:, 1]) + 0.93 / np.linalg.norm(pos[:, 2] - pos[:, 3]) \
				+ 0.53 / np.linalg.norm(pos[:, 1] - pos[:, 2]) + 0.53 / np.linalg.norm(pos[:, 1] - pos[:, 2])) \
				   / 4.0
		dist.append(distance)

return bl_x, bl_y, tr_x, tr_y, dist
'''

'''
img = util.img_as_ubyte(data.coins()) > 110
label_img = label(img, connectivity=img.ndim)
props = regionprops(label_img)

region = regionprops(np.array(pic, dtype=int))
region.area >= delta

minr, minc, maxr, maxc = region.bbox
rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, \
						  edgecolor='red', linewidth=2
ct=region.centroid

test = scipy.ndimage.morphology.binary_dilation(test,structure=np.ones((10,10)))
'''


if __name__ == '__main__':
	alpha = np.load('alpha.npy')
	sigma = np.load('sigma.npy', )
	mu = np.load('mu.npy')

	data, redpixels, imgs, imgs_hsv, distances = loadTestData()
	#for i in range(len(imgs)):
	for i in [1,4,5,6,7,8,9,10,11]:
		pic = imgs_hsv[i]
		is_red = redpixels[i]

		p = pred(pic, mu, sigma, alpha)
		mask=(p>max(-17.9999, np.percentile(p, 98)))
		'''
		print(np.percentile(p,97))
		plt.figure()
		plt.imshow(mask)
		plt.show()
		'''
		restart=detect_rb(mask, imgs[i])

		if restart:
			mask = (p > max(-17.9999, np.percentile(p, 95)))
			ct = detect_rb(mask, imgs[i])

