#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 1/24/18 4:42 PM 

@author: Hantian Liu
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pdb

def getDistances(area_train, dist_train, area_test, dist_test):
	# Create linear regression object
	regr = linear_model.LinearRegression()

	# Train the model using the training sets
	regr.fit(area_train, dist_train)

	# Make predictions using the testing set
	pred = regr.predict(area_test)

	# The coefficients
	print('Coefficients: \n', regr.coef_)
	# The mean squared error
	print("Mean squared error: %.2f" \
		  % mean_squared_error(dist_test, pred))
	# Explained variance score: 1 is perfect prediction
	print('Variance score: %.2f' % r2_score(dist_test, pred))

	# Plot outputs
	plt.figure()
	plt.scatter(area_train, dist_train, color = 'black')
	plt.scatter(area_test, dist_test, color = 'red')
	plt.plot(area_test, pred, color = 'blue', linewidth = 3)

	plt.xticks(())
	plt.yticks(())

	plt.show()
	return regr.coef_


def loadArea(is_train):
	if is_train:
		rpixels = np.load('rpixels_train.npy')
		imgs = np.load('imgs_train.npy')
		dist = np.load('dist_train.npy')
	else:
		rpixels=np.load('rpixels_test.npy')
		imgs=np.load('imgs_test.npy')
		dist=np.load('dist_test.npy')

	data = []
	for i in range(len(imgs)):
		is_red = rpixels[i]
		rows=np.where(is_red)[0]
		cols=np.where(is_red)[1]
		#pdb.set_trace()
		h=np.max(rows)-np.min(rows)
		w=np.max(cols)-np.min(cols)
		area=h*w
		data.append(area)
	data=np.asarray(data)
	data=data[:,np.newaxis]
	data=1/np.sqrt(data)

	dist=dist[:,np.newaxis]

	if is_train:
		valid_data=np.zeros([np.shape(data)[0]-1,np.shape(data)[1]])
		valid_data[0:6, :] = data[0:6, :]
		valid_data[6:, :] = data[7:, :]
		valid_dist = np.zeros([np.shape(dist)[0] - 1, np.shape(dist)[1]])
		valid_dist[0:6,:]=dist[0:6,:]
		valid_dist[6:,:]=dist[7:,:]
		return valid_data, valid_dist
	return data, dist


if __name__ == '__main__':
	area_train, dist_train =loadArea(True)
	area_test, dist_test=loadArea(False)
	'''
	fig=plt.figure()
	ax=fig.add_subplot(111)
	for i in range(len(area_train)):
		print('i: '+str(i))
		print(1/area_train[i]*dist_train[i])
		ax.scatter(area_train[i], dist_train[i], color = 'black')
	plt.show()
	'''
	coeff=getDistances(area_train, dist_train, area_test, dist_test)
	print('\ndistances=1/sqrt(barrel area)*'+str(coeff[0]))
