#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 1/21/18 4:03 PM 

@author: Hantian Liu
"""

import numpy as np
import pdb
import math
import matplotlib
from loadTrainingData import loadTrainingData
from loadTrainingData import loadTestData
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from loadData import loadData


def pdf(x, sigma, mu):
	"""

	:param x: n*3
	:param sigma: 3*3 diag
	:param mu: 1*3 (for certain k)
	:return: prob 1*n
	"""
	n, dim = np.shape(x)
	A = np.linalg.inv(sigma)
	# prod = A[0, 0] * (diff[:, 0] ** 2 + diff[:, 1] ** 2 + diff[:, 2] ** 2)
	diff = x - mu
	prod = np.multiply(np.dot(diff, A), diff)
	prod = np.sum(prod, axis = 1)
	# prod=np.dot(diff.transpose(),sigma)
	# prod=np.dot(prod, diff)
	prob = ( 1/np.linalg.det(sigma) / ((2 * np.pi) ** dim))**0.5 * np.exp(-0.5 * prod)

	return prob


def initialize(k, data):
	"""

	:param k: scalar
	:param data: n*3 RGB
	:return: alpha k*1, sigma k*3*3 diag, mu k*3
	"""
	n, dim = np.shape(data)
	sigma=[]
	for i in range(k):
		sigmak=np.array([[np.amax(data[:,0])-np.amin(data[:,0]), 0, 0],\
						 [0, np.amax(data[:,1])-np.amin(data[:,1]), 0],\
						 [0, 0, np.amax(data[:,2])-np.amin(data[:,2])]])
		sigma.append(sigmak/2)

	alpha = np.ones([k, 1])
	alpha = alpha / k
	datamean = np.mean(data, axis = 0)

	mu = data[np.random.choice(n, k, replace = False), :]
	mu = np.reshape(mu, [k, 3])
	'''
	mu = np.random.choice(100, size = 3*k, replace = False)
	mu = np.reshape(mu, [k, 3])
	mu = mu / np.linalg.norm(mu) *datamean
	
	mu = np.zeros([k, 3])
	for i in range(0,k):
		mu[i,:]=datamean-i
	'''
	return alpha, sigma, mu


def Estep(x, mu, sigma, alpha):
	"""

	:param x: n*3
	:param mu: k*3
	:param sigma: k*3*3 diag
	:param alpha: k*1
	:return: r n*k
	"""
	n, dim = np.shape(x)
	k, dim = np.shape(mu)
	r = np.zeros([n, k])
	'''
	for i in range(0, n):
		sum=0
		for j in range(0, k):
			sum=sum+alpha[j,:]*pdf(x[i,:], sigma[j], mu[j,:])
		for z in range(0, k):
			r[i, z]=alpha[z,:]*pdf(x[i,:], sigma[z], mu[z,:])/sum
	
	sum=np.zeros([1,n])
	for j in range(0, k):
		sum=sum+alpha[j, :] * pdf(x, sigma[j], mu[j, :])
	'''
	for z in range(0, k):
		r[:, z] = alpha[z, :] * pdf(x, sigma[z], mu[z, :])
	# pdb.set_trace()
	r = np.divide(r, np.tile(np.sum(r, axis = 1), (k, 1)).transpose())
	return r


def Mstep(x, r):
	"""

	:param x: n*3
	:param r: n*k
	:return: updated
			 alpha k*1, sigma k*3*3 diag, mu k*3
	"""
	n, k = np.shape(r)
	sigma = [np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])] * k
	mu = np.zeros([k, 3])
	alpha = np.zeros([k, 1])
	for i in range(0, k):
		alpha[i, :] = 1 / n * sum(r[:, i:i + 1])
		mu[i, 0] = sum(r[:, i:i + 1] * x[:, 0:1]) / sum(r[:, i:i + 1])
		mu[i, 1] = sum(r[:, i:i + 1] * x[:, 1:2]) / sum(r[:, i:i + 1])
		mu[i, 2] = sum(r[:, i:i + 1] * x[:, 2:3]) / sum(r[:, i:i + 1])

		diff = x - mu[i:i + 1, :]
		# diff2 = diff[:, 0:1] ** 2 + diff[:, 1:2] ** 2 + diff[:, 2:3] ** 2
		# sigmak = np.sum(diff2*r[:, i:i+1]) / sum(r[:, i:i+1]) #/ n
		# sigma[i] = sigma[i] * sigmak

		rcut = r[:, i:i + 1]
		rcut = rcut[:, :, np.newaxis]
		pvalue = np.ones([n, 3, 3])
		pvalue = rcut * pvalue  # n*3*3

		span = diff[:, :, np.newaxis]
		diffmat = np.ones([n, 3, 3])
		diffmat = span * diffmat  # n*3*3
		diffmat_t = np.transpose(diffmat, (0, 2, 1))  # n*3*3

		sigmak = np.multiply(pvalue, np.multiply(diffmat, diffmat_t))
		sigmak = np.sum(sigmak, axis = 0)
		sigmak = sigmak / sum(rcut)
		sigma[i] = np.multiply(sigma[i], sigmak)

	return alpha, sigma, mu


def EM(k, x, epsilon):
	"""

	:param k: hyperparameter
	:param x: n*3 RGB pixels
	:param epsilon: tolerance
	:return: final
			 alpha k*1, sigma k*3*3 diag, mu k*3, r
	"""
	alpha, sigma, mu = initialize(k, x)
	n, dim = np.shape(x)
	epoch = 0
	log_prev = 20000
	log_curr = 10000

	while epoch < 1000 and abs(log_prev - log_curr) > epsilon:
		print('epoch: ' + str(epoch + 1))
		epoch = epoch + 1
		log_prev = log_curr
		alphaprev = alpha
		sigmaprev = sigma
		muprev = mu
		r = Estep(x, muprev, sigmaprev, alphaprev)
		alpha, sigma, mu = Mstep(x, r)
		# print(alpha)
		# print(sigma)
		# print(mu)
		'''
		log_curr=0
		for i in range(0, n):
			pper=0
			for j in range(0, k):
				#pdb.set_trace()
				pper = pper + alpha[j, :] * pdf(x[i:i+1,:], sigma[j], mu[j, :])
			log_curr=log_curr+np.log(pper)
		'''
		alphamat = np.tile(alpha.transpose(), (n, 1))
		px = np.zeros([k, n])
		for j in range(0, k):
			px[j, :] = pdf(x, sigma[j], mu[j, :])
		px = px.transpose()

		p = px * alphamat
		p = np.sum(p, axis = 1)
		log_curr=np.sum(np.log(p))
		'''

		p = np.zeros([1, n])
		for j in range(0, k):
			p = p + np.log(alpha[j, :] * pdf(data, sigma[j], mu[j, :]))
		p = p.transpose()
		log_curr = np.sum(p)
		'''

		print('log likelihood: ')
		print('previous: ' + str(log_prev))
		print('now: ' + str(log_curr))
		print('')
	return alpha, sigma, mu, r


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

	'''
	p = np.zeros([1, n])
	for j in range(0, k):
		p = p + np.log(alpha[j, :] * pdf(data, sigma[j], mu[j, :]))
	p = p.transpose()
	p = np.reshape(p, [h, w])
	'''

	alphamat = np.tile(alpha.transpose(), (n, 1))
	px = np.zeros([k, n])
	for j in range(0, k):
		px[j, :] = pdf(data, sigma[j], mu[j, :])
	px = px.transpose()

	p = px * alphamat
	p = np.sum(p, axis = 1)
	p = np.reshape(np.log(p), [h, w])

	return p


if __name__ == '__main__':

	data = loadTrainingData()
	'''
	fig=plt.figure()
	ax = fig.add_subplot(111, projection = '3d')
	ax.scatter(data[:,0],data[:,1],data[:,2])
	plt.show()
	

	k = 4
	epsilon = 0.001
	alpha, sigma, mu, r = EM(k, data, epsilon)
	np.save('alpha.npy', alpha)
	np.save('sigma.npy', sigma)
	np.save('mu.npy', mu)

	'''
	alpha=np.load('alpha.npy')
	sigma=np.load('sigma.npy',)
	mu=np.load('mu.npy')

	data, redpixels, imgs_rgb, imgs, distances = loadTestData()
	for i in range(len(imgs)):
		pic = imgs[i]
		is_red=redpixels[i]

		p = pred(pic, mu, sigma, alpha)

		#check=p[is_red]

		pdb.set_trace()

		fig=plt.figure()
		ax1=fig.add_subplot(311)
		ax1.imshow(pic)
		ax2=fig.add_subplot(312)
		ax2.imshow(is_red)
		ax3=fig.add_subplot(313)
		ax3.imshow(p > np.percentile(p, 98))
		plt.show()
	pdb.set_trace()
