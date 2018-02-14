#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 1/21/18 3:16 PM 

@author: Hantian Liu
"""

from crossValidation import crossValidation
from loadData import loadData
import numpy as np

if __name__ == '__main__':
	redpixels, imgs, distances=loadData()
	crossValidation(redpixels, imgs, distances)


