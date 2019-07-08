#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: owen-at-MISL
@email: om82@drexel.edu
"""

import numpy as np
from tqdm import tqdm
import tensorflow as tf

def softmax(a): #function to calculate softmax
    e = np.exp(a)
    div = np.tile(e.sum(1,keepdims=1),(1,a.shape[1]))
    sm = e/div
    return sm
    
def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]
    

#calculate forensic similarity in a tensorflow session.
#use this if you will be calculating forensic similarity multiple times, and do not want to create a tensorflow session each time

def calculate_forensic_similarity_insession(X1,X2,patch_size, sess, compare_output, tf_x1, tf_x2, MISL_phase,  batch_size = 48, quiet = False):

	if not len(X1)==len(X2):
		raise ValueError('Inputs must be the same length')
        
	list_sim = [] #intialize list of similarity values

	for ix in tqdm(batch(range(len(X1)), batch_size),total = int(np.ceil(len(X1)/batch_size)),
			desc='Calculating Forensic Similarity',disable=quiet): #batchify
		#calculate output
		result = sess.run(compare_output, feed_dict={tf_x1:X1[ix],tf_x2:X2[ix], MISL_phase:False})

		sim_batch = softmax(result)[:,1] #second value is similarity (first value/0th index is dissimilarity and not used)
		list_sim.append(sim_batch)
	#numpyify
	similarity = np.hstack(list_sim)
	return similarity


def calculate_forensic_similarity(X1, X2, f_pretrained_weights, patch_size, batch_size=48,quiet=False):
    
	#Load CompareNet Model
	if patch_size == 256:
		from model import Similarity_256 as SimilarityNetwork
	elif patch_size == 128:
		from model import Similarity_128 as SimilarityNetwork
	else:
	    raise TypeError('Unsupported patch size {}'.format(patch_size))

	#reset tf
	tf.reset_default_graph()    
	#PLACE HOLDERS
	tf_x1 = tf.placeholder(tf.float32, shape=[None,patch_size,patch_size,3], name='input_data1')
	tf_x2 = tf.placeholder(tf.float32, shape=[None,patch_size,patch_size,3], name='input_data2')
	MISL_phase =tf.placeholder(tf.bool, name='phase')
	#CREATE NETWORK
	compare_output = SimilarityNetwork(tf_x1,tf_x2,MISL_phase)
	#initialize saver
	mislnet_restore = tf.train.Saver()

	with tf.Session() as sess:
	    mislnet_restore.restore(sess,f_pretrained_weights) #load pretrained network
	    similarity = calculate_forensic_similarity_insession(X1, X2, patch_size, sess, compare_output,
								 tf_x1, tf_x2, MISL_phase, batch_size = batch_size, quiet = quiet)


	return similarity
