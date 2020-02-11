#coding:utf-8
import codecs
import os
import random

import numpy as np
import tensorflow as tf

class Model(object):
	'''
	base model for knowledge graph embedding
	'''
	def __init__(self, config):
		'''

		:param config:
		'''

		self.config = config

		self.data_path = config.data_path

		self.ent_num = config.ent_num
		self.rel_num = config.rel_num
		self.neg_num = config.neg_num
		# self.type_num = config.type_num
	
		self.embed_dim = config.embed_dim
		self.margin = config.margin
	
		with tf.name_scope("input"):
			self.add_input()
			
		with tf.name_scope("cal_neg"):
			self.cal_neg_sample()

		with tf.name_scope("embedding"):
			self.embedding_def()
			self.add_embed()

		with tf.name_scope("loss"):
			self.add_loss()
			self.add_neg_loss()

		with tf.name_scope("predict"):
			self.add_predict()
	
	def embedding_def(self):
		pass
	
	def add_input(self):
		'''

		:return:
		'''
		self.pos_h = tf.placeholder(tf.int32, [None])
		self.pos_t = tf.placeholder(tf.int32, [None])
		self.pos_r = tf.placeholder(tf.int32, [None])

		self.neg_hs = tf.placeholder(tf.int32, [None, self.neg_num])
		self.neg_ts = tf.placeholder(tf.int32, [None, self.neg_num])
		self.neg_rs = tf.placeholder(tf.int32, [None, self.neg_num])
		self.neg_type_sims = tf.placeholder(tf.float32, [None, self.neg_num, 1])
		self.neg_rel_sims = tf.placeholder(tf.float32, [None, self.neg_num, 1])

		# self.ent_type = tf.placeholder(tf.int32, [None, self.type_num])
		# self.co_rel = tf.placeholder(tf.int32, [None, self.rel_num])
	
	def cal_neg_sample(self):
		'''

		:return:
		'''
		sim_input = tf.concat([self.neg_type_sims, self.neg_rel_sims], -1)
		neg_fc1 = tf.layers.dense(sim_input, units=32, activation=tf.nn.relu, name='neg_fc1')
		neg_sim = tf.layers.dense(neg_fc1, units=1, activation=tf.nn.relu, name='neg_fc2')
		self.neg_sim = tf.reduce_mean(neg_sim, -1)
		
		neg_idx_0 = tf.reshape(np.arange(0, self.neg_num), [-1, 1])
		neg_idx_1 = tf.reshape(tf.argmax(neg_sim, -1), [-1, 1])
		neg_idx = tf.concat([neg_idx_0, neg_idx_1], -1)
		
		self.neg_h = tf.gather_nd(self.neg_hs, neg_idx)
		self.neg_t = tf.gather_nd(self.neg_ts, neg_idx)
		self.neg_r = tf.gather_nd(self.neg_rs, neg_idx)

	def add_embed(self):
		'''

		:return:
		'''
		pass

	def add_loss(self):
		'''

		:return:
		'''
		pass
	
	def add_neg_loss(self):
		'''
		
		:return:
		'''
		
		

	def add_predict(self):
		'''

		:return:
		'''
		pass
		
		

	
