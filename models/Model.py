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
		# self.type_num = config.type_num
	
		self.embed_dim = config.embed_dim
		self.margin = config.margin
	
		with tf.name_scope("input"):
			self.add_input()

		with tf.name_scope("embedding"):
			self.embedding_def()
			self.add_embed()

		with tf.name_scope("loss"):
			self.add_loss()

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

		self.neg_h = tf.placeholder(tf.int32, [None])
		self.neg_t = tf.placeholder(tf.int32, [None])
		self.neg_r = tf.placeholder(tf.int32, [None])

		# self.ent_type = tf.placeholder(tf.int32, [None, self.type_num])
		# self.co_rel = tf.placeholder(tf.int32, [None, self.rel_num])

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

	def add_predict(self):
		'''

		:return:
		'''
		pass

	
