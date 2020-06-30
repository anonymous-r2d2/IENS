#coding:utf-8
import numpy as np
import tensorflow as tf
from .Model import Model

class DistMult(Model):
	r'''
	DistMult is based on the bilinear model where each relation is represented by a diagonal rather than a full matrix. 
	DistMult enjoys the same scalable property as TransE and it achieves superior performance over TransE.
	'''
	def _calc(self, h, t, r):
		return tf.reduce_sum(h * r * t, -1, keep_dims = False)

	def embedding_def(self):
		self.ent_embeddings = tf.get_variable(name = "ent_embeddings", shape = [self.ent_num, self.embed_dim], initializer = tf.contrib.layers.xavier_initializer(uniform = True))
		self.rel_embeddings = tf.get_variable(name = "rel_embeddings", shape = [self.rel_num, self.embed_dim], initializer = tf.contrib.layers.xavier_initializer(uniform = True))
		self.parameter_lists = {"ent_embeddings":self.ent_embeddings, \
								"rel_embeddings":self.rel_embeddings}
		
	def add_embed(self):
		self.p_h = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_h)
		self.p_t = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_t)
		self.p_r = tf.nn.embedding_lookup(self.rel_embeddings, self.pos_r)
		self.n_h = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_h)
		self.n_t = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_t)
		self.n_r = tf.nn.embedding_lookup(self.rel_embeddings, self.neg_r)
		
		self.n_hs = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_hs)
		self.n_ts = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_ts)
		self.n_rs = tf.nn.embedding_lookup(self.rel_embeddings, self.neg_rs)
	
	def add_loss(self):
		
		p_h = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_h)
		p_t = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_t)
		p_r = tf.nn.embedding_lookup(self.rel_embeddings, self.pos_r)
		n_h = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_h)
		n_t = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_t)
		n_r = tf.nn.embedding_lookup(self.rel_embeddings, self.neg_r)
		_p_score = self._calc(p_h, p_t, p_r)
		_n_score = self._calc(n_h, n_t, n_r)
		loss_func = tf.reduce_sum(tf.nn.softplus(- 1 * _p_score) + tf.nn.softplus(1 * _n_score))
		regul_func = tf.reduce_sum(p_h ** 2 + p_t ** 2 + p_r ** 2 + n_h ** 2 + n_t ** 2 + n_r ** 2)
		self.loss =  loss_func + self.lmbda * regul_func
	
	def add_neg_loss(self):
		_ns_score = self._calc(self.n_hs, self.n_ts, self.n_rs)
		ns_score = tf.nn.softmax(tf.reduce_sum(_ns_score, -1), axis=-1)
		
		neg_sim = tf.nn.softmax(self.neg_sim, axis=-1)
		self.neg_loss = tf.reduce_mean(tf.reduce_sum(neg_sim * ns_score, axis=-1))

	def predict_def(self):
		self.predict = tf.reduce_sum(self._calc(self.p_h, self.p_t, self.p_r), -1, keep_dims = False)