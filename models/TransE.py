#coding:utf-8
import numpy as np
import tensorflow as tf
from .Model import Model

class TransE(Model):
	def _calc(self, h, t, r):
		h = tf.nn.l2_normalize(h, -1)
		t = tf.nn.l2_normalize(t, -1)
		r = tf.nn.l2_normalize(r, -1)
		return abs(h + r - t)

	def embedding_def(self):
		self.ent_embeddings = tf.get_variable(name = "ent_embeddings", shape = [self.ent_num, self.embed_dim], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
		self.rel_embeddings = tf.get_variable(name = "rel_embeddings", shape = [self.rel_num, self.embed_dim], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
		self.parameter_lists = {"ent_embeddings":self.ent_embeddings, "rel_embeddings":self.rel_embeddings}

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

		_p_score = self._calc(self.p_h, self.p_t, self.p_r)
		_n_score = self._calc(self.n_h, self.n_t, self.n_r)

		p_score =  tf.reduce_sum(_p_score, -1, keep_dims = True)
		n_score =  tf.reduce_sum(_n_score, -1, keep_dims = True)

		self.loss = tf.reduce_sum(tf.maximum(p_score - n_score + self.margin, 0))
		
	def add_neg_loss(self):
	
		_ns_score = self._calc(self.n_hs, self.n_ts, self.n_rs)
		ns_score = tf.nn.softmax(tf.reduce_sum(_ns_score, -1), axis = -1)
		
		self.neg_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(self.neg_sim - ns_score), axis = -1))
		

	def add_predict(self):
		# print(self.p_h.shape)
		# print(self._calc(self.p_h, self.p_t, self.p_r).shape)
		# print(tf.reduce_sum(self._calc(self.p_h, self.p_t, self.p_r), -1, keep_dims = False).shape)
		self.predict = tf.reduce_sum(self._calc(self.p_h, self.p_t, self.p_r), -1, keep_dims = False)