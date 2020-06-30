#coding:utf-8
import numpy as np
import tensorflow as tf
from .Model import Model

class ComplEx(Model):

	def embedding_def(self):
		self.ent1_embeddings = tf.get_variable(name = "ent1_embeddings", shape = [self.ent_num, self.embed_dim], initializer = tf.contrib.layers.xavier_initializer(uniform = True))
		self.rel1_embeddings = tf.get_variable(name = "rel1_embeddings", shape = [self.rel_num, self.embed_dim], initializer = tf.contrib.layers.xavier_initializer(uniform = True))
		self.ent2_embeddings = tf.get_variable(name = "ent2_embeddings", shape = [self.ent_num, self.embed_dim], initializer = tf.contrib.layers.xavier_initializer(uniform = True))
		self.rel2_embeddings = tf.get_variable(name = "rel2_embeddings", shape = [self.rel_num, self.embed_dim], initializer = tf.contrib.layers.xavier_initializer(uniform = True))
		self.parameter_lists = {"ent_re_embeddings":self.ent1_embeddings, \
								"ent_im_embeddings":self.ent2_embeddings, \
								"rel_re_embeddings":self.rel1_embeddings, \
								"rel_im_embeddings":self.rel2_embeddings}
	r'''
	ComplEx extends DistMult by introducing complex-valued embeddings so as to better model asymmetric relations. 
	It is proved that HolE is subsumed by ComplEx as a special case.
	'''
	def _calc(self, e1_h, e2_h, e1_t, e2_t, r1, r2):
		return tf.reduce_sum(e1_h * e1_t * r1 + e2_h * e2_t * r1 + e1_h * e2_t * r2 - e2_h * e1_t * r2, -1, keep_dims = False)

	def add_embed(self):
		self.p1_h = tf.nn.embedding_lookup(self.ent1_embeddings, self.pos_h)
		self.p2_h = tf.nn.embedding_lookup(self.ent2_embeddings, self.pos_h)
		self.p1_t = tf.nn.embedding_lookup(self.ent1_embeddings, self.pos_t)
		self.p2_t = tf.nn.embedding_lookup(self.ent2_embeddings, self.pos_t)
		self.p1_r = tf.nn.embedding_lookup(self.rel1_embeddings, self.pos_r)
		self.p2_r = tf.nn.embedding_lookup(self.rel2_embeddings, self.pos_r)
		
		self.n1_h = tf.nn.embedding_lookup(self.ent1_embeddings, self.neg_h)
		self.n2_h = tf.nn.embedding_lookup(self.ent2_embeddings, self.neg_h)
		self.n1_t = tf.nn.embedding_lookup(self.ent1_embeddings, self.neg_t)
		self.n2_t = tf.nn.embedding_lookup(self.ent2_embeddings, self.neg_t)
		self.n1_r = tf.nn.embedding_lookup(self.rel1_embeddings, self.neg_r)
		self.n2_r = tf.nn.embedding_lookup(self.rel2_embeddings, self.neg_r)
		
		self.n1_hs = tf.nn.embedding_lookup(self.ent1_embeddings, self.neg_hs)
		self.n2_hs = tf.nn.embedding_lookup(self.ent2_embeddings, self.neg_hs)
		self.n1_ts = tf.nn.embedding_lookup(self.ent1_embeddings, self.neg_ts)
		self.n2_ts = tf.nn.embedding_lookup(self.ent2_embeddings, self.neg_ts)
		self.n1_rs = tf.nn.embedding_lookup(self.rel1_embeddings, self.neg_rs)
		self.n2_rs = tf.nn.embedding_lookup(self.rel2_embeddings, self.neg_rs)

	def add_loss(self):
		_p_score = self._calc(self.p1_h, self.p2_h, self.p1_t, self.p2_t, self.p1_r, self.p2_r)
		_n_score = self._calc(self.n1_h, self.n2_h, self.n1_t, self.n2_t, self.n1_r, self.n2_r)
		loss_func = tf.reduce_sum(tf.nn.softplus(- 1 * _p_score) + tf.nn.softplus(1 * _n_score))
		regul_func = tf.reduce_sum(self.p1_h ** 2 + self.p1_t ** 2 + self.p1_r ** 2 + self.n1_h ** 2 + self.n1_t ** 2 + self.n1_r ** 2 + self.p2_h ** 2 + self.p2_t ** 2 + self.p2_r ** 2 + self.n2_h ** 2 + self.n2_t ** 2 + self.n2_r ** 2)
		self.loss =  loss_func + self.lmbda * regul_func
		
	
	
	def add_neg_loss(self):
		_ns_score = self._calc(self.n1_hs, self.n2_hs, self.n1_ts, self.n2_ts, self.n1_rs, self.n2_rs)
		ns_score = tf.nn.softmax(tf.reduce_sum(_ns_score, -1), axis=-1)
		
		neg_sim = tf.nn.softmax(self.neg_sim, axis=-1)
		self.neg_loss = tf.reduce_mean(tf.reduce_sum(neg_sim * ns_score, axis=-1))

	def predict_def(self):
		self.predict = -self._calc(self.p1_h, self.p2_h, self.p1_t, self.p2_t, self.p1_r, self.p2_r)

