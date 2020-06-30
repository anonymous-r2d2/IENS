#coding:utf-8
import numpy as np
import tensorflow as tf
from .Model import Model

def tf_resize(tensor, axis, size):
	shape = tensor.get_shape().as_list()
	osize = shape[axis]
	if osize == size:
		return tensor
	if (osize > size):
		shape[axis] = size
		return tf.slice(tensor, begin = (0,) * len(shape), size = shape)
	paddings = [[0, 0] for i in range(len(shape))]
	paddings[axis][1] = size - osize
	return tf.pad(tensor, paddings = paddings)

class TransD(Model):
	r'''
	TransD constructs a dynamic mapping matrix for each entity-relation pair by considering the diversity of entities and relations simultaneously. 
	Compared with TransR/CTransR, TransD has fewer parameters and has no matrix vector multiplication.
	'''
	def _transfer(self, e, t, r):
		# return e + tf.reduce_sum(e * t, -1, keep_dims = True) * r
		return tf_resize(e, -1, r.get_shape()[-1]) + tf.reduce_sum(e * t, -1, keepdims = True) * r

	def _calc(self, h, t, r):
		h = tf.nn.l2_normalize(h, -1)
		t = tf.nn.l2_normalize(t, -1)
		r = tf.nn.l2_normalize(r, -1)
		return abs(h + r - t)

	def embedding_def(self):
		#Obtaining the initial configuration of the model
		#Defining required parameters of the model, including embeddings of entities and relations, entity transfer vectors, and relation transfer vectors
		self.ent_embeddings = tf.get_variable(name = "ent_embeddings", shape = [self.ent_num, self.embed_dim], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
		self.rel_embeddings = tf.get_variable(name = "rel_embeddings", shape = [self.rel_num, self.embed_dim], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
		self.ent_transfer = tf.get_variable(name = "ent_transfer", shape = [self.ent_num, self.embed_dim], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
		self.rel_transfer = tf.get_variable(name = "rel_transfer", shape = [self.rel_num, self.embed_dim], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
		self.parameter_lists = {"ent_embeddings":self.ent_embeddings, \
								"rel_embeddings":self.rel_embeddings, \
								"ent_transfer":self.ent_transfer, \
								"rel_transfer":self.rel_transfer}

	def add_embed(self):
		# Embedding entities and relations of triples, e.g. pos_h_e, pos_t_e and pos_r_e are embeddings for positive triples
		pos_h_e = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_h)
		pos_t_e = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_t)
		pos_r_e = tf.nn.embedding_lookup(self.rel_embeddings, self.pos_r)
		neg_h_e = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_h)
		neg_t_e = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_t)
		neg_r_e = tf.nn.embedding_lookup(self.rel_embeddings, self.neg_r)
		neg_h_es = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_hs)
		neg_t_es = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_ts)
		neg_r_es = tf.nn.embedding_lookup(self.rel_embeddings, self.neg_rs)
		# Getting the required parameters to transfer entity embeddings, e.g. pos_h_t, pos_t_t and pos_r_t are transfer parameters for positive triples
		pos_h_t = tf.nn.embedding_lookup(self.ent_transfer, self.pos_h)
		pos_t_t = tf.nn.embedding_lookup(self.ent_transfer, self.pos_t)
		pos_r_t = tf.nn.embedding_lookup(self.rel_transfer, self.pos_r)
		neg_h_t = tf.nn.embedding_lookup(self.ent_transfer, self.neg_h)
		neg_t_t = tf.nn.embedding_lookup(self.ent_transfer, self.neg_t)
		neg_r_t = tf.nn.embedding_lookup(self.rel_transfer, self.neg_r)
		neg_h_ts = tf.nn.embedding_lookup(self.ent_transfer, self.neg_hs)
		neg_t_ts = tf.nn.embedding_lookup(self.ent_transfer, self.neg_ts)
		neg_r_ts = tf.nn.embedding_lookup(self.rel_transfer, self.neg_rs)
		# Calculating score functions for all positive triples and negative triples
		self.p_h = self._transfer(pos_h_e, pos_h_t, pos_r_t)
		self.p_t = self._transfer(pos_t_e, pos_t_t, pos_r_t)
		self.p_r = pos_r_e
		self.n_h = self._transfer(neg_h_e, neg_h_t, neg_r_t)
		self.n_t = self._transfer(neg_t_e, neg_t_t, neg_r_t)
		self.n_r = neg_r_e
		
		self.n_hs = self._transfer(neg_h_es, neg_h_ts, neg_r_ts)
		self.n_ts = self._transfer(neg_t_es, neg_t_ts, neg_r_ts)
		self.n_rs = neg_r_es

	def add_loss(self):
		#The shape of _p_score is (batch_size, 1, hidden_size)
		#The shape of _n_score is (batch_size, negative_ent + negative_rel, hidden_size)
		_p_score = self._calc(self.p_h, self.p_t, self.p_r)
		_n_score = self._calc(self.n_h, self.n_t, self.n_r)
		#The shape of p_score is (batch_size, 1, 1)
		#The shape of n_score is (batch_size, negative_ent + negative_rel, 1)
		p_score =  tf.reduce_sum(_p_score, -1, keep_dims = True)
		n_score =  tf.reduce_sum(_n_score, -1, keep_dims = True)
		#Calculating loss to get what the framework will optimize
		self.loss = tf.reduce_sum(tf.maximum(p_score - n_score + self.margin, 0))
	
	def add_neg_loss(self):
		_ns_score = self._calc(self.n_hs, self.n_ts, self.n_rs)
		ns_score = tf.nn.softmax(tf.reduce_sum(_ns_score, -1), axis=-1)
		
		neg_sim = tf.nn.softmax(self.neg_sim, axis=-1)
		self.neg_loss = tf.reduce_mean(tf.reduce_sum(neg_sim * ns_score, axis=-1))

	def predict_def(self):
		self.predict = tf.reduce_sum(self._calc(self.p_h, self.p_t, self.p_r), -1, keep_dims = False)
