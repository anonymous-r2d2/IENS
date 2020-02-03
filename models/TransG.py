#coding:utf-8
import numpy as np
import tensorflow as tf
from .Model import Model


class TransG(Model):
	r'''
	TransG
	'''
	def _calc(self, h, t, r, w):
		config = self.get_config()
		h = tf.nn.l2_normalize(h, -1)
		t = tf.nn.l2_normalize(t, -1)
		r = tf.nn.l2_normalize(r, -1)
		w = tf.nn.l2_normalize(w, -1)
		#h t [batch_size, T, ent_size]
		#r_trans [cluster_num, batch_size, T, rel_size]     w_trans [cluster_num. batch_size, T]
		r_trans = tf.transpose(r, perm=[2, 0, 1, 3])
		w_trans = tf.transpose(w, perm=[2, 0, 1])
		#
		_, self.v_h = tf.nn.moments(tf.pow(tf.linalg.norm(tf.reshape(h, [-1, config.ent_size]), ord=2, axis=1), 2), axes=0)
		_, self.v_t = tf.nn.moments(tf.pow(tf.linalg.norm(tf.reshape(t, [-1, config.ent_size]), ord=2, axis=1), 2), axes=0)

		loss = tf.reduce_sum(-tf.math.log(tf.clip_by_value(tf.reduce_sum(tf.transpose(tf.multiply(w_trans, tf.exp(tf.pow(-tf.linalg.norm(tf.abs(r_trans + h - t), ord=2, axis=3), 2))), [1, 2, 0]), axis=2), 1e-8, tf.float32.max)), axis=1, keep_dims=True)
		return loss

	def _calc_pre(self, h, t, r, w):
		h = tf.nn.l2_normalize(h, -1)
		t = tf.nn.l2_normalize(t, -1)
		r = tf.nn.l2_normalize(r, -1)
		w = tf.nn.l2_normalize(w, -1)
		r_trans = tf.transpose(r, perm=[1, 0, 2])
		w_trans = tf.transpose(w, perm=[0, 1])
		loss = -tf.math.log(tf.clip_by_value(tf.reduce_sum(
			tf.transpose(tf.multiply(w_trans, tf.exp(-tf.linalg.norm(tf.abs(r_trans + h - t), ord=2, axis=2))),
						 [1, 0]), axis=1, keep_dims=False),1e-8,tf.float32.max))
		return loss



	def embedding_def(self):
		#Obtaining the initial configuration of the model
		config = self.get_config()
		#
		self.cluster_num = 4
		#Defining required parameters of the model, including embeddings of entities and relations, entity transfer vectors, and relation transfer vectors

		self.ent_embeddings = tf.get_variable(name = "ent_embeddings", shape = [config.entTotal, config.ent_size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
		self.rel_embeddings = tf.get_variable(name = "rel_embeddings", shape = [config.relTotal, self.cluster_num, config.rel_size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
		self.rel_weights = tf.get_variable(name = "rel_weights", shape = [config.relTotal, self.cluster_num], initializer = tf.constant_initializer(value=1/self.cluster_num, dtype=tf.float32))

		self.parameter_lists = {"ent_embeddings":self.ent_embeddings, \
								"rel_embeddings":self.rel_embeddings, \
								"rel_weights":self.rel_weights}

	def loss_def(self):
		#Obtaining the initial configuration of the model
		config = self.get_config()
		#To get positive triples and negative triples for training
		#The shapes of pos_h, pos_t are (batch_size, 1, ent_size)
		#The shapes of pos_r is (batch_size, 1, rel_size)
		#The shapes of neg_h, neg_t (batch_size, negative_ent + negative_rel, cluster_num, ent_size)
		#The shapes of neg_r is (batch_size, negative_ent + negative_rel, cluster_num, rel_size)
		pos_h, pos_t, pos_r = self.get_positive_instance(in_batch = True)
		neg_h, neg_t, neg_r = self.get_negative_instance(in_batch = True)
		#Embedding entities and relations of triples, e.g. pos_h_e, pos_t_e and pos_r_e are embeddings for positive triples
		pos_h_e = tf.nn.embedding_lookup(self.ent_embeddings, pos_h)
		pos_t_e = tf.nn.embedding_lookup(self.ent_embeddings, pos_t)
		pos_r_e = tf.nn.embedding_lookup(self.rel_embeddings, pos_r)
		pos_r_w = tf.nn.embedding_lookup(self.rel_weights, pos_r)
		neg_h_e = tf.nn.embedding_lookup(self.ent_embeddings, neg_h)
		neg_t_e = tf.nn.embedding_lookup(self.ent_embeddings, neg_t)
		neg_r_e = tf.nn.embedding_lookup(self.rel_embeddings, neg_r)
		neg_r_w = tf.nn.embedding_lookup(self.rel_weights, neg_r)

		#The shape of p_score is (batch_size, 1)
		#The shape of n_score is (batch_size, 1)
		p_score = self._calc(pos_h_e, pos_t_e, pos_r_e, pos_r_w)
		n_score = self._calc(neg_h_e, neg_t_e, neg_r_e, neg_r_w)
		#Calculating loss to get what the framework will optimize
		self.loss = tf.reduce_sum(tf.maximum(p_score - n_score + config.margin, 0))

	def predict_def(self):
		predict_h, predict_t, predict_r = self.get_predict_instance()
		predict_h_e = tf.nn.embedding_lookup(self.ent_embeddings, predict_h)
		predict_t_e = tf.nn.embedding_lookup(self.ent_embeddings, predict_t)
		predict_r_e = tf.nn.embedding_lookup(self.rel_embeddings, predict_r)
		predict_w_e = tf.nn.embedding_lookup(self.rel_weights, predict_r)
		self.predict = self._calc_pre(predict_h_e, predict_t_e, predict_r_e, predict_w_e)
