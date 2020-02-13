#coding:utf-8
import codecs
import random

import numpy as np
import tensorflow as tf
import os
import time
import datetime
import ctypes
import json

class Config(object):
	'''
	
	'''
	def __init__(self, config):
		'''
		
		:param config:
		'''
		self.data_path = config.data_path
		
		self.embed_dim = config.embed_dim
		
		self.batch_size = config.batch_size
		self.lr = config.lr
		self.neg_num = config.neg_num
		self.alpha = config.lr
		self.lmbda = config.lmbda
		self.opt = config.opt
		self.train_epochs = config.epochs
		self.margin = config.margin
		
		self.neg_mode = config.neg_mode
		self.read_neg_samples = True
		
		self.mu = 5
		self.theta = 0.5
		self.beta = 0.25
		
		self.optimizer = None
	
	def init(self, model):
		'''
		
		:return:
		'''
		self.make_data()
		self.init_model(model)

	def read_dict(self):
		'''
		
		:return:
		'''
		rf = codecs.open(os.path.join(self.data_path, 'entity2id.txt'), 'r', encoding='utf8')
		self.ent_num = int(rf.readline().strip())
		rf.close()
		
		rf = codecs.open(os.path.join(self.data_path, 'relation2id.txt'), 'r', encoding='utf8')
		self.rel_num = int(rf.readline().strip())
		rf.close()
		
		rf = codecs.open(os.path.join(self.data_path, 'type2id.txt'), 'r', encoding='utf8')
		self.type_num = int(rf.readline().strip())
		rf.close()
		
	
	def read_data(self):
		'''

		:return:
		'''
		self.train_pos_triples = []
		self.test_pos_triples = []
		self.valid_pos_triples = []

		self.tph = np.zeros([self.ent_num], dtype=np.int32)
		self.hpt = np.zeros([self.ent_num], dtype=np.int32)

		self.head_freq = np.zeros([self.ent_num], dtype=np.int32)
		self.tail_freq = np.zeros([self.ent_num], dtype=np.int32)

		self.ent_type = np.zeros([self.ent_num, self.type_num], dtype=np.int32)
		self.tail_rel = np.zeros([self.ent_num, self.rel_num], dtype=np.int32)
		self.head_rel = np.zeros([self.ent_num, self.rel_num], dtype=np.int32)

		rf = codecs.open(os.path.join(self.data_path, 'ent_type.txt'), 'r', encoding='utf8')
		rf.readline().strip()
		for line in rf:
			ent, _types = line.strip().split('\t')
			_types = map(int, _types.split(' '))
			for type_item in _types:
				self.ent_type[int(ent)][type_item] = 1
		rf.close()

		rf = codecs.open(os.path.join(self.data_path, 'train2id.txt'), 'r', encoding='utf8')
		self.train_num = int(rf.readline().strip())
		for line in rf:
			h, t, r = map(int, line.strip().split())
			self.tail_rel[t][r] = 1
			self.head_rel[h][r] = 1
			self.head_freq[h] += 1
			self.tail_freq[t] += 1
			self.tph[h] += 1
			self.hpt[t] += 1
			self.train_pos_triples.append((h, t, r))
		rf.close()

		rf = codecs.open(os.path.join(self.data_path, 'test2id.txt'), 'r', encoding='utf8')
		self.test_num = int(rf.readline().strip())
		for line in rf:
			h, t, r = map(int, line.strip().split())
			self.tail_rel[t][r] = 1
			self.head_rel[h][r] = 1
			self.head_freq[h] += 1
			self.tail_freq[t] += 1
			self.tph[h] += 1
			self.hpt[t] += 1
			self.test_pos_triples.append((h, t, r))
		rf.close()


		rf = codecs.open(os.path.join(self.data_path, 'valid2id.txt'), 'r', encoding='utf8')
		self.valid_num = int(rf.readline().strip())
		for line in rf:
			h, t, r = map(int, line.strip().split())
			self.tail_rel[t][r] = 1
			self.head_rel[h][r] = 1
			self.head_freq[h] += 1
			self.tail_freq[t] += 1
			self.tph[h] += 1
			self.hpt[t] += 1
			self.valid_pos_triples.append((h, t, r))
		rf.close()

		# sort

	def init_type_constrain(self):
		'''
		
		:return:
		'''
		self.head_lef = [0 for _ in range(self.rel_num)]
		self.head_rig = [0 for _ in range(self.rel_num)]
		self.tail_lef = [0 for _ in range(self.rel_num)]
		self.tail_rig = [0 for _ in range(self.rel_num)]
		
		self.head_type = []
		self.tail_type = []
		
		tot_lef = 0
		tot_rig = 0
		
		rf = codecs.open(os.path.join(self.data_path, 'type_constrain.txt'), 'r', encoding='utf8')
		rel_num = int(rf.readline().strip())
		for i in range(rel_num):
			line = rf.readline()
			content = list(map(int, line.strip().split()))
			rel, tot, head_types = content[0], content[1], content[2:]
			self.head_lef[rel] = tot_lef
			self.head_type += sorted(head_types)
			tot_lef + len(head_types)
			self.head_rig[rel] = tot_lef
			
			line = rf.readline()
			content = list(map(int, line.strip().split()))
			rel, tot, tail_types = content[0], content[1], content[2:]
			self.tail_lef[rel] = tot_rig
			self.tail_type += sorted(tail_types)
			tot_rig + len(tail_types)
			self.tail_rig[rel] = tot_rig
		
		

	def init_sample(self):
		'''

		:return:
		'''
		return
		
		
	def get_neg_head(self, h, t, r):
		'''

		:param h:
		:param t:
		:param r:
		:return:
		'''
		sampled_ent = random.sample(range(self.ent_num), self.neg_num)
		tmp_neg_samples = []
		
		for i in sampled_ent:
			indicate = 1.0 if self.head_rel[i][r] == 1 else 0.0
			type_sim = (self.mu + 1)*((float)(np.sum(self.ent_type[h] & self.ent_type[i]))) / \
					   ((float)(np.sum(self.ent_type[h] | self.ent_type[i])) + self.mu*(float)(np.sum(self.ent_type[h] & self.ent_type[i])))
			head_rel_sim = (float)(np.sum(self.head_rel[h] & self.head_rel[i])) / \
						   (float)(np.sum(self.head_rel[h] | self.head_rel[i])) * \
						   (self.theta + indicate) / (self.theta + 1.0)
			tmp_neg_samples.append((i, t, r, type_sim, head_rel_sim))

		return tmp_neg_samples

	def get_neg_tail(self, h, t, r):
		'''

		:param h:
		:param t:
		:param r:
		:return:
		'''
		sampled_ent = random.sample(range(self.ent_num), self.neg_num)
		tmp_neg_samples = []
		
		for i in sampled_ent:
			indicate = 1.0 if self.tail_rel[i][r] == 1 else 0.0
			type_sim = (self.mu + 1) * ((float)(np.sum(self.ent_type[t] & self.ent_type[i]))) / \
					   ((float)(np.sum(self.ent_type[t] | self.ent_type[i])) + self.mu * (float)(
						   np.sum(self.ent_type[t] & self.ent_type[i])))
			# print(h, t, r, np.sum(self.tail_rel[t] | self.tail_rel[i]), np.sum(self.tail_rel[t] & self.tail_rel[i]))
			tail_rel_sim = (float)(np.sum(self.tail_rel[t] & self.tail_rel[i])) / \
						   (float)(np.sum(self.tail_rel[t] | self.tail_rel[i])) * \
						   (self.theta + indicate) / (self.theta + 1.0)
			tmp_neg_samples.append((h, i, r, type_sim, tail_rel_sim))
			
		return tmp_neg_samples

	def get_neg_sample(self, h, t, r):
		'''

		:param h:
		:param t:
		:param r:
		:return:
		'''
		if random.uniform(0, 1) < self.tph[h]/(self.hpt[t] + self.tph[h]):
			return self.get_neg_head(h, t, r)
		else:
			return self.get_neg_tail(h, t, r)
	
	def make_neg_samples(self):
		'''
		
		:return:
		'''
		self.train_nhs = []
		self.train_nts = []
		self.train_nrs = []
		self.train_type_sims = []
		self.train_rel_sims = []
		
		for h, t, r in self.train_pos_triples:
			neg_sample = self.get_neg_sample(h, t, r)
			hs, ts, rs, type_sims, rel_sims = [list(item) for item in zip(*neg_sample)]
			self.train_nhs.append(np.array(hs))
			self.train_nts.append(np.array(ts))
			self.train_nrs.append(np.array(rs))
			self.train_type_sims.append(np.array(type_sims))
			self.train_rel_sims.append(np.array(rel_sims))
			

		
	def make_data(self):
		'''
		
		:return:
		'''
		print('read dict')
		self.read_dict()
		print('read data')
		self.read_data()
		print('init type constrain')
		self.init_type_constrain()
		print('init sample')
		self.init_sample()
		print('make neg samples')
		self.make_neg_samples()
		
		self.train_ph, self.train_pt, self.train_pr = zip(*self.train_pos_triples)
		self.test_ph, self.test_pt, self.test_pr = zip(*self.test_pos_triples)
		self.valid_ph, self.valid_pt, self.valid_pr = zip(*self.valid_pos_triples)
		

	def find(self, h, t, r):
		'''
		
		:param h:
		:param t:
		:param r:
		:return:
		'''
		return (h, t, r) in self.train_pos_triples or (h, t, r) in self.test_pos_triples or (h, t, r) in self.valid_pos_triples
			
	def init_model(self, model):
		self.model = model
		self.graph = tf.Graph()
		with self.graph.as_default():
			config = tf.ConfigProto()
			config.gpu_options.allow_growth = True
			self.sess = tf.Session(config=config)
			with self.sess.as_default():
				initializer = tf.contrib.layers.xavier_initializer(uniform=True)
				with tf.variable_scope("model", reuse=None, initializer=initializer):
					self.trainModel = self.model(config=self)
					if self.optimizer != None:
						pass
					elif self.opt == "Adagrad" or self.opt == "adagrad":
						self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.alpha,
																   initial_accumulator_value=1e-20)
					elif self.opt == "Adadelta" or self.opt == "adadelta":
						self.optimizer = tf.train.AdadeltaOptimizer(self.alpha)
					elif self.opt == "Adam" or self.opt == "adam":
						self.optimizer = tf.train.AdamOptimizer(self.alpha)
					else:
						self.optimizer = tf.train.GradientDescentOptimizer(self.alpha)
						
					embed_var_list = [var for var in tf.trainable_variables() if 'embed' in var.name]
					neg_var_list = [var for var in tf.trainable_variables() if 'neg' in var.name]
					
					self.embed_train_op = self.optimizer.minimize(self.trainModel.loss, var_list=embed_var_list)
					if self.neg_mode == 'dynamic':
						self.neg_train_op = self.optimizer.minimize(self.trainModel.neg_loss, var_list=neg_var_list)
					# grads_and_vars = self.optimizer.compute_gradients(self.trainModel.loss)
					# self.train_op = self.optimizer.apply_gradients(grads_and_vars)
					
				self.saver = tf.train.Saver()
				self.sess.run(tf.global_variables_initializer())
	
	def get_parameters_by_name(self, var_name):
		'''
		
		:param var_name:
		:return:
		'''
		with self.graph.as_default():
			with self.sess.as_default():
				if var_name in self.trainModel.parameter_lists:
					return self.sess.run(self.trainModel.parameter_lists[var_name])
				else:
					return None
	
	def save_parameters(self):
		'''
		
		:return:
		'''
		print('save parameters:')
		for var_name in self.trainModel.parameter_lists:
			print('\tsave {}'.format(var_name))
			wf = codecs.open(os.path.join(self.data_path, '{}.txt'.format(var_name)), 'w', encoding='utf8')
			for i, embed in enumerate(self.get_parameters_by_name(var_name)):
				wf.write("{}\t".format(i))
				for var in embed:
					wf.write("{} ".format(var))
				wf.write("\n")
				# wf.write("{}\t{}\n".format(i, ' '.join(list(embed))))
			wf.close()
		
	def set_parameters_by_name(self, var_name, tensor):
		'''
		
		:param var_name:
		:param tensor:
		:return:
		'''
		with self.graph.as_default():
			with self.sess.as_default():
				if var_name in self.trainModel.parameter_lists:
					self.trainModel.parameter_lists[var_name].assign(tensor).eval()
		
	def train(self):
		'''
		
		:return:
		'''
		print('train')
		with self.graph.as_default():
			with self.sess.as_default():
				for step in range(self.train_epochs):
					_embed_loss = .0
					_neg_loss = .0
					for i in range(self.train_num//self.batch_size):
						batch_ph = self.train_ph[i*self.batch_size: (i+1)*self.batch_size]
						batch_pt = self.train_pt[i * self.batch_size: (i + 1) * self.batch_size]
						batch_pr = self.train_pr[i * self.batch_size: (i + 1) * self.batch_size]
						batch_nhs = self.train_nhs[i * self.batch_size: (i + 1) * self.batch_size]
						batch_nts = self.train_nts[i * self.batch_size: (i + 1) * self.batch_size]
						batch_nrs = self.train_nrs[i * self.batch_size: (i + 1) * self.batch_size]
						batch_type_sims = self.train_type_sims[i * self.batch_size: (i + 1) * self.batch_size]
						batch_rel_sims = self.train_rel_sims[i * self.batch_size: (i + 1) * self.batch_size]
						
						
						feed_dict = {self.trainModel.pos_h: batch_ph, self.trainModel.pos_t: batch_pt, self.trainModel.pos_r: batch_pr,
									 self.trainModel.neg_hs: batch_nhs, self.trainModel.neg_ts: batch_nts, self.trainModel.neg_rs: batch_nrs,
									 self.trainModel.neg_type_sims: batch_type_sims, self.trainModel.neg_rel_sims: batch_rel_sims}
						if self.neg_mode == 'dynamic':
							if step%2 == 0:
								_, batch_embed_loss = self.sess.run([self.embed_train_op, self.trainModel.loss], feed_dict=feed_dict)
								_embed_loss += batch_embed_loss
							else:
								_, batch_neg_loss = self.sess.run([self.neg_train_op, self.trainModel.neg_loss], feed_dict=feed_dict)
								_neg_loss += batch_neg_loss
						else:
							_, batch_embed_loss = self.sess.run([self.embed_train_op, self.trainModel.loss],
																feed_dict=feed_dict)
							_embed_loss += batch_embed_loss
					print("step: {:04d}, embed_loss: {:.4f}, neg_loss: {:.4f}".format(step, _embed_loss, _neg_loss))
					
					
	def test(self):
		self.h1_raw_tot, self.h3_raw_tot, self.h10_raw_tot, self.h1_filter_tot, self.h3_filter_tot, self.h10_filter_tot = 0, 0, 0, 0, 0, 0
		self.h1_raw_tot_constrain, self.h3_raw_tot_constrain, self.h10_raw_tot_constrain, self.h1_filter_tot_constrain, self.h3_filter_tot_constrain, self.h10_filter_tot_constrain = 0, 0, 0, 0, 0, 0
		self.h_raw_rank, self.h_filter_rank, self.h_raw_reci_rank, self.h_filter_reci_rank = .0, .0, .0, .0
		self.h_raw_rank_constrain, self.h_filter_rank_constrain, self.h_raw_reci_rank_constrain, self.h_filter_reci_rank_constrain = .0, .0, .0, .0
		
		self.t1_raw_tot, self.t3_raw_tot, self.t10_raw_tot, self.t1_filter_tot, self.t3_filter_tot, self.t10_filter_tot = 0, 0, 0, 0, 0, 0
		self.t1_raw_tot_constrain, self.t3_raw_tot_constrain, self.t10_raw_tot_constrain, self.t1_filter_tot_constrain, self.t3_filter_tot_constrain, self.t10_filter_tot_constrain = 0, 0, 0, 0, 0, 0
		self.t_raw_rank, self.t_filter_rank, self.t_raw_reci_rank, self.t_filter_reci_rank = .0, .0, .0, .0
		self.t_raw_rank_constrain, self.t_filter_rank_constrain, self.t_raw_reci_rank_constrain, self.t_filter_reci_rank_constrain = .0, .0, .0, .0
		
		with self.graph.as_default():
			with self.sess.as_default():
				for (h, t, r) in self.test_pos_triples:
					batch_h = [i for i in range(self.ent_num)]
					batch_t = [t for _ in range(self.ent_num)]
					batch_r = [r for _ in range(self.ent_num)]
					
					feed_dict = {self.trainModel.pos_h: batch_h, self.trainModel.pos_t: batch_t, self.trainModel.pos_r: batch_r}
					
					pred = self.sess.run(self.trainModel.predict, feed_dict=feed_dict)
					
					self.test_head(h, r, t, pred)
					
					batch_h = [h for _ in range(self.ent_num)]
					batch_t = [i for i in range(self.ent_num)]
					batch_r = [r for _ in range(self.ent_num)]
					
					feed_dict = {self.trainModel.pos_h: batch_h, self.trainModel.pos_t: batch_t,
								 self.trainModel.pos_r: batch_r}
					
					pred = self.sess.run(self.trainModel.predict, feed_dict=feed_dict)
					self.test_tail(h, r, t, pred)
					
		self.test_link_pred()
							
	def test_link_pred(self):
		'''
		
		:return:
		'''
		self.h1_raw_tot /= self.test_num
		self.h3_raw_tot /= self.test_num
		self.h10_raw_tot /= self.test_num
		self.h1_filter_tot /= self.test_num
		self.h3_filter_tot /= self.test_num
		self.h10_filter_tot /= self.test_num
		
		self.h_raw_rank /= self.test_num
		self.h_filter_rank /= self.test_num
		self.h_raw_reci_rank /= self.test_num
		self.h_filter_reci_rank /= self.test_num
		
		self.t1_raw_tot /= self.test_num
		self.t3_raw_tot /= self.test_num
		self.t10_raw_tot /= self.test_num
		self.t1_filter_tot /= self.test_num
		self.t3_filter_tot /= self.test_num
		self.t10_filter_tot /= self.test_num
		
		self.t_raw_rank /= self.test_num
		self.t_filter_rank /= self.test_num
		self.t_raw_reci_rank /= self.test_num
		self.t_filter_reci_rank /= self.test_num
		
		self.h1_raw_tot_constrain /= self.test_num
		self.h3_raw_tot_constrain /= self.test_num
		self.h10_raw_tot_constrain /= self.test_num
		self.h1_filter_tot_constrain /= self.test_num
		self.h3_filter_tot_constrain /= self.test_num
		self.h10_filter_tot_constrain /= self.test_num
		
		self.h_raw_rank_constrain /= self.test_num
		self.h_filter_rank_constrain /= self.test_num
		self.h_raw_reci_rank_constrain /= self.test_num
		self.h_filter_reci_rank_constrain /= self.test_num
		
		self.t1_raw_tot_constrain /= self.test_num
		self.t3_raw_tot_constrain /= self.test_num
		self.t10_raw_tot_constrain /= self.test_num
		self.t1_filter_tot_constrain /= self.test_num
		self.t3_filter_tot_constrain /= self.test_num
		self.t10_filter_tot_constrain /= self.test_num
		
		self.t_raw_rank_constrain /= self.test_num
		self.t_filter_rank_constrain /= self.test_num
		self.t_raw_reci_rank_constrain /= self.test_num
		self.t_filter_reci_rank_constrain /= self.test_num
		
		
		print('no type onstraint results: ')
		print('metric:\t\t\t MRR \t\t MR \t\t hit@10 \t hit@3  \t hit@1 ')
		print('l(raw):\t\t\t {} \t {} \t {} \t {} \t {} '.format(self.h_raw_reci_rank, self.h_raw_rank,
																   self.h10_raw_tot, self.h3_raw_tot, self.h1_raw_tot))
		print('r(raw):\t\t\t {} \t {} \t {} \t {} \t {} '.format(self.t_raw_reci_rank, self.t_raw_rank,
																   self.t10_raw_tot, self.t3_raw_tot, self.t1_raw_tot))
		print('averaged(raw):\t\t {} \t {} \t {} \t {} \t {} '.format((self.h_raw_reci_rank + self.t_raw_reci_rank)/2,
																		(self.h_raw_rank + self.t_raw_rank)/2,
																		(self.h10_raw_tot + self.t10_raw_tot)/2,
																		(self.h3_raw_tot + self.t3_raw_tot)/2,
																		(self.h1_raw_tot + self.t1_raw_tot)/2))
		print()
		print('l(filter):\t\t {} \t {} \t {} \t {} \t {} '.format(self.h_filter_reci_rank, self.h_filter_rank,
																  self.h10_filter_tot, self.h3_filter_tot,
																  self.h1_filter_tot))
		print('r(filter):\t\t {} \t {} \t {} \t {} \t {} '.format(self.t_filter_reci_rank, self.t_filter_rank,
																  self.t10_filter_tot, self.t3_filter_tot,
																  self.t1_filter_tot))
		print('averaged(filter):\t {} \t {} \t {} \t {} \t {} '.format(
																(self.h_filter_reci_rank + self.t_filter_reci_rank) / 2,
																(self.h_filter_rank + self.t_filter_rank) / 2,
																(self.h10_filter_tot + self.t10_filter_tot) / 2,
																(self.h3_filter_tot + self.t3_filter_tot) / 2,
																(self.h1_filter_tot + self.t1_filter_tot) / 2))
		
		print('type onstraint results: ')
		print('metric:\t\t\t MRR \t\t MR \t\t hit@10 \t hit@3  \t hit@1 ')
		print('l(raw):\t\t\t {} \t {} \t {} \t {} \t {} '.format(self.h_raw_reci_rank_constrain, self.h_raw_rank_constrain,
																 self.h10_raw_tot_constrain, self.h3_raw_tot_constrain, self.h1_raw_tot_constrain))
		print('r(raw):\t\t\t {} \t {} \t {} \t {} \t {} '.format(self.t_raw_reci_rank_constrain, self.t_raw_rank_constrain,
																 self.t10_raw_tot_constrain, self.t3_raw_tot_constrain, self.t1_raw_tot_constrain))
		print('averaged(raw):\t\t {} \t {} \t {} \t {} \t {} '.format((self.h_raw_reci_rank_constrain + self.t_raw_reci_rank_constrain) / 2,
																	  (self.h_raw_rank_constrain + self.t_raw_rank_constrain) / 2,
																	  (self.h10_raw_tot_constrain + self.t10_raw_tot_constrain) / 2,
																	  (self.h3_raw_tot_constrain + self.t3_raw_tot_constrain) / 2,
																	  (self.h1_raw_tot_constrain + self.t1_raw_tot_constrain) / 2))
		print()
		print('l(filter):\t\t {} \t {} \t {} \t {} \t {} '.format(self.h_filter_reci_rank_constrain, self.h_filter_rank_constrain,
																  self.h10_filter_tot_constrain, self.h3_filter_tot_constrain,
																  self.h1_filter_tot_constrain))
		print('r(filter):\t\t {} \t {} \t {} \t {} \t {} '.format(self.t_filter_reci_rank_constrain, self.t_filter_rank_constrain,
																  self.t10_filter_tot_constrain, self.t3_filter_tot_constrain,
																  self.t1_filter_tot_constrain))
		print('averaged(filter):\t {} \t {} \t {} \t {} \t {} '.format(
			(self.h_filter_reci_rank_constrain + self.t_filter_reci_rank_constrain) / 2,
			(self.h_filter_rank_constrain + self.t_filter_rank_constrain) / 2,
			(self.h10_filter_tot_constrain + self.t10_filter_tot_constrain) / 2,
			(self.h3_filter_tot_constrain + self.t3_filter_tot_constrain) / 2,
			(self.h1_filter_tot_constrain + self.t1_filter_tot_constrain) / 2))
		
		
	def test_head(self, h, r, t, pred):
		'''
		
		:param h:
		:param r:
		:param t:
		:param pred:
		:return:
		'''
		print("test_head:", h, r, t, len(pred))
		lef, rig = self.head_lef[r], self.head_rig[r]
		
		h_raw_rank = 0
		h_filter_rank = 0
		
		h_raw_rank_constrain = 0
		h_filter_rank_constrain = 0
		
		_score = pred[h]
		
		for i, score in enumerate(pred):
			if score < _score:
				h_raw_rank += 1
				if not self.find(i, t, r):
					h_filter_rank += 1
			while lef < rig and self.head_type[lef] < i:
				lef += 1
			if lef < rig and i == self.head_type[lef]:
				if score < _score:
					h_raw_rank_constrain += 1
					if not self.find(i, t, r):
						h_filter_rank_constrain += 1
						
		if h_raw_rank < 1:
			self.h1_raw_tot += 1
		if h_raw_rank < 3:
			self.h3_raw_tot += 1
		if h_raw_rank < 10:
			self.h10_raw_tot += 1
		
		if h_filter_rank < 1:
			self.h1_filter_tot += 1
		if h_filter_rank < 3:
			self.h3_filter_tot += 1
		if h_filter_rank < 10:
			self.h10_filter_tot += 1
		
		if h_raw_rank_constrain < 1:
			self.h1_raw_tot_constrain += 1
		if h_raw_rank_constrain < 3:
			self.h3_raw_tot_constrain += 1
		if h_raw_rank_constrain < 10:
			self.h10_raw_tot_constrain += 1
		
		if h_filter_rank_constrain < 1:
			self.h1_filter_tot_constrain += 1
		if h_filter_rank_constrain < 3:
			self.h3_filter_tot_constrain += 1
		if h_filter_rank_constrain < 10:
			self.h10_filter_tot_constrain += 1
			
		self.h_raw_rank += h_raw_rank + 1
		self.h_filter_rank += h_filter_rank + 1
		self.h_raw_reci_rank += 1.0/(h_raw_rank + 1)
		self.h_filter_reci_rank += 1.0/(h_filter_rank + 1)
		
		self.h_raw_rank_constrain += h_raw_rank_constrain + 1
		self.h_filter_rank_constrain += h_filter_rank_constrain + 1
		self.h_raw_reci_rank_constrain += 1.0 / (h_raw_rank_constrain + 1)
		self.h_filter_reci_rank_constrain += 1.0 / (h_filter_rank_constrain + 1)
	
	def test_tail(self, h, r, t, pred):
		'''

		:param h:
		:param r:
		:param t:
		:param pred:
		:return:
		'''
		lef, rig = self.tail_lef[r], self.tail_rig[r]
		
		t_raw_rank = 0
		t_filter_rank = 0
		
		t_raw_rank_constrain = 0
		t_filter_rank_constrain = 0
		
		_score = pred[t]
		
		for i, score in enumerate(pred):
			if score < _score:
				t_raw_rank += 1
				if not self.find(h, i, r):
					t_filter_rank += 1
			while lef < rig and self.tail_type[lef] < i:
				lef += 1
			if lef < rig and i == self.tail_type[lef]:
				if score < _score:
					t_raw_rank_constrain += 1
					if not self.find(h, i, r):
						t_filter_rank_constrain += 1
		
		if t_raw_rank < 1:
			self.t1_raw_tot += 1
		if t_raw_rank < 3:
			self.t3_raw_tot += 1
		if t_raw_rank < 10:
			self.t10_raw_tot += 1
		
		if t_filter_rank < 1:
			self.t1_filter_tot += 1
		if t_filter_rank < 3:
			self.t3_filter_tot += 1
		if t_filter_rank < 10:
			self.t10_filter_tot += 1
		
		if t_raw_rank_constrain < 1:
			self.t1_raw_tot_constrain += 1
		if t_raw_rank_constrain < 3:
			self.t3_raw_tot_constrain += 1
		if t_raw_rank_constrain < 10:
			self.t10_raw_tot_constrain += 1
		
		if t_filter_rank_constrain < 1:
			self.t1_filter_tot_constrain += 1
		if t_filter_rank_constrain < 3:
			self.t3_filter_tot_constrain += 1
		if t_filter_rank_constrain < 10:
			self.t10_filter_tot_constrain += 1
		
		self.t_raw_rank += t_raw_rank + 1
		self.t_filter_rank += t_filter_rank + 1
		self.t_raw_reci_rank += 1.0 / (t_raw_rank + 1)
		self.t_filter_reci_rank += 1.0 / (t_filter_rank + 1)
		
		self.t_raw_rank_constrain += t_raw_rank_constrain + 1
		self.t_filter_rank_constrain += t_filter_rank_constrain + 1
		self.t_raw_reci_rank_constrain += 1.0 / (t_raw_rank_constrain + 1)
		self.t_filter_reci_rank_constrain += 1.0 / (t_filter_rank_constrain + 1)
		
		
			
		
		
