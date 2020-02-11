# coding:utf-8
import codecs
import random

import numpy as np
import tensorflow as tf
import os
import time
import datetime
import ctypes
import json


class DataHelper(object):
    '''
    
    '''
    
    def __init__(self, config):
        '''

        :param config:
        '''
        self.data_path = config.data_path
        
        self.neg_num = config.neg_num
        self.neg_mode = config.neg_mode
        self.read_neg_samples = True
        
        self.mu = 5
        self.theta = 0.5
        self.beta = 0.25
        
    
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
            type_sim = (self.mu + 1) * ((float)(np.sum(self.ent_type[h] & self.ent_type[i]))) / \
                       ((float)(np.sum(self.ent_type[h] | self.ent_type[i])) + self.mu * (float)(
                           np.sum(self.ent_type[h] & self.ent_type[i])))
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
        if random.uniform(0, 1) < self.tph[h] / (self.hpt[t] + self.tph[h]):
            return self.get_neg_head(h, t, r)
        else:
            return self.get_neg_tail(h, t, r)
    
    def make_neg_samples(self):
        '''

        :return:
        '''
        self.train_neg_triples = []
        
        for h, t, r in self.train_pos_triples:
            # s = time.time()
            self.train_neg_triples.append(self.get_neg_sample(h, t, r))
        # print(time.time() - s)
        
        for h, t, r in self.test_pos_triples:
            self.test_neg_triples.append(self.get_neg_sample(h, t, r))
        
        for h, t, r in self.valid_pos_triples:
            self.valid_neg_triples.append(self.get_neg_sample(h, t, r))
        
        self.save_neg_samples()
    
    def save_neg_samples(self):
        '''

        :return:
        '''
        wf = codecs.open(os.path.join(self.data_path, 'neg_train2id.txt'), 'w', encoding='utf8')
        for (h, t, r) in self.train_neg_triples:
            wf.write("{}\t{}\t{}\n".format(h, t, r))
        wf.close()
        
        wf = codecs.open(os.path.join(self.data_path, 'neg_test2id.txt'), 'w', encoding='utf8')
        for (h, t, r) in self.test_neg_triples:
            wf.write("{}\t{}\t{}\n".format(h, t, r))
        wf.close()
        
        wf = codecs.open(os.path.join(self.data_path, 'neg_valid2id.txt'), 'w', encoding='utf8')
        for (h, t, r) in self.valid_neg_triples:
            wf.write("{}\t{}\t{}\n".format(h, t, r))
        wf.close()
    
