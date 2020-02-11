import codecs
import random


class NegativeSampling(object):
    '''
    
    '''
    
    def __init__(self):
        '''
        
        '''

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
        _h = 0
        _sim = 0
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
            sim = self.theta * type_sim + (1 - self.theta) * head_rel_sim
            tmp_neg_samples.append((i, t, r, type_sim, head_rel_sim))
            if sim > _sim:
                _sim = sim
                _h = i
        return (_h, t, r)

    def get_neg_tail(self, h, t, r):
        '''

        :param h:
        :param t:
        :param r:
        :return:
        '''
        _t = 0
        _sim = 0
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
            sim = self.theta * type_sim + (1.0 - self.theta) * tail_rel_sim
            tmp_neg_samples.append((h, i, r, type_sim, tail_rel_sim))
            if sim > _sim:
                _sim = sim
                _t = i
        return (h, _t, r)

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
        self.test_neg_triples = []
        self.valid_neg_triples = []
    
        if self.read_neg_samples:
            rf = codecs.open(os.path.join(self.data_path, 'neg_train2id.txt'), 'r', encoding='utf8')
            for line in rf:
                h, t, r = line.strip().split()
                self.train_neg_triples.append((h, t, r))
            rf.close()
        
            rf = codecs.open(os.path.join(self.data_path, 'neg_test2id.txt'), 'r', encoding='utf8')
            for line in rf:
                h, t, r = line.strip().split()
                self.test_neg_triples.append((h, t, r))
            rf.close()
        
            rf = codecs.open(os.path.join(self.data_path, 'neg_valid2id.txt'), 'r', encoding='utf8')
            for line in rf:
                h, t, r = line.strip().split()
                self.valid_neg_triples.append((h, t, r))
            rf.close()
            return
    
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