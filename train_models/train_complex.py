import sys
sys.path.append("..")
import config.Config
import models
import tensorflow as tf
import numpy as np
import os
import codecs

from tensorflow import app
from tensorflow import flags

FLAGS = flags.FLAGS
flags.DEFINE_string('gpu', '7', 'gpu will be used')
flags.DEFINE_string('data_path', '../benchmarks/kg_video/', 'path of data')
flags.DEFINE_string('save_path', '../res/paper/kg_video/transe', 'path of save model and data')

# hyperparameter
flags.DEFINE_integer('embed_dim', 100, 'embedding dimension')
flags.DEFINE_integer('batch_size', 4096, 'batch size')
flags.DEFINE_float('lr', 0.0005, 'learning rate')
flags.DEFINE_integer('neg_num', 1, 'negative num')
flags.DEFINE_string('neg_mode', 'dynamic', 'negatvie sampling mode')
flags.DEFINE_float('lmbda', 0.01, 'lmbda')
flags.DEFINE_float('margin', 1.0, 'margin')
flags.DEFINE_string('opt', 'Adam', 'optimition method')
flags.DEFINE_integer('epochs', 1000, 'train epochs')

flags.DEFINE_integer('ent_num', 0, 'entity num')
flags.DEFINE_integer('rel_num', 0, 'relation num')
flags.DEFINE_integer('type_num', 0, 'type num')


def main(_):
    cuda_list = FLAGS.gpu
    data_path = FLAGS.data_path
    save_path = FLAGS.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # set cuda
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_list


    # Input training files from benchmarks/FB15K/ folder.
    con = config.Config(FLAGS)
    # True: Input test files from the same folder.



    # Set the knowledge embedding model
    con.init(models.ComplEx)
    # Train the model.
    con.train()
    con.save_parameters()
    # To test models after training needs "set_test_flag(True)".
    # con.test()



if __name__ == '__main__':
    app.run()