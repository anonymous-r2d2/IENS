import config
import models
import tensorflow as tf
import numpy as np

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
flags.DEFINE_string('data_path', '../benchmarks/FB15K/', 'path of data')
flags.DEFINE_string('save_path', '../res/kg_1M/transg', 'path of save model and data')

# hyperparameter
flags.DEFINE_integer('threads', 8, 'work threads')
flags.DEFINE_integer('epochs', 500, 'train epochs')
flags.DEFINE_integer('batch_size', 1024, 'batch size')
flags.DEFINE_integer('embed_dim', 300, 'embedding dimension')
flags.DEFINE_string('opt', 'SGD', 'optimition method')

def main(_):
    cuda_list = FLAGS.gpu
    data_path = FLAGS.data_path
    save_path = FLAGS.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # set cuda
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_list


    # Input training files from benchmarks/FB15K/ folder.
    con = config.Config()
    # True: Input test files from the same folder.
    con.set_in_path(data_path)
    con.set_work_threads(FLAGS.threads)
    con.set_train_times(FLAGS.epochs)
    con.set_batch_size(FLAGS.batch_size)
    con.set_alpha(0.001)
    con.set_margin(1.0)
    con.set_bern(0)
    con.set_dimension(FLAGS.embed_dim)
    con.set_ent_neg_rate(1)
    con.set_rel_neg_rate(0)
    con.set_opt_method(FLAGS.opt)
    con.init()
    con.set_model(models.TransE)
    con.run()
    parameters = con.get_parameters("numpy")



    conR = config.Config()
    conR.set_in_path(data_path)
    conR.set_test_link_prediction(True)
    conR.set_test_triple_classification(True)
    conR.set_work_threads(FLAGS.threads)
    conR.set_train_times(FLAGS.epochs)
    conR.set_batch_size(FLAGS.batch_size)
    conR.set_alpha(0.001)
    conR.set_bern(0)
    conR.set_dimension(FLAGS.embed_dim)
    conR.set_margin(1)
    conR.set_ent_neg_rate(1)
    conR.set_rel_neg_rate(0)
    conR.set_opt_method(FLAGS.opt)

    # Models will be exported via tf.Saver() automatically.
    con.set_export_files(os.path.join(save_path, "model.vec.tf"), 0)
    # Model parameters will be exported to json files automatically.
    con.set_out_files(os.path.join(save_path, "embedding.vec.json"))
    # Initialize experimental settings.
    conR.init()
    # Load pretrained TransE results.
    conR.set_model(models.TransR)
    parameters["transfer_matrix"] = np.array(
        [(np.identity(FLAGS.embed_dim).reshape((FLAGS.embed_dim * FLAGS.embed_dim))) for i in range(conR.get_rel_total())])
    conR.set_parameters(parameters)
    # Train the model.
    conR.run()
    # To test models after training needs "set_test_flag(True)".
    conR.test()
    con.predict_head_entity(18932, 25, 5)
    con.predict_tail_entity(18930, 25, 5)
    con.predict_relation(18930, 18932, 5)
    con.predict_triple(18930, 18932, 25)
    con.predict_triple(18930, 18932, 40)

if __name__ == '__main__':
    app.run()


# #Train TransR based on pretrained TransE results.
# #++++++++++++++TransE++++++++++++++++++++
#
# con = config.Config()
# con.set_in_path("../benchmarks/FB15K/")
# con.set_work_threads(4)
# con.set_train_times(500)
# con.set_nbatches(100)
# con.set_alpha(0.001)
# con.set_bern(0)
# con.set_dimension(100)
# con.set_margin(1)
# con.set_ent_neg_rate(1)
# con.set_rel_neg_rate(0)
# con.set_opt_method("SGD")
# con.init()
# con.set_model(models.TransE)
# con.run()
# parameters = con.get_parameters("numpy")
#
# #++++++++++++++TransR++++++++++++++++++++
#
# conR = config.Config()
# #Input training files from benchmarks/FB15K/ folder.
# conR.set_in_path("../benchmarks/FB15K/")
# #True: Input test files from the same folder.
# conR.set_test_link_prediction(True)
# conR.set_test_triple_classification(True)
#
# conR.set_work_threads(4)
# conR.set_train_times(500)
# conR.set_nbatches(100)
# conR.set_alpha(0.001)
# conR.set_bern(0)
# conR.set_dimension(100)
# conR.set_margin(1)
# conR.set_ent_neg_rate(1)
# conR.set_rel_neg_rate(0)
# conR.set_opt_method("SGD")
#
# #Models will be exported via tf.Saver() automatically.
# conR.set_export_files("../res/FB15K/transR/model.vec.tf", 0)
# #Model parameters will be exported to json files automatically.
# conR.set_out_files("../res/FB15K/transR/embedding.vec.json")
# #Initialize experimental settings.
# conR.init()
# #Load pretrained TransE results.
# conR.set_model(models.TransR)
# parameters["transfer_matrix"] = np.array([(np.identity(100).reshape((100*100))) for i in range(conR.get_rel_total())])
# conR.set_parameters(parameters)
# #Train the model.
# conR.run()
# #To test models after training needs "set_test_flag(True)".
# conR.test()

