import tensorflow as tf
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

N_EPOCHS = 1000
lr = 0.001
DATASET = 'movielens_1m'

C = 16
F = 16
N_MATS = 200
SEQ_LEN = 8
SAMPLE_SIZE = 128
NEGATIVE_SAMPLE = 1
BATCH_SIZE = 3
DECAY = 0.001
DTYPE = tf.float32
INT_DTYPE = tf.int8
DIR = '../data/' + DATASET + '/'
TRAIN_DIR = '../data/' + DATASET + '/train_adjacency_matrices_perday/'
NUM_NEGATIVE_TEST = 1000
l = 6
