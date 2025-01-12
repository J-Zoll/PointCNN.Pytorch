import argparse 
import math
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

import provider
import math
import random
import time

import torch
from torch import nn
from torch.autograd import Variable

from classifier import Classifier


random.seed(0)
dtype = torch.cuda.FloatTensor



# Load Hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_cls',
                    help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=2, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
parser.add_argument('--verbose', type=bool, default=False, help="Output batch loss [default: False]")
FLAGS = parser.parse_args()

NUM_POINT = FLAGS.num_point
LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
       
MAX_NUM_POINT = 2048

DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

VERBOSE = FLAGS.verbose

LEARNING_RATE_MIN = 0.00001
        
NUM_CLASS = 40
BATCH_SIZE = FLAGS.batch_size #32
NUM_EPOCHS = FLAGS.max_epoch
jitter = 0.01
jitter_val = 0.01

rotation_range = [0, math.pi / 18, 0, 'g']
rotation_rage_val = [0, 0, 0, 'u']
order = 'rxyz'

scaling_range = [0.05, 0.05, 0.05, 'g']
scaling_range_val = [0, 0, 0, 'u']


# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

model = Classifier().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9)
loss_fn = nn.CrossEntropyLoss()

global_step = 1

TRAIN_FILES = provider.getDataFiles(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
TEST_FILES = provider.getDataFiles(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))

losses = []
accuracies = []

for epoch in range(1, NUM_EPOCHS+1):
    train_file_idxs = np.arange(0, len(TRAIN_FILES))
    np.random.shuffle(train_file_idxs)

    for fn in range(len(TRAIN_FILES)):
        #log_string('----' + str(fn) + '-----')
        current_data, current_label = provider.loadDataFile(TRAIN_FILES[train_file_idxs[fn]])
        current_data = current_data[:, 0:NUM_POINT, :]

        current_data, current_label, _ = provider.shuffle_data(current_data, np.squeeze(current_label))
        current_label = np.squeeze(current_label)

        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE

        total_correct = 0
        total_seen = 0
        loss_sum = 0

        if epoch > 1:
            LEARNING_RATE *= DECAY_RATE ** (global_step // DECAY_STEP)
            if LEARNING_RATE > LEARNING_RATE_MIN:
                print("NEW LEARNING RATE:", LEARNING_RATE)
                optimizer = torch.optim.SGD(model.parameters(), lr = LEARNING_RATE, momentum = 0.9)

        for batch_idx in tqdm(range(num_batches)):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx + 1) * BATCH_SIZE

            # Lable
            label = current_label[start_idx:end_idx]
            label = torch.from_numpy(label).long()
            label = Variable(label, requires_grad=False).to(device)
            # Augment batched point clouds by rotation and jittering
            rotated_data = provider.rotate_point_cloud(current_data[start_idx:end_idx, :, :])
            jittered_data = provider.jitter_point_cloud(rotated_data) # P_Sampled
            P_sampled = jittered_data
            F_sampled = np.zeros((BATCH_SIZE, NUM_POINT, 0))
            optimizer.zero_grad()

            t0 = time.time()
            P_sampled = torch.from_numpy(P_sampled).float()
            P_sampled = Variable(P_sampled, requires_grad=False).to(device)

            #F_sampled = torch.from_numpy(F_sampled)

            out = model((P_sampled, P_sampled))
            loss = loss_fn(out, label)
            loss.backward()
            optimizer.step()
            if VERBOSE:
                print("epoch: "+str(epoch) + "   loss: "+str(loss.item()))
            if global_step % 25 == 0:
                loss_v = loss.item()
                print("Loss:", loss_v)
            else:
                loss_v = 0
            global_step += 1