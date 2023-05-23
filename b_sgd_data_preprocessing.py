"""
A script for the data preprocessing
"""

import numpy as np
import time
import sys
import os
import argparse
from numpy.random import normal, uniform
from numpy.linalg import norm
import itertools
import pandas as pd
from matplotlib import pyplot as plt
import math
import datetime
from IPython import display
from scipy.optimize import minimize
from logreg_functions_non_convex import *
from sklearn.datasets import load_svmlight_file
from scipy import sparse
from scipy import linalg
from numpy.random import RandomState
from tqdm import tqdm

myrepr = lambda x: repr(round(x, 8)).replace('.',',') if isinstance(x, float) else repr(x) #for some methods we used diffrent rounding

parser = argparse.ArgumentParser(description='Generate data and provide information about it for workers and parameter server')

parser.add_argument('--dataset', action='store', dest='dataset', type=str, default='mushrooms', help='The name of the dataset')
parser.add_argument('--loss_func', action='store', dest='loss_func', type=str, default="log-reg",
                    help='loss function ')
parser.add_argument('--la', action='store', dest='la', type=float, default=1, help='lambda')
args = parser.parse_args()

dataset = args.dataset
loss_func = args.loss_func
la = args.la

#debug section

#dataset = 'a9a'
#dataset = 'w8a'
#dataset = 'madelon'
#dataset = 'svmguide1'
#dataset = 'ijcnn1'
#dataset = 'splice'

loss_func = 'log-reg'

if loss_func is None:
    raise ValueError("loss_func has to be specified")

def nan_check (lst):
    """
    Check whether has any item of list np.nan elements
    :param lst: list of datafiles (eg. numpy.ndarray)
    :return:
    """
    for i, item in enumerate (lst):
        if np.sum(np.isnan(item)) > 0:
            raise ValueError("nan files in item {0}".format(i))

def sort_dataset_by_label(X, y):
    sort_index = np.argsort(y)
    X_sorted = X[sort_index].copy()
    y_sorted = y[sort_index].copy()
    return X_sorted, y_sorted

currentDT = datetime.datetime.now()
print (currentDT.strftime("%Y-%m-%d %H:%M:%S"))

data_name = dataset + ".txt"
user_dir = os.path.expanduser('~/')
RAW_DATA_PATH = os.getcwd() +'/data/'

project_path = os.getcwd() + "/"

data_path = project_path + "data_{0}/".format(dataset)

if not os.path.exists(data_path):
    os.mkdir(data_path)

enc_labels = np.nan
data_dense = np.nan
    
#if not (os.path.isfile(data_path + 'X.npy') and os.path.isfile(data_path + 'y.npy')):
if os.path.isfile(RAW_DATA_PATH + data_name):
    data, labels = load_svmlight_file(RAW_DATA_PATH + data_name)
    enc_labels = labels.copy()
    data_dense = data.todense()
    if not np.array_equal(np.unique(labels), np.array([-1, 1], dtype='float')):
        min_label = min(np.unique(enc_labels))
        max_label = max(np.unique(enc_labels))
        enc_labels[enc_labels == min_label] = -1
        enc_labels[enc_labels == max_label] = 1
    #print (enc_labels.shape, enc_labels[-5:])
else:
    raise ValueError("cannot load " + data_name)

assert (type(data_dense) == np.matrix or type(data_dense) == np.ndarray)
assert (type(enc_labels) == np.ndarray)

if np.sum(np.isnan(enc_labels)) > 0:
    raise ValueError("nan values of labels")

if np.sum(np.isnan(data_dense)) > 0:
    raise ValueError("nan values in data matrix")

print (f"{dataset}'s shape: {data_dense.shape}", )

X_0 = np.float64(data_dense)
y_0 = enc_labels
assert len(X_0.shape) == 2
assert len(y_0.shape) == 1
data_len = enc_labels.shape[0]
nan_check([X_0,y_0])
np.save(data_path + 'X', X_0)
np.save(data_path + 'y', y_0)

#partition of data for each worker
n_0, d_0 = X_0.shape
any_vector = np.zeros(d_0)
hess_f_non_reg = logreg_hess_non_reg( any_vector, X_0, y_0)

L_0 = np.float64(linalg.eigh(a=hess_f_non_reg + la*regularizer_hess(any_vector), eigvals_only=True, turbo=True, type=1, eigvals=(d_0-1, d_0-1))[0])
L = np.zeros(n_0, dtype=np.float64)

for i in tqdm(range(n_0)):
    L[i] = linalg.eigh(a= np.outer(X_0[i], X_0[i])/4 + la*regularizer_hess(any_vector), eigvals_only=True, turbo=True, type=1, eigvals=(d_0-1, d_0-1))[0]

np.save(data_path + f'la_{la}', la)
np.save(data_path + f'L_0_{myrepr(la)}', L_0)
np.save(data_path + f'L_{myrepr(la)}', L)

x_0 = np.zeros(d_0, dtype=np.float64)

if not os.path.isfile(data_path + 'w_init_{0}_{1}.npy'.format(loss_func, dataset)):
    x_0 = np.random.normal(loc=0.0, scale=1.0, size=d_0)
    np.save(data_path + 'w_init_{0}_{1}.npy'.format(loss_func, dataset), np.float64(x_0))
else:
    x_0 = np.array(np.load(data_path + 'w_init_{0}_{1}.npy'.format(loss_func, dataset)), dtype=np.float64)

print (f"la = {la}; L_0={L_0}; L_max={np.max(L)}")



