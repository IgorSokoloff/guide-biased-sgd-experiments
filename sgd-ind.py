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
from scipy import sparse
from numpy.random import RandomState
from logreg_functions_non_convex import *

import warnings
warnings.filterwarnings("ignore")

myrepr = lambda x: repr(round(x, 8)).replace('.',',') if isinstance(x, float) else repr(x)
sqnorm = lambda x: norm(x, ord=2) ** 2

def stopping_criterion(sq_norm, eps, epoch, N_epochs, comm, N_comms):
    return (epoch <= N_epochs) and (comm <= N_comms) and (sq_norm >=eps)

def get_scaling(sgd_probs, inds, setting, m_0):
    scaling = np.zeros(m_0)
    if setting =="unbiased":
        scaling[inds] = 1/(m_0*sgd_probs[inds])
        return scaling
    elif setting =="biased":
        computed_batch = inds.shape[0]
        scaling[inds] = 1/(computed_batch)
        return scaling
    else:
        raise ValueError("setting should be biased or unbiased ")

def sgd_ind_estimator(X, x, y, la, rs_sgd, sgd_probs, setting):
    m_0 = X.shape[0]
    arr = np.arange(m_0)
    bernoulli_samples = rs_sgd.binomial(1, sgd_probs)
    inds = arr[bernoulli_samples == 1]
    #BREAK POINT; check that that size is variable
    scaling = get_scaling(sgd_probs, inds, setting, m_0)
    sgrads = logreg_grads(x, X[inds], y[inds], la)
    res = np.multiply(sgrads, scaling[inds, np.newaxis])
    grad_estimate = np.sum(res, axis=0)
    return grad_estimate, np.sum(bernoulli_samples)

def run_algorithm(x_0, X_0, y_0, la, sgd_probs, stepsize, eps, experiment_name, project_path, dataset, N_epochs, N_comms, setting):
    currentDT = datetime.datetime.now()
    print(currentDT.strftime("%Y-%m-%d %H:%M:%S"))
    print(experiment_name + f" has started")
    print ("step_size: ", step_size)
    rs_sgd = RandomState(123)
    m_0 = X_0.shape[0]
    expected_batch_size = int(np.sum(sgd_probs))
    #COMPUTE_GD_EVERY = int(m_0 / expected_batch_size) # todo: check
    COMPUTE_GD_EVERY = 10
    comms_ar = [0]
    epochs_ar = [0]
    it = 0
    cum_computed_batch = 0
    grad = logreg_grad(x_0, X_0, y_0, la)
    sq_norm_ar = [np.linalg.norm(x=grad, ord=2) ** 2]
    x = x_0.copy()
    PRINT_EVERY = 1000
    #TODO:

    while stopping_criterion(sq_norm_ar[-1], eps, epochs_ar[-1], N_epochs, comms_ar[-1], N_comms):
        g, computed_batch = sgd_ind_estimator(X_0, x, y_0, la, rs_sgd, sgd_probs, setting)
        cum_computed_batch += computed_batch
        x = x - stepsize*g
        it += 1

        if it%COMPUTE_GD_EVERY ==0:
            grad = logreg_grad(x, X_0, y_0, la)
            sq_norm_ar.append(sqnorm(grad))
            comms_ar.append(it)
            epochs_ar.append(cum_computed_batch/m_0)

        if it%PRINT_EVERY ==0:
            display.clear_output(wait=True)
            print_last_point_metrics(epochs_ar, comms_ar, sq_norm_ar)

            save_data(epochs_ar, comms_ar, sq_norm_ar, x.copy(), stepsize, experiment_name, project_path, dataset) #TODO

    save_data(epochs_ar, comms_ar, sq_norm_ar, x.copy(), stepsize, experiment_name, project_path, dataset)
    print(experiment_name + f" finished")
    print("End-point:")
    print_last_point_metrics(epochs_ar, comms_ar, sq_norm_ar)


def save_data(epochs_ar, comms_ar, sq_norm_ar, x_solution, stepsize, experiment_name, project_path, dataset):
    print("data saving")
    experiment = experiment_name
    logs_path = project_path + "logs/logs_{0}_{1}/".format(dataset, experiment)

    if not os.path.exists(project_path + "logs/"):
        os.makedirs(project_path + "logs/")

    if not os.path.exists(logs_path):
        os.makedirs(logs_path)

    np.save(logs_path + 'epochs' + '_' +  experiment, np.array(epochs_ar, dtype=np.float64))
    np.save(logs_path + 'comms' + '_' +    experiment, np.array(comms_ar, dtype=np.float64))
    np.save(logs_path + 'solution' + '_' +          experiment, x_solution)
    np.save(logs_path + 'stepsize' + '_' +          experiment, stepsize)
    np.save(logs_path + 'grad_norms' + '_' + experiment, np.array(sq_norm_ar, dtype=np.float64) )

def print_last_point_metrics(epochs_ar, comms_ar, sq_norm_ar):
    print(f"comms: {comms_ar[-1]}; epochs:{epochs_ar[-1]} sq_norm_ar: {sq_norm_ar[-1]};")

parser = argparse.ArgumentParser(description='Run SGD algorithm')
parser.add_argument('--factor', action='store', dest='factor', type=float, default=1.0, help='Stepsize factor')
parser.add_argument('--tol', action='store', dest='tol', type=float, default=1e-7, help='tolerance')
#parser.add_argument('--dataset', action='store', dest='dataset', type=str, default='splice', help='Dataset name for saving logs')
parser.add_argument('--dataset', action='store', dest='dataset', type=str, default='a9a', help='Dataset name for saving logs')
#parser.add_argument('--dataset', action='store', dest='dataset', type=str, default='w8a', help='Dataset name for saving logs')

parser.add_argument('--prb', action='store', dest='prb', type=float, default='0.1', help='Probability of entry being participated')
#parser.add_argument('--prb_type', action='store', dest='prb_type', type=str, default='uniform', help='uniform, importance, custom')
parser.add_argument('--prb_type', action='store', dest='prb_type', type=str, default='importance', help='uniform, importance, custom')
parser.add_argument('--setting_type', action='store', dest='setting_type', type=str, default="biased", help='unbiased ot biased')
parser.add_argument('--max_epochs', action='store', dest='max_epochs', type=float, default=100, help='Maximum number of epochs')
parser.add_argument('--max_comms', action='store', dest='max_comms', type=int, default=100, help='Maximum number of comms')
parser.add_argument('--la', action='store', dest='la', type=float, default=0.1, help='lambda')
parser.add_argument('--batch_prop', action='store', dest='batch_prop', type=float, default=0.1, help='batch proportion')
parser.add_argument('--importance_normed_probs', action='store', dest='importance_normed_probs', type=int, default=0, help='normed or not')
args = parser.parse_args()

eps = args.tol
dataset = args.dataset
factor = args.factor
prb_type = args.prb_type
prb = args.prb
setting = args.setting_type
max_comms = args.max_comms
max_epochs = args.max_epochs
la = args.la
batch_prop = args.batch_prop
importance_normed_probs = args.importance_normed_probs
loss_func = "log-reg"

'''
#dataset = "mushrooms"
dataset = "w8a"
dataset = "splice"
#dataset = "a9a"
loss_func = "log-reg"
factor = 1.0
eps = 1e-2
prb_type = 'uniform'
prb = 0.01
#setting = "unbiased"
setting = "biased"
max_comms = 100
max_epochs = 1e+10
la = 0.1
'''

assert setting in ["unbiased", "biased"]
assert prb_type in ["uniform", "importance", "custom"]

user_dir = os.path.expanduser('~/')
project_path = os.getcwd() + "/"
#print(project_path)
data_path = project_path + "data_{0}/".format(dataset)

X_0 = np.load(data_path + 'X.npy') #whole dateset
y_0 = np.load(data_path + 'y.npy')
assert (X_0.dtype == 'float64')
assert (len(X_0.shape) == 2)

#la = np.load(data_path + 'la.npy')
L_0 = np.float64(np.load(data_path + f'L_0_{myrepr(la)}.npy'))
L = np.load(data_path + f'L_{myrepr(la)}.npy')
x_0 = np.array(np.load(data_path + 'w_init_{0}_{1}.npy'.format(loss_func, dataset)), dtype=np.float64)

m_0,dim = X_0.shape

#rs_probs = RandomState(123)

if prb_type == "uniform":
    if importance_normed_probs:
        print (importance_normed_probs)
        batch_size = batch_prop * m_0
        imp_probs = batch_size*np.minimum((L / np.sum(L)), 1)
        prb = np.sum(imp_probs)/m_0
        sgd_probs = np.full(m_0, prb)
    else:
        sgd_probs = np.full(m_0, prb)
elif prb_type == "importance":
    batch_size = batch_prop*m_0
    sgd_probs = np.minimum(batch_size*(L/np.sum(L)), 1)
elif prb_type == "custom":
    ValueError("Not implemented yet")
else:
    ValueError("wrong prb_type")

if setting == "unbiased":
    A = np.max ((1-sgd_probs)*L/(sgd_probs*m_0))
    B = 1
    stepsize_base = min(1/(L_0*B), 1/np.sqrt(L_0*A*max_comms))
elif setting == "biased":
    if importance_normed_probs:
        batch_size = batch_prop * m_0
        imp_probs = batch_size * np.minimum((L / np.sum(L)), 1)
        prb = np.sum(imp_probs) / m_0
        A = np.max(L) / prb
    p_min = np.min(sgd_probs)
    A = np.max(L)/p_min
    stepsize_base = 1/np.sqrt((L_0*A*max_comms))
else:
    raise ValueError("setting should be biased or unbiased ")

step_size = np.float64(stepsize_base*factor)

experiment_name = {"uniform":"sgd-ind_{0}_{1}_{2}_{4}_{3}x".format(setting, prb_type, myrepr(prb), myrepr(factor), myrepr(la)),
                "importance":"sgd-ind_{0}_{1}_{2}_{4}_{3}x".format(setting, prb_type, myrepr(batch_prop), myrepr(factor), myrepr(la))
                   }[prb_type]

print ('------------------------------------------------------------------------------------------------')
print(experiment_name)
print (f"la={np.round(la,2)}; L={np.round(L_0,2)}; L_max={np.round(np.max(L),2)}; A={np.round(A,2)}; stepsize_base={np.format_float_scientific(stepsize_base, unique=False, precision=2)};")
start_time = time.time()
run_algorithm(x_0, X_0, y_0, la, sgd_probs, step_size, eps, experiment_name, project_path, dataset, max_epochs, max_comms, setting)
time_diff = time.time() - start_time
print(f"Computation time: {time_diff} sec")
print ('------------------------------------------------------------------------------------------------')


