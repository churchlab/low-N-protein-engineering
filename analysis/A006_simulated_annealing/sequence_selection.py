import os
import sys
import warnings
import multiprocessing as mp
import random
import copy
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append('../common')
import data_io_utils
import paths
import constants
import utils


def convert_result_vals_to_mat(res):
    res['seq_history'] = np.stack(res['seq_history'], 0)
    res['fitness_history'] = np.stack(res['fitness_history'], 0)
    res['fitness_std_history'] = np.stack(res['fitness_std_history'], 0)
    #res['fitness_mem_pred_history'] = np.stack(res['fitness_mem_pred_history'], 0)
        
    return res

def isolate_highly_functional_trajectories(res, burnin, fitness_threshold):
    fit_mat = res['fitness_history'] # iter x chains
    
    # 1. Select trajectories that reach functional sequences
    # after the specified number of burn-in iterations.
    bi_fit_mat = fit_mat[burnin:]
    
    mask = np.any(bi_fit_mat > fitness_threshold, axis=0)
    
    return mask

def isolate_valley_crossing_trajectories(res, burnin, fitness_threshold, valley_threshold=0.2):
    fit_mat = res['fitness_history'] # iter x chains
    
    bi_fit_mat = fit_mat[burnin:]
    
    functional_mask = np.any(bi_fit_mat > fitness_threshold, axis=0)
    valley_mask = np.any(fit_mat < valley_threshold, axis=0) # dont include burn in
    
    return np.logical_and(functional_mask, valley_mask)
    
    
def build_pwm(seqs):
    ohe = np.stack([utils.encode_aa_seq_as_one_hot(s, flatten=False) for s in seqs], 0)
    pwm = np.mean(ohe, axis=0) + 1e-6
    return pwm
    
def calc_effective_num_residues_per_site(seqs):
    pwm = build_pwm(seqs)
    return np.exp(-np.sum(pwm*np.log(pwm), axis=0))

def get_best_sequence_in_each_trajectory(res, burnin=0, max_sa_itr=None):
    seq_mat = res['seq_history']
    fit_mat = res['fitness_history']
    fit_std_mat = res['fitness_std_history']
    
    if max_sa_itr is None:
        max_sa_itr = fit_mat.shape[0]
    
    best_seq_idx = np.argmax(fit_mat[burnin:max_sa_itr,:], axis=0) + burnin
    
    best_seqs = []
    best_seq_fitness = []
    best_seq_fitness_std = []
    for i in range(seq_mat.shape[1]):
        best_seqs.append(seq_mat[best_seq_idx[i], i])
        best_seq_fitness.append(fit_mat[best_seq_idx[i], i])
        best_seq_fitness_std.append(fit_std_mat[best_seq_idx[i], i])
        
    return np.array(best_seqs), np.array(best_seq_fitness), np.array(best_seq_fitness_std), best_seq_idx

def obtain_top_n_functional_seqs(res, burnin, n=100):
    ufit = -np.sort(-np.unique(res['fitness_history'].reshape(-1)))
    
    idx = 1
    while True:
        int_traj_mask = isolate_highly_functional_trajectories(res, burnin, ufit[idx])
        
        idx += 1
        if np.sum(int_traj_mask) >= n:
            break

    best_seqs, best_seq_fitness, best_seq_fitness_std, best_seq_idx = get_best_sequence_in_each_trajectory(res)
    
    return best_seqs[int_traj_mask], best_seq_fitness[int_traj_mask], best_seq_fitness_std[int_traj_mask], int_traj_mask

