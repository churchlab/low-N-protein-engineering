import os
import sys
import warnings
import multiprocessing as mp
import random
import copy

import numpy as np
import pandas as pd
import tensorflow as tf
from unirep import babbler1900 as babbler

sys.path.append('../common')
import data_io_utils
import paths
import constants
import utils

# WARNING. This will work fine, but Surge is no longer using this path. 
# Surge has collected all the checkpoints at s3://efficient-protein-design/evotuning_checkpoints
#
## These are the set of weight paths Surge is currently using. 
# paths.GFP_ET_GLOBAL_INIT_1_WEIGHT_PATH
# paths.GFP_ET_GLOBAL_INIT_2_WEIGHT_PATH
# paths.GFP_ET_RANDOM_INIT_1_WEIGHT_PATH
EVOTUNED_UNIREP_MODEL_WEIGHT_PATH = "./evotuned/unirep" 


GFP_LIB_REGION = [29, 110] # [inclusive, exclusive] - see A051a mlpe-gfp-pilot
BLAC_LIB_REGION = [132, 213] # [inclusive, exclusive] - see A051a mlpe-gfp-pilot

def dump_numpy_weight_files_from_tf_ckpt(ckpt):
    print('Dumping numpy weight files for:', ckpt)
    tf.reset_default_graph()
    b = babbler(batch_size=10, model_path="./1900_weights") # not the weights I want.
    b.dump_checkpoint_weights(ckpt, os.path.dirname(ckpt)) # dumps weights in checkpoint

    
    
    
### SIMULATED ANNEALING ###

def acceptance_prob(f_proposal, f_current, k, T):
    ap = np.exp((f_proposal - f_current)/(k*T))
    ap[ap > 1] = 1
    return ap

def make_n_random_edits(seq, nedits, alphabet=constants.AA_ALPHABET_STANDARD_ORDER,
        min_pos=None, max_pos=None): ## Test
    """
    min_pos is inclusive. max_pos is exclusive
    """
    
    lseq = list(seq)
    lalphabet = list(alphabet)
    
    if min_pos is None:
        min_pos = 0
    
    if max_pos is None:
        max_pos = len(seq)
    
    # Create non-redundant list of positions to mutate.
    l = list(range(min_pos, max_pos))
    nedits = min(len(l), nedits)
    random.shuffle(l)
    pos_to_mutate = l[:nedits]    
    
    for i in range(nedits):
        pos = pos_to_mutate[i]     
        aa_to_choose_from = list(set(lalphabet) - set([seq[pos]]))
                        
        lseq[pos] = aa_to_choose_from[np.random.randint(len(aa_to_choose_from))]
        
    return "".join(lseq)

def propose_seqs(seqs, mu_muts_per_seq, min_pos=None, max_pos=None):
    
    mseqs = []
    for i,s in enumerate(seqs):
        n_edits = np.random.poisson(mu_muts_per_seq[i]-1) + 1
        mseqs.append(make_n_random_edits(s, n_edits, min_pos=min_pos, max_pos=max_pos)) 
        
    return mseqs


def anneal(
        init_seqs, 
        k, 
        T_max, 
        mu_muts_per_seq,
        get_fitness_fn,
        n_iter=1000, 
        decay_rate=0.99,
        min_mut_pos=None,
        max_mut_pos=None):
    
    print('Initializing')
    state_seqs = copy.deepcopy(init_seqs)
    state_fitness, state_fitness_std, state_fitness_mem_pred = get_fitness_fn(state_seqs)
    
    seq_history = [copy.deepcopy(state_seqs)]
    fitness_history = [copy.deepcopy(state_fitness)]
    fitness_std_history = [copy.deepcopy(state_fitness_std)]
    fitness_mem_pred_history = [copy.deepcopy(state_fitness_mem_pred)]
    for i in range(n_iter):
        print('Iteration:', i)
        
        print('\tProposing sequences.')
        proposal_seqs = propose_seqs(state_seqs, mu_muts_per_seq, 
                min_pos=min_mut_pos, max_pos=max_mut_pos)
        
        print('\tCalculating predicted fitness.')
        proposal_fitness, proposal_fitness_std, proposal_fitness_mem_pred = get_fitness_fn(proposal_seqs)
        
        
        print('\tMaking acceptance/rejection decisions.')
        aprob = acceptance_prob(proposal_fitness, state_fitness, k, T_max*(decay_rate**i))
        
        # Make sequence acceptance/rejection decisions
        for j, ap in enumerate(aprob):
            if np.random.rand() < ap:
                # accept
                state_seqs[j] = copy.deepcopy(proposal_seqs[j])
                state_fitness[j] = copy.deepcopy(proposal_fitness[j])
                state_fitness_std[j] = copy.deepcopy(proposal_fitness_std[j])
                state_fitness_mem_pred[j] = copy.deepcopy(proposal_fitness_mem_pred[j])
            # else do nothing (reject)
            
        seq_history.append(copy.deepcopy(state_seqs))
        fitness_history.append(copy.deepcopy(state_fitness))
        fitness_std_history.append(copy.deepcopy(state_fitness_std))
        fitness_mem_pred_history.append(copy.deepcopy(state_fitness_mem_pred))
        
    return {
        'seq_history': seq_history,
        'fitness_history': fitness_history,
        'fitness_std_history': fitness_std_history,
        'fitness_mem_pred_history': fitness_mem_pred_history,
        'init_seqs': init_seqs,
        'T_max': T_max,
        'mu_muts_per_seq': mu_muts_per_seq,
        'k': k,
        'n_iter': n_iter,
        'decay_rate': decay_rate,
        'min_mut_pos': min_mut_pos,
        'max_mut_pos': max_mut_pos,
    }
