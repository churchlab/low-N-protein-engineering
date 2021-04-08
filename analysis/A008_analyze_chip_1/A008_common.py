import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.append('../common')
import data_io_utils
import paths
import constants
import utils

sys.path.append('../A006_simulated_annealing/')
import A006_common
from unirep import babbler1900 as babbler

sys.path.append('../A003_policy_optimization/')
import A003_common


UNIREP_BATCH_SIZE = 3500

def split_seq_id_into_features(sid):

    if isinstance(sid, str) and sid[:4] == 'GFP_':
        splits = sid.split('-')

        if splits[4] == 'SmallTrust':
            rep_hash = splits[5]
            special_case = splits[4]
        else:
            rep_hash = splits[4]
            special_case = ''
        

        res = {
            'model': splits[1],
            'ntrain': int(splits[2]),
            'rep': int(splits[3]),
            'rep_hash': rep_hash,
            'seq_traj_idx': splits[-1],
            'special_case': special_case
        }
        
    elif isinstance(sid, str) and sid[:5] == 'BLAC_':
        splits = sid.split('-')

        if splits[1] == 'LargeMut':
            special_case = splits[1]
            del splits[1]
        else:
            special_case = ''
        

        res = {
            'model': splits[1],
            'ntrain': int(splits[2]),
            'rep': int(splits[3]),
            'rep_hash': splits[4],
            'seq_traj_idx': splits[5],
            'special_case': special_case
        }
        
        

    else: # one of Grigory's

        res = {
            'model': np.nan,
            'ntrain': np.nan,
            'rep': np.nan,
            'rep_hash': np.nan,
            'seq_traj_idx': np.nan,
            'special_case': ''
        }
    
    return res


# Copied from A008a
# Originally From A006_simulated_annealing/hyperborg/GFP_simulated_annealing.py
def load_base_model(model_name):
    print('Setting up base model')
    tf.reset_default_graph()

    if model_name == 'ET_Global_Init_1':
        base_model = babbler(batch_size=UNIREP_BATCH_SIZE, model_path=paths.GFP_ET_GLOBAL_INIT_1_WEIGHT_PATH)
        print('Loading weights from:', paths.GFP_ET_GLOBAL_INIT_1_WEIGHT_PATH)
    elif model_name == 'ET_Global_Init_2':
        base_model = babbler(batch_size=UNIREP_BATCH_SIZE, model_path=paths.GFP_ET_GLOBAL_INIT_2_WEIGHT_PATH)
        print('Loading weights from:', paths.GFP_ET_GLOBAL_INIT_2_WEIGHT_PATH)
    elif model_name == 'ET_Random_Init_1':
        base_model = babbler(batch_size=UNIREP_BATCH_SIZE, model_path=paths.GFP_ET_RANDOM_INIT_1_WEIGHT_PATH)
        print('Loading weights from:', paths.GFP_ET_RANDOM_INIT_1_WEIGHT_PATH)
    elif model_name =='OneHot':
        # Just need it to generate one-hot reps.
        # Top model created within OneHotRegressionModel doesn't actually get used.
        base_model = models.OneHotRegressionModel('EnsembledRidge') 
    else:
        assert False, 'Unsupported base model'
        
    return base_model
        
# Generate representations
def generate_batch_reps(seq_list, sess, base_model):        
    if 'babbler1900' == base_model.__class__.__name__:
        assert len(seq_list) <= UNIREP_BATCH_SIZE
        hiddens = base_model.get_all_hiddens(seq_list, sess)
        rep = np.stack([np.mean(s, axis=0) for s in hiddens],0)

    elif 'OneHotRegressionModel' == base_model.__class__.__name__:
        rep = base_model.encode_seqs(seq_list)

    return rep

# From A003/019
def generate_reps(seqs, sess, base_model):
    batch_size = UNIREP_BATCH_SIZE
    
    avg_hiddens = []
    for i in range(0, len(seqs), batch_size):
        min_idx = i
        max_idx = min(i+batch_size,len(seqs))
        
        print(min_idx, max_idx)
        avg_hiddens.append(
            generate_batch_reps(seqs[min_idx:max_idx], sess, base_model))
        
    return np.vstack(avg_hiddens)