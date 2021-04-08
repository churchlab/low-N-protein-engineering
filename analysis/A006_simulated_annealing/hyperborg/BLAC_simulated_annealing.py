import os
import sys
import warnings
import random
import copy
import pickle
import subprocess

import numpy as np
import pandas as pd
import scipy
import tensorflow as tf


sys.path.append('../../common')
import data_io_utils
import paths
import constants
import utils

sys.path.append('../../A003_policy_optimization/')
import models
import A003_common

sys.path.append('..')
import A006_common
from unirep import babbler1900 as babbler

class BetaLacOneHotEncoder(object):
    
    def __init__(self):
        pass
        
    def encode_seqs(self, seqs):
        return utils.encode_aa_seq_list_as_matrix_of_flattened_one_hots(seqs)

### CONFIGURATION ###
# Load config
config_file = str(sys.argv[1])
print('Config file:', config_file)

with open(config_file, 'rb') as f:
    config = pickle.load(f)
    
seed = config['seed']
n_train_seqs = config['n_train_seqs']
model_name = config['model'] 
n_chains = config['n_chains']
T_max = config['T_max']
sa_n_iter = config['sa_n_iter']
temp_decay_rate = config['temp_decay_rate']
min_mut_pos = config['min_mut_pos']
max_mut_pos = config['max_mut_pos']
nmut_threshold = config['nmut_threshold']
output_file = config['output_file']

for k in config:
    if k == 'init_seqs':
        print(k + ':', config[k][:5])
    else:
        print(k + ':', config[k])

## do few iterations to debug
#sa_n_iter = 3
#print('WARNING: doing a debugging number of iterations')


# Hard constants
TRAINING_SET_FILE = paths.FIRNBERG_SPLIT_0_FILE

UNIREP_BATCH_SIZE = n_chains ## CHANGED relative to GFP script

# Use params defined in A003 models.py as these were used for the
# retrospective data efficiency work.
TOP_MODEL_ENSEMBLE_NMEMBERS = models.ENSEMBLED_RIDGE_PARAMS['n_members']
TOP_MODEL_SUBSPACE_PROPORTION = models.ENSEMBLED_RIDGE_PARAMS['subspace_proportion']
TOP_MODEL_NORMALIZE = models.ENSEMBLED_RIDGE_PARAMS['normalize']
TOP_MODEL_DO_SPARSE_REFIT = True
TOP_MODEL_PVAL_CUTOFF = models.ENSEMBLED_RIDGE_PARAMS['pval_cutoff']

SIM_ANNEAL_K = 1
SIM_ANNEAL_INIT_SEQ_MUT_RADIUS = 3


assert n_chains <= UNIREP_BATCH_SIZE

#####################

# Sync required data
print('Syncing data')
data_io_utils.sync_s3_path_to_local(TRAINING_SET_FILE, is_single_file=True)
data_io_utils.sync_s3_path_to_local(paths.EVOTUNING_CKPT_DIR)

# Set seeds. This locks in the set of training sequences
# This also locks in the initial sequences used for SA
# as well as each chain's mutation rate.
np.random.seed(seed)
random.seed(seed)

# Grab training data. note the subset we grab will be driven by the random seed above.
print('Setting up training data')
train_df = pd.read_csv(TRAINING_SET_FILE)
sub_train_df = train_df.sample(n=n_train_seqs)

print(sub_train_df.head())

train_seqs = list(sub_train_df['seq'])
train_qfunc = np.array(sub_train_df['quantitative_function'])



# Somewhat out of place, but set initial sequences for simulated annealing as well
# as the mutation rate for each chain.
init_seqs = A006_common.propose_seqs(
        [constants.BETA_LAC_AA_SEQ]*n_chains, 
        [SIM_ANNEAL_INIT_SEQ_MUT_RADIUS]*n_chains, 
        min_pos=A006_common.BLAC_LIB_REGION[0], 
        max_pos=A006_common.BLAC_LIB_REGION[1])
mu_muts_per_seq = 1.5*np.random.rand(n_chains) + 1
print('mu_muts_per_seq:', mu_muts_per_seq) # debug



# Set up base model
print('Setting up base model')
tf.reset_default_graph()

if model_name == 'ET_Global_Init_1':
    base_model = babbler(batch_size=UNIREP_BATCH_SIZE, model_path=paths.BLAC_ET_GLOBAL_INIT_1_WEIGHT_PATH)
    print('Loading weights from:', paths.BLAC_ET_GLOBAL_INIT_1_WEIGHT_PATH)
elif model_name == 'ET_Random_Init_1':
    base_model = babbler(batch_size=UNIREP_BATCH_SIZE, model_path=paths.BLAC_ET_RANDOM_INIT_1_WEIGHT_PATH)
    print('Loading weights from:', paths.BLAC_ET_RANDOM_INIT_1_WEIGHT_PATH)
elif model_name =='OneHot':
    # Just need it to generate one-hot reps.
    # Doing it this way to be consistent with the GFP pipeline
    base_model = BetaLacOneHotEncoder()
else:
    assert False, 'Unsupported base model'
    
    
    
# Start a tensorflow session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    # Generate representations
    def generate_reps(seq_list):        
        if 'babbler1900' == base_model.__class__.__name__:
            assert len(seq_list) <= UNIREP_BATCH_SIZE
            hiddens = base_model.get_all_hiddens(seq_list, sess)
            rep = np.stack([np.mean(s, axis=0) for s in hiddens],0)
            
        elif 'BetaLacOneHotEncoder' == base_model.__class__.__name__:
            rep = base_model.encode_seqs(seq_list)
            
        return rep
     
    print('Generating training seq reps')
    train_reps = generate_reps(train_seqs)
     
    # Build & train the top model.
    print('Building top model')
    top_model = A003_common.train_ensembled_ridge(
        train_reps, 
        train_qfunc, 
        n_members=TOP_MODEL_ENSEMBLE_NMEMBERS, 
        subspace_proportion=TOP_MODEL_SUBSPACE_PROPORTION,
        normalize=TOP_MODEL_NORMALIZE, 
        do_sparse_refit=TOP_MODEL_DO_SPARSE_REFIT, 
        pval_cutoff=TOP_MODEL_PVAL_CUTOFF
    )
    
    # Do simulated annealing
    def get_fitness(seqs):
        reps = generate_reps(seqs)
        yhat, yhat_std, yhat_mem = top_model.predict(reps, 
                return_std=True, return_member_predictions=True)
                
        nmut = utils.levenshtein_distance_matrix(
                [constants.BETA_LAC_AA_SEQ], list(seqs)).reshape(-1)
        
        mask = nmut > nmut_threshold
        yhat[mask] = -np.inf 
        yhat_std[mask] = 0 
        yhat_mem[mask,:] = -np.inf 
        
        return yhat, yhat_std, yhat_mem  
    
    sa_results = A006_common.anneal(
        init_seqs, 
        k=SIM_ANNEAL_K, 
        T_max=T_max, 
        mu_muts_per_seq=mu_muts_per_seq,
        get_fitness_fn=get_fitness,
        n_iter=sa_n_iter, 
        decay_rate=temp_decay_rate,
        min_mut_pos=min_mut_pos,
        max_mut_pos=max_mut_pos)

    
# Aggregate results and export
results = {
    'sa_results': sa_results,
    'top_model': top_model,
    'train_df': sub_train_df,
    'train_seq_reps': train_reps,
    'base_model': model_name
}
    
output_file = os.path.basename(output_file)
with open(output_file, 'wb') as f:
    pickle.dump(file=f, obj=results)
    
print('Syncing results to S3')
print('Post-publication note: Skipping. Public S3 sync for this bucket is disabled.')
# Sync up to S3.
#cmd = ('aws s3 cp %s s3://efficient-protein-design/chip_1/simulated_annealing/beta_lactamase/%s' % 
#       (output_file, output_file))
#subprocess.check_call(cmd, shell=True)


# Check if there is a standard out log file passed by Hyperborg. Sync it to s3 if so.
#possible_log = config_file.replace('.p', '.log')
#if os.path.exists(possible_log):
#    cmd = ('aws s3 cp %s s3://efficient-protein-design/chip_1/simulated_annealing/beta_lactamase/%s' % 
#       (possible_log, possible_log))
#    subprocess.check_call(cmd, shell=True)
    
