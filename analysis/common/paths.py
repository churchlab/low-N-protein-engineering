import os
import sys

import data_io_utils

EVOTUNING_CKPT_DIR = os.path.join(data_io_utils.S3_DATA_ROOT, 'evotuning_checkpoints')

RANDOM_WEIGHTS_WEIGHT_PATH = os.path.join(EVOTUNING_CKPT_DIR, '1900_weights_random')

GFP_ET_GLOBAL_INIT_1_WEIGHT_PATH = os.path.join(EVOTUNING_CKPT_DIR, 'gfp/unirep_global_init_1')
GFP_ET_GLOBAL_INIT_2_WEIGHT_PATH = os.path.join(EVOTUNING_CKPT_DIR, 'gfp/unirep_global_init_2')
GFP_ET_RANDOM_INIT_1_WEIGHT_PATH = os.path.join(EVOTUNING_CKPT_DIR, 'gfp/unirep_random_init_1')

BLAC_ET_GLOBAL_INIT_1_WEIGHT_PATH = os.path.join(EVOTUNING_CKPT_DIR,
        'beta_lactamase/unirep_global_init_1')
BLAC_ET_RANDOM_INIT_1_WEIGHT_PATH = os.path.join(EVOTUNING_CKPT_DIR,
        'beta_lactamase/unirep_random_init_1')


DATASETS_DIR = os.path.join(data_io_utils.S3_DATA_ROOT, 'datasets')

SARKISYAN_DATA_FILE = os.path.join(DATASETS_DIR, 'sarkisyan.csv')
FPBASE_DATA_FILE = os.path.join(DATASETS_DIR, 'FPBase_parents_and_simulated_mutants.csv')
SYNTHETIC_NEIGH_DATA_FILE = os.path.join(DATASETS_DIR, 'Exp4-8_inferred_brightness_parents_assigned.csv')
SYNTHETIC_NEIGH_PARENTS_INFO_FILE = os.path.join(DATASETS_DIR, 'Exp4-8_parents_id_and_seq.csv')
FP_HOMOLOGS_DATA_FILE = os.path.join(DATASETS_DIR, 'Exp9_all_ex_lasers_FL2_inferred_brightness_parents_decomposed_tts.csv')



################################################################################
# TTS SPLITS RELATED
################################################################################
# Relevant analysis dirs: A003
TTS_SPLITS_DIR = os.path.join(DATASETS_DIR, 'tts_splits')
DATA_DISTRIBUTIONS_DIR = os.path.join(TTS_SPLITS_DIR, 'data_distributions')
GEN_SETS_SPLITS_DIR = os.path.join(TTS_SPLITS_DIR, 'generalization_sets')

## Training sets (data distributions)
# Splits 1 and 2 blocked for now.

# Sarkisyan
SARKISYAN_SPLIT_0_FILE = os.path.join(DATA_DISTRIBUTIONS_DIR, 
        'sarkisyan_split_0.csv')
SARKISYAN_SPLIT_1_FILE = os.path.join(DATA_DISTRIBUTIONS_DIR, 
        'sarkisyan_split_1.csv')
SARKISYAN_SPLIT_2_FILE = os.path.join(DATA_DISTRIBUTIONS_DIR, 
        'sarkisyan_split_2.csv')
    
# FP Homologs training data distributions
FP_HOMOLOGS_DATA_DIST_SPLIT_0_FILE = os.path.join(DATA_DISTRIBUTIONS_DIR, 
        'fp_homologs_data_dist_split_0.csv')
FP_HOMOLOGS_DATA_DIST_SPLIT_1_FILE = None # os.path.join(DATA_DISTRIBUTIONS_DIR, 
        #'fp_homologs_data_dist_split_1.csv')
FP_HOMOLOGS_DATA_DIST_SPLIT_2_FILE = None # os.path.join(DATA_DISTRIBUTIONS_DIR, 
        # 'fp_homologs_data_dist_split_2.csv')


## Generalization sets. 
# Splits 1 and 2 blocked for now.

# Synthetic neighborhoods
SYNTHETIC_NEIGH_SPLIT_0_FILE = os.path.join(GEN_SETS_SPLITS_DIR, 
        'Exp4-8_inferred_brightness_parents_assigned_split_0.csv')
SYNTHETIC_NEIGH_SPLIT_1_FILE = os.path.join(GEN_SETS_SPLITS_DIR, 
        'Exp4-8_inferred_brightness_parents_assigned_split_1.csv')
SYNTHETIC_NEIGH_SPLIT_2_FILE = os.path.join(GEN_SETS_SPLITS_DIR, 
        'Exp4-8_inferred_brightness_parents_assigned_split_2.csv')
    
    
# FP Homologs generalization sets
FP_HOMOLOGS_GEN_SPLIT_0_FILE = os.path.join(GEN_SETS_SPLITS_DIR, 
        'fp_homologs_gen_split_0.csv')
FP_HOMOLOGS_GEN_SPLIT_1_FILE = os.path.join(GEN_SETS_SPLITS_DIR, 
        'fp_homologs_gen_split_1.csv')
FP_HOMOLOGS_GEN_SPLIT_2_FILE = os.path.join(GEN_SETS_SPLITS_DIR, 
        'fp_homologs_gen_split_2.csv')


# FPBase
FPBASE_SPLIT_0_FILE = os.path.join(GEN_SETS_SPLITS_DIR, 
        'FPBase_parents_and_simulated_mutants_split_0.csv')
FPBASE_SPLIT_1_FILE = None # os.path.join(GEN_SETS_SPLITS_DIR, 
        #'FPBase_parents_and_simulated_mutants_split_1.csv')
FPBASE_SPLIT_2_FILE = None # os.path.join(GEN_SETS_SPLITS_DIR, 
        #'FPBase_parents_and_simulated_mutants_split_2.csv')

    
## Beta lactamase
FIRNBERG_SPLIT_0_FILE = os.path.join(DATA_DISTRIBUTIONS_DIR,
        'firnberg_split_0.csv')
FIRNBERG_SPLIT_1_FILE = os.path.join(DATA_DISTRIBUTIONS_DIR,
        'firnberg_split_1.csv')
FIRNBERG_SPLIT_2_FILE = os.path.join(DATA_DISTRIBUTIONS_DIR,
        'firnberg_split_2.csv')

################################################################################
# POLICY EVALUATION RESULTS RELATED
################################################################################
POLICY_EVAL_DIR = os.path.join(data_io_utils.S3_DATA_ROOT, 'policy_evaluation')

POLICY_EVAL_SPLIT_0_DIR = os.path.join(POLICY_EVAL_DIR, 'split_0')
POLICY_EVAL_SPLIT_1_DIR = os.path.join(POLICY_EVAL_DIR, 'split_1')
POLICY_EVAL_SPLIT_2_DIR = os.path.join(POLICY_EVAL_DIR, 'split_2')

