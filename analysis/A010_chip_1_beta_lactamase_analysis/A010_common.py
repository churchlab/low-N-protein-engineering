import os
import sys
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy

sys.path.append('../common')
import data_io_utils
import paths
import constants
import utils

sys.path.append('../A003_policy_optimization/')
import A003_common

with open('blac_edit2qfunc.p', 'rb') as f:
    blac_edit2qfunc = pickle.load(f)

def calculate_predicted_fitness_based_on_singles(seq, edit2qfunc=blac_edit2qfunc, wt_seq=constants.BETA_LAC_AA_SEQ):
    """
    seq = string, a sequence
    edit2qfunc = a dictionary that maps single-edit edit strings to
        quantitative function.
    """
    
    edits = A003_common.build_edit_string_substitutions_only(seq=seq, wt_seq=wt_seq)
    
    if len(edits) > 0:
    
        qfuncs = []
        for es in edits:
            if es in edit2qfunc:
                qfuncs.append(edit2qfunc[es])
            else:
                qfuncs.append(np.nan)

        qfuncs = np.array(qfuncs)
        
        # WT + deltas
        add_model = edit2qfunc[None] + np.sum(qfuncs - edit2qfunc[None])
        #mult_model = scipy.stats.gmean(qfuncs)
        #min_model = np.min(qfuncs)
        
    else: # WT
        add_model = edit2qfunc[None]
        #mult_model = edit2qfunc[None]
        #min_model = edit2qfunc[None]
        
    return add_model #, mult_model, min_model 