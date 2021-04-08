import os
import sys
import warnings
import multiprocessing as mp
import random
import ast
import pickle
import copy

import numpy as np
import pandas as pd
import scipy

sys.path.append('../common')
import data_io_utils
import paths
import utils
import constants

from sklearn.gaussian_process.kernels import PairwiseKernel, RBF, ConstantKernel, DotProduct

import A003_common

UNIREP_GLOBAL_NPY = os.path.join(data_io_utils.S3_DATA_ROOT, 
        'datasets/for_acquisition', 'all_seqs_sark_synneigh_fphomologs_unirep_inferred_avg_hidden.npy')
UNIREP_GFP_ET_GLOBAL_INIT_1_NPY = os.path.join(data_io_utils.S3_DATA_ROOT, 
        'datasets/for_acquisition', 'all_seqs_sark_synneigh_fphomologs_ET_GLOBAL_INIT_1_avg_hidden.npy')
UNIREP_GFP_ET_GLOBAL_INIT_2_NPY = os.path.join(data_io_utils.S3_DATA_ROOT, 
        'datasets/for_acquisition', 'all_seqs_sark_synneigh_fphomologs_ET_GLOBAL_INIT_2_avg_hidden.npy')
UNIREP_GFP_ET_RANDOM_INIT_1_NPY = os.path.join(data_io_utils.S3_DATA_ROOT, 
        'datasets/for_acquisition', 'all_seqs_sark_synneigh_fphomologs_ET_RANDOM_INIT_1_avg_hidden.npy')

# Used to be the following. The above paths were re-inferences done on Aug 4, 2019. See 019,020,021 notebooks.
# Re-inferencing was done due to re-organization of model checkpoints and I wanted to make sure
# we'd still get the same results.
#UNIREP_GLOBAL_NPY = os.path.join(data_io_utils.S3_DATA_ROOT, 
#        'datasets/for_acquisition', 'all_seqs_sark_synneigh_fphomologs_unirep_inferred_avg_hidden.npy')
#GFP_ET_GLOBAL_INIT_1_NPY = os.path.join(data_io_utils.S3_DATA_ROOT, 
#        'datasets/for_acquisition', 'all_seqs_sark_synneigh_fphomologs_evotuned_unirep_inferred_CORRECT_avg_hidden.npy')
#GFP_ET_GLOBAL_INIT_2_NPY = os.path.join(data_io_utils.S3_DATA_ROOT, 
#        'datasets/for_acquisition', 'all_seqs_sark_synneigh_fphomologs_evotuned_unirep_2_inferred_avg_hidden.npy')
#GFP_ET_RANDOM_INIT_1_NPY = os.path.join(data_io_utils.S3_DATA_ROOT, 
#        'datasets/for_acquisition', 'all_seqs_sark_synneigh_fphomologs_evotuned_random_init_inferred_avg_hidden.npy')


ENSEMBLED_RIDGE_PARAMS = {
    'n_members': 100,
    'subspace_proportion': 0.5,
    'pval_cutoff': 0.01,
    'normalize': True
} # Chip 1 designs are based on these numbers, don't change.

def build_seq2rep_dict(rep_npy):
    data_dir = os.path.join(data_io_utils.S3_DATA_ROOT, 'datasets/for_acquisition')
    all_seqs_file = os.path.join(data_dir, 'all_seqs_sark_synneigh_fphomologs_for_unirep_inference.txt')
    
    print(rep_npy)
    
    with open(all_seqs_file) as f:
        seqs = f.readlines()
    seqs = [s.strip() for s in seqs] 
    
    reps = np.load(rep_npy)
    
    seq2rep = {}
    for i,seq in enumerate(seqs):
        seq2rep[seq] = reps[i].astype(float)
        
    return seq2rep

###########################
# >>>> PARENT MODEL CLASSES 
###########################

class FixedRepRegressionModel(object): ## DO NOT USE DIRECTLY. PARENT CLASS.
    """
    Generic parent class for all fixed rep + LassoLars or Ridge or KNN or GP top models.
    """
    
    def __init__(self, rep_npy, top_model_type, rep_name_to_include_fpbase=None):
        self.seq2rep = build_seq2rep_dict(rep_npy)
        
        if rep_name_to_include_fpbase is not None:
            assert rep_name_to_include_fpbase in ['unirep', 'evotuned_unirep']
            
            fpbase_rep_map_file = os.path.join(data_io_utils.S3_DATA_ROOT, 
                    'datasets/for_acquisition', 'fpbase_repname2seq2repvec.p')
            
            with open(fpbase_rep_map_file, 'rb') as f:
                repname2seq2repvec = pickle.load(f)
            
            seq2repvec = repname2seq2repvec[rep_name_to_include_fpbase]
            seq2repvec.update(self.seq2rep)
            
            self.seq2rep = seq2repvec
                
        
        self.top_model_type = top_model_type
        self.sparse_refit = False
        self.model = None
        self.alpha_in_interior = None
        
        self.kernel = PairwiseKernel(metric="cosine")
    
    def get_name(self):
        return self.__class__.__name__
        
    def save(self, output_file):
        save_dict = {
            'alpha_in_interior': self.alpha_in_interior,
            'model': self.model,
            'top_model_type': self.top_model_type,
            'sparse_refit': self.sparse_refit,
        }
        
        with open(output_file, 'wb') as f:
            pickle.dump(obj=save_dict, file=f)
    
    def interior_alpha_found(self):
        if self.top_model_type == 'Ridge':
            idx = np.argwhere(self.model.alpha_ == self.model.alphas).reshape(-1)[0]
            return idx > 0 and idx < (len(self.model.alphas)-1)
        else:
            return True
        
    def encode_seqs(self, aa_seqs):
        # UniRep represent using a look-up table.
        return np.stack([self.seq2rep[s] for s in aa_seqs], 0)
    
    def train(self, aa_seqs, qfunc): # Meets required interface
        # Suppress annoying LASSO warnings. These warnings
        # just mention that lasso has a hard time doing
        # certain matrix stuff when the number of training points
        # are low.
        warnings.filterwarnings('ignore')
        
        encoded_seqs = self.encode_seqs(aa_seqs)
        
        if self.top_model_type == 'LassoLars':
            # Train lasso model. 
            # DO NOT USE sparse refit.
            self.model = A003_common.cv_train_lasso_lars_with_sparse_refit(
                encoded_seqs, qfunc, do_sparse_refit=self.sparse_refit)
            
        elif self.top_model_type == 'Ridge':
            self.model = A003_common.cv_train_ridge_with_sparse_refit(
                encoded_seqs, qfunc, do_sparse_refit=self.sparse_refit)
            
        elif self.top_model_type == 'BayesianRidge':
            self.model = A003_common.train_blr(encoded_seqs, qfunc)
            
        elif self.top_model_type == 'EnsembledRidge':
            self.model = A003_common.train_ensembled_ridge(
                encoded_seqs, 
                qfunc, 
                do_sparse_refit=self.sparse_refit, 
                n_members=ENSEMBLED_RIDGE_PARAMS['n_members'], 
                subspace_proportion=ENSEMBLED_RIDGE_PARAMS['subspace_proportion'], 
                pval_cutoff=ENSEMBLED_RIDGE_PARAMS['pval_cutoff'], 
                normalize=ENSEMBLED_RIDGE_PARAMS['normalize']
            ) 
            
        elif self.top_model_type == 'KNN':
            self.model = A003_common.cv_train_knn(
                encoded_seqs, qfunc, do_sparse_refit=self.sparse_refit)
        
        elif self.top_model_type == 'GP':
            self.model = A003_common.cv_train_GP(
                encoded_seqs, qfunc, kernel=self.kernel)
            
        self.alpha_in_interior = self.interior_alpha_found()
        
        warnings.filterwarnings('default')
        
    def predict(self, aa_seqs): # Meets required interface
        assert self.model is not None
        
        encoded_seqs = self.encode_seqs(aa_seqs)
        
        # Predict
        return self.model.predict(encoded_seqs)
    
class FixedRepLassoLarsModel(FixedRepRegressionModel):
    
    def __init__(self, rep_npy, rep_name_to_include_fpbase=None):
        TOP_MODEL_TYPE = 'LassoLars'
        
        super(FixedRepLassoLarsModel, self).__init__(rep_npy, top_model_type=TOP_MODEL_TYPE,
                rep_name_to_include_fpbase=rep_name_to_include_fpbase)
        
class FixedRepRidgeModel(FixedRepRegressionModel):
    
    def __init__(self, rep_npy, rep_name_to_include_fpbase=None):
        TOP_MODEL_TYPE = 'Ridge'
        
        super(FixedRepRidgeModel, self).__init__(rep_npy, top_model_type=TOP_MODEL_TYPE,
                rep_name_to_include_fpbase=rep_name_to_include_fpbase)
        

class FixedRepEnsembledRidgeModel(FixedRepRegressionModel):
    
    def __init__(self, rep_npy, rep_name_to_include_fpbase=None):
        TOP_MODEL_TYPE = 'EnsembledRidge'
        
        super(FixedRepEnsembledRidgeModel, self).__init__(rep_npy, top_model_type=TOP_MODEL_TYPE,
                rep_name_to_include_fpbase=rep_name_to_include_fpbase)
        

        
class OneHotRegressionModel(object):
    
    def __init__(self, top_model_type):
        self.top_model_type = top_model_type
        self.sparse_refit=False
        self.model = None
        self.alpha_in_interior = None
    
    def get_name(self):
        return self.__class__.__name__
    
    def save(self, output_file):
        save_dict = {
            'alpha_in_interior': self.alpha_in_interior,
            'model': self.model,
            'top_model_type': self.top_model_type,
            'sparse_refit': self.sparse_refit,
        }
        
        with open(output_file, 'wb') as f:
            pickle.dump(obj=save_dict, file=f)
        
    def interior_alpha_found(self):
        if self.top_model_type == 'Ridge':
            idx = np.argwhere(self.model.alpha_ == self.model.alphas).reshape(-1)[0]
            return idx > 0 and idx < (len(self.model.alphas)-1)
        else:
            return True
        
    def encode_seqs(self, aa_seqs):
        # NOTE: HACKY!!
        # Length differences in all of GFPs we look at here
        # are due to the second amino acid being there or not.
        # Adjust for this, when doing linear regression.
        # This is like doing an alignment to a reference, and
        # wildcarding an insertion.
        aa_seqs = copy.deepcopy(aa_seqs)
        for i,seq in enumerate(aa_seqs):
            if len(seq) == A003_common.MAX_GFP_SEQ_LEN-1:
                seq = seq[0] + 'X' + seq[1:]
                aa_seqs[i] = seq
        
        for seq in aa_seqs:
            assert len(seq) == A003_common.MAX_GFP_SEQ_LEN
        
        # One hot encode
        encoded_seqs = utils.encode_aa_seq_list_as_matrix_of_flattened_one_hots(
            aa_seqs, pad_to_len=A003_common.MAX_GFP_SEQ_LEN, wildcard='X')
        
        return encoded_seqs
    
    def train(self, aa_seqs, qfunc): # Meets required interface
        # Suppress annoying LASSO warnings. These warnings
        # just mention that lasso has a hard time doing
        # certain matrix stuff when the number of training points
        # are low.
        warnings.filterwarnings('ignore')
        
        encoded_seqs = self.encode_seqs(aa_seqs)
        
        if self.top_model_type == 'LassoLars':
            # Train lasso model. 
            self.model = A003_common.cv_train_lasso_lars_with_sparse_refit(
                encoded_seqs, qfunc, do_sparse_refit=self.sparse_refit)
        elif self.top_model_type == 'Ridge':
            self.model = A003_common.cv_train_ridge_with_sparse_refit(
                encoded_seqs, qfunc, do_sparse_refit=self.sparse_refit)
        elif self.top_model_type == 'BayesianRidge':
            self.model = A003_common.train_blr(encoded_seqs, qfunc)
            
        elif self.top_model_type == 'EnsembledRidge':
            self.model = A003_common.train_ensembled_ridge(
                encoded_seqs, 
                qfunc, 
                do_sparse_refit=self.sparse_refit, 
                n_members=ENSEMBLED_RIDGE_PARAMS['n_members'], 
                subspace_proportion=ENSEMBLED_RIDGE_PARAMS['subspace_proportion'], 
                pval_cutoff=ENSEMBLED_RIDGE_PARAMS['pval_cutoff'], 
                normalize=ENSEMBLED_RIDGE_PARAMS['normalize']
            ) 
            
        elif self.top_model_type == 'GP':
            self.model = A003_common.cv_train_GP(
                encoded_seqs, qfunc, kernel=self.kernel)
            
        self.alpha_in_interior = self.interior_alpha_found()
        
        warnings.filterwarnings('default')
        
    def predict(self, aa_seqs): # Meets required interface
        assert self.model is not None

        encoded_seqs = self.encode_seqs(aa_seqs)
        
        # Predict
        return self.model.predict(encoded_seqs)
        
        
###########################
# PARENT MODEL CLASSES <<<<
###########################

class Doc2VecLassoLarsModel(FixedRepLassoLarsModel):
    
    def __init__(self, include_fpbase_seqs=False):
        REP_NPY = os.path.join(data_io_utils.S3_DATA_ROOT, 
                'datasets/for_acquisition', 
                'Yang_2018_random_3_7.pkl_effprot_seqs_.npy')
        
        assert not include_fpbase_seqs, 'Unsupported for doc2vec'
        rn = None
        
        super(Doc2VecLassoLarsModel, self).__init__(REP_NPY, rep_name_to_include_fpbase=rn)  

class Doc2VecRidgeModel(FixedRepRidgeModel):
    
    def __init__(self, include_fpbase_seqs=False, do_sparse_refit=False):
        REP_NPY = os.path.join(data_io_utils.S3_DATA_ROOT, 
                'datasets/for_acquisition', 
                'Yang_2018_random_3_7.pkl_effprot_seqs_.npy')
        
        assert not include_fpbase_seqs, 'Unsupported for doc2vec'
        rn = None
        
        super(Doc2VecRidgeModel, self).__init__(REP_NPY, rep_name_to_include_fpbase=rn)       
        self.sparse_refit = do_sparse_refit # property of super.
        
    def get_name(self):
        name = self.__class__.__name__
        if self.sparse_refit:
            name += '_sparse_refit'
            
        return name
    
class Doc2VecEnsembledRidgeModel(FixedRepEnsembledRidgeModel):
    
    def __init__(self, include_fpbase_seqs=False, do_sparse_refit=False):
        REP_NPY = os.path.join(data_io_utils.S3_DATA_ROOT, 
                'datasets/for_acquisition', 
                'Yang_2018_random_3_7.pkl_effprot_seqs_.npy')
        
        assert not include_fpbase_seqs, 'Unsupported for doc2vec'
        rn = None
        
        super(Doc2VecEnsembledRidgeModel, self).__init__(REP_NPY, rep_name_to_include_fpbase=rn)
        self.sparse_refit = do_sparse_refit # property of super.
        
    def get_name(self):
        name = self.__class__.__name__
        if self.sparse_refit:
            name += '_sparse_refit'
            
        return name
        
        
        
    
class UniRepLassoLarsModel(FixedRepLassoLarsModel):
    
    def __init__(self, include_fpbase_seqs=False):
        REP_NPY = UNIREP_GLOBAL_NPY
        
        if include_fpbase_seqs:
            rn = 'unirep'
        else:
            rn = None
        
        super(UniRepLassoLarsModel, self).__init__(REP_NPY, rep_name_to_include_fpbase=rn)
        
class UniRepRidgeModel(FixedRepRidgeModel):
    
    def __init__(self, include_fpbase_seqs=False):
        REP_NPY = UNIREP_GLOBAL_NPY
        
        if include_fpbase_seqs:
            rn = 'unirep'
        else:
            rn = None
        
        super(UniRepRidgeModel, self).__init__(REP_NPY, rep_name_to_include_fpbase=rn)
        
class UniRepEnsembledRidgeModel(FixedRepEnsembledRidgeModel):
    
    def __init__(self, include_fpbase_seqs=False):
        REP_NPY = UNIREP_GLOBAL_NPY
        
        if include_fpbase_seqs:
            rn = 'unirep'
        else:
            rn = None
        
        super(UniRepEnsembledRidgeModel, self).__init__(REP_NPY, rep_name_to_include_fpbase=rn)
        

class EvotunedUniRepLassoLarsModel(FixedRepLassoLarsModel):
    
    def __init__(self, include_fpbase_seqs=False):
        REP_NPY = UNIREP_GFP_ET_GLOBAL_INIT_1_NPY
        
        if include_fpbase_seqs:
            rn = 'evotuned_unirep'
        else:
            rn = None
        
        super(EvotunedUniRepLassoLarsModel, self).__init__(REP_NPY, rep_name_to_include_fpbase=rn)
        
class EvotunedUniRepRidgeModel(FixedRepRidgeModel):
    
    def __init__(self, include_fpbase_seqs=False, do_sparse_refit=False):
        REP_NPY = UNIREP_GFP_ET_GLOBAL_INIT_1_NPY
        
        if include_fpbase_seqs:
            rn = 'evotuned_unirep'
        else:
            rn = None
        
        super(EvotunedUniRepRidgeModel, self).__init__(REP_NPY, rep_name_to_include_fpbase=rn)
        self.sparse_refit = do_sparse_refit # property of super.
        
    def get_name(self):
        if self.sparse_refit:
            return self.__class__.__name__  + '_sparse_refit'
        else:
            return self.__class__.__name__
        
class EvotunedUniRep2RidgeModel(FixedRepRidgeModel):
    
    def __init__(self, include_fpbase_seqs=False, do_sparse_refit=False):
        REP_NPY = UNIREP_GFP_ET_GLOBAL_INIT_2_NPY
        
        if include_fpbase_seqs:
            rn = 'evotuned_unirep_2'
        else:
            rn = None
        
        super(EvotunedUniRep2RidgeModel, self).__init__(REP_NPY, rep_name_to_include_fpbase=rn)
        self.sparse_refit = do_sparse_refit # property of super.
        
    def get_name(self):
        if self.sparse_refit:
            return self.__class__.__name__  + '_sparse_refit'
        else:
            return self.__class__.__name__        
        
        
class EvotunedRandomInitRidgeModel(FixedRepRidgeModel):
    
    def __init__(self, include_fpbase_seqs=False, do_sparse_refit=False):
        REP_NPY = UNIREP_GFP_ET_RANDOM_INIT_1_NPY
        
        if include_fpbase_seqs:
            rn = 'evotuned_random_init'
        else:
            rn = None
        
        super(EvotunedRandomInitRidgeModel, self).__init__(REP_NPY, rep_name_to_include_fpbase=rn)
        self.sparse_refit = do_sparse_refit # property of super.
        
    def get_name(self):
        if self.sparse_refit:
            return self.__class__.__name__  + '_sparse_refit'
        else:
            return self.__class__.__name__
        
        
#### USE code below for any UniRep + Ridge, Lasso, EnsembledRidge model
# Not the old classes above.
class UniRepLassoLarsModel(FixedRepLassoLarsModel):
    
    def __init__(self, include_fpbase_seqs=False, do_sparse_refit=False, rep='global'):
        
        if rep == 'global':
            REP_NPY = UNIREP_GLOBAL_NPY
        elif rep == 'et_global_init_1':
            REP_NPY = UNIREP_GFP_ET_GLOBAL_INIT_1_NPY
        elif rep == 'et_global_init_2':
            REP_NPY = UNIREP_GFP_ET_GLOBAL_INIT_2_NPY
        elif rep == 'et_random_init_1':
            REP_NPY = UNIREP_GFP_ET_RANDOM_INIT_1_NPY
        
        self.rep = rep
        
        super(UniRepLassoLarsModel, self).__init__(REP_NPY, rep_name_to_include_fpbase=None)
        self.sparse_refit = do_sparse_refit
        
    def get_name(self):
        name = self.__class__.__name__ + '_' + self.rep
                
        if self.sparse_refit:
            name += '_sparse_refit'
        
        return name
    

    
class UniRepRidgeModel(FixedRepRidgeModel):
    
    def __init__(self, include_fpbase_seqs=False, do_sparse_refit=False, rep='global'):
        
        if rep == 'global':
            REP_NPY = UNIREP_GLOBAL_NPY
        elif rep == 'et_global_init_1':
            REP_NPY = UNIREP_GFP_ET_GLOBAL_INIT_1_NPY
        elif rep == 'et_global_init_2':
            REP_NPY = UNIREP_GFP_ET_GLOBAL_INIT_2_NPY
        elif rep == 'et_random_init_1':
            REP_NPY = UNIREP_GFP_ET_RANDOM_INIT_1_NPY
        
        self.rep = rep
        
        super(UniRepRidgeModel, self).__init__(REP_NPY, rep_name_to_include_fpbase=None)
        self.sparse_refit = do_sparse_refit
        
    def get_name(self):
        name = self.__class__.__name__ + '_' + self.rep
                
        if self.sparse_refit:
            name += '_sparse_refit'
        
        return name
        
        
class UniRepEnsembledRidgeModel(FixedRepEnsembledRidgeModel):
    
    def __init__(self, include_fpbase_seqs=False, do_sparse_refit=False, rep='global'):
        
        if rep == 'global':
            REP_NPY = UNIREP_GLOBAL_NPY
        elif rep == 'et_global_init_1':
            REP_NPY = UNIREP_GFP_ET_GLOBAL_INIT_1_NPY
        elif rep == 'et_global_init_2':
            REP_NPY = UNIREP_GFP_ET_GLOBAL_INIT_2_NPY
        elif rep == 'et_random_init_1':
            REP_NPY = UNIREP_GFP_ET_RANDOM_INIT_1_NPY
        
        self.rep = rep
        
        super(UniRepEnsembledRidgeModel, self).__init__(REP_NPY, rep_name_to_include_fpbase=None)
        self.sparse_refit = do_sparse_refit
        
    def get_name(self):
        name = self.__class__.__name__ + '_' + self.rep
                
        if self.sparse_refit:
            name += '_sparse_refit'
        
        return name    
    
        
        


class LassoLarsModel(OneHotRegressionModel):
    
    def __init__(self):
        super(LassoLarsModel, self).__init__(top_model_type='LassoLars')
        
class RidgeModel(OneHotRegressionModel):
    
    def __init__(self, do_sparse_refit=False):
        super(RidgeModel, self).__init__(top_model_type='Ridge')
        self.sparse_refit = do_sparse_refit # property of super
        
    def get_name(self):
        if self.sparse_refit:
            return self.__class__.__name__  + '_sparse_refit'
        else:
            return self.__class__.__name__
         
        
class EnsembledRidgeModel(OneHotRegressionModel):
    
    def __init__(self, do_sparse_refit=False):
        super(EnsembledRidgeModel, self).__init__(top_model_type='EnsembledRidge')
        self.sparse_refit=do_sparse_refit
        
    def get_name(self):
        name = self.__class__.__name__
        if self.sparse_refit:
            name += '_sparse_refit'
            
        return name
    
class SparseRefitLassoLarsModel(LassoLarsModel):
    
    def __init__(self):
        super(SparseRefitLassoLarsModel, self).__init__()
        self.sparse_refit = True
