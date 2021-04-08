import sys
import os
import pickle
import subprocess

import numpy as np
import pandas as pd

from sklearn.linear_model import LassoLars, LassoLarsCV, Ridge, RidgeCV, BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor

from scipy.stats import ttest_ind, ttest_rel

sys.path.append('../common')
import data_io_utils
import paths
import utils
import constants

MAX_GFP_SEQ_LEN = 239

# Copied from A049m in mlpe-gfp-pilot
FP_HOMOLOGS_TEST_SET_PARENTS = np.array(['GFPhal', 'muGFP', 'hfriFP', 'TagGFP2', 'GFPxm191uv',
       'Superfolder_GFP', 'Folding_Reporter_GFP', 'Ypet', 'TagYFP',
       'Clover', 'mClover3', 'Citrine', 'mEmerald', 'mVenus',
       'J8VIQ3_9SPHN', 'sfGFP_internal'], dtype='<U26')

class EnsembledRidgeCV(object):
    
    def __init__(self, n_members, subspace_proportion=0.2, normalize=True, pval_cutoff=0.01, do_sparse_refit=False):
        self.n_members = n_members
        self.subspace_proportion = subspace_proportion
        self.normalize = normalize
        self.pval_cutoff = pval_cutoff
        self.do_sparse_refit = do_sparse_refit
        
    def fit(self, x_train, y_train):
        
        subspace_size = int(np.round(x_train.shape[1]*self.subspace_proportion))
        
        models = []
        for i in range(self.n_members):
            feature_idx = np.random.choice(x_train.shape[1], subspace_size, replace=False)
            observation_idx = np.random.choice(x_train.shape[0], x_train.shape[0], replace=True) # bootstrap
            
            x_train_sub = x_train[observation_idx]
            y_train_sub = y_train[observation_idx]
            
            models.append(
                {
                    'model': cv_train_ridge_with_sparse_refit(x_train_sub[:,feature_idx], y_train_sub, 
                            normalize=self.normalize, pval_cutoff=self.pval_cutoff, do_sparse_refit=self.do_sparse_refit),
                    'feature_idx': feature_idx
                }
            )
            
        self.model_ensemble = models
        
    def predict(self, x, return_std=False, return_member_predictions=False):
        yhats = [m['model'].predict(x[:, m['feature_idx']]) for m in self.model_ensemble]
        
        yhat_mat = np.stack(yhats) # [n_members x x.shape[0]]
        
        yhat_mu = np.mean(yhat_mat, axis=0)
        yhat_std = np.std(yhat_mat, axis=0)
        
        to_return = (yhat_mu,)
        
        if return_std:
            to_return += (yhat_std,)
            
        if return_member_predictions:
            to_return += (yhat_mat.T,) # [n_obs x ensemble members]
            
        if len(to_return) == 1:
            to_return = to_return[0]
        
        return to_return
        
        #if return_std:
        #    #Hacky solution. Doesn't capture shape of curve.
        #    return yhat_mu, yhat_std
        #else:
        #    return yhat_mu
        

def train_ensembled_ridge(x_train, y_train, n_members=100, subspace_proportion=1.0, pval_cutoff=0.01, 
        do_sparse_refit=False, normalize=True):
    
    model = EnsembledRidgeCV(n_members=n_members, subspace_proportion=subspace_proportion, 
            normalize=normalize, pval_cutoff=pval_cutoff, do_sparse_refit=do_sparse_refit)
    model.fit(x_train, y_train)
    
    return model

def cv_train_ridge_with_sparse_refit(x_train, y_train, pval_cutoff=0.01, do_sparse_refit=False, normalize=True):
    
    
    model = RidgeCV(alphas=np.logspace(-6,6,1000), gcv_mode='auto', normalize=normalize, store_cv_values=True)
    model.fit(x_train, y_train)
    
    if do_sparse_refit:
        best_alpha_idx = np.argwhere(model.alphas == model.alpha_).reshape(-1)[0]
        
        sparse_alpha_idx = -1
        for i in range(best_alpha_idx+1, len(model.alphas)):
            pval = ttest_rel(model.cv_values_[:,best_alpha_idx], 
                    model.cv_values_[:,i]).pvalue

            if pval < pval_cutoff:
                sparse_alpha_idx = i-1
                break

        if sparse_alpha_idx == -1:
            # take the sparsest solution as we couldn't
            # find a large enough alpha to have worse performance.
            sparse_alpha_idx = len(model.alphas)-1
            
        model_sparse = Ridge(alpha=model.alphas[sparse_alpha_idx], normalize=normalize)
        model_sparse.fit(x_train, y_train)
        
        model_sparse.best_alpha = model.alpha_
        model_sparse.sparse_alpha = model.alphas[sparse_alpha_idx]
        model_sparse.alpha_ = model.alphas[sparse_alpha_idx] 
        model_sparse.alphas = model.alphas
        
        return model_sparse

    else:
        return model

def train_blr(x_train, y_train, normalize=True):
    model = BayesianRidge(normalize=normalize)
    model.fit(x_train, y_train)
    
    return model

def cv_train_knn(x_train, y_train, pval_cutoff=0.001, do_sparse_refit=False, normalize=True):
    
    assert not do_sparse_refit, 'Sparse refit does not apply, left for compatibility just in case bc Grig does not understand pipeline'
    
    model = KNeighborsRegressor(n_neighbors=5, metric='euclidean')
    model.fit(x_train, y_train)
    
    return model

def cv_train_GP(x_train, y_train, pval_cutoff=0.001, kernel=None):
        
    #TODO
    model = GaussianProcessRegressor(kernel=kernel)
    model.fit(x_train, y_train)
    
    return model

def cv_train_lasso_lars_with_sparse_refit(x_train, y_train, pval_cutoff=0.001, do_sparse_refit=True):
    model = LassoLarsCV(n_jobs=-1, cv=min(x_train.shape[0], 10))
    model.fit(x_train, y_train)
    best_alpha_idx = int(np.argwhere(model.alpha_ == model.cv_alphas_))

    if do_sparse_refit:
        sparse_alpha_idx = -1
        for i in range(best_alpha_idx+1, len(model.cv_alphas_)):
            pval = ttest_ind(model.mse_path_[best_alpha_idx], 
                             model.mse_path_[i]).pvalue

            if pval < pval_cutoff:
                sparse_alpha_idx = i-1
                break

        if sparse_alpha_idx == -1:
            # take the sparsest solution
            sparse_alpha_idx = len(model.cv_alphas_)-1
            
        model_sparse = LassoLars(alpha=model.cv_alphas_[sparse_alpha_idx])
        model_sparse.fit(x_train, y_train)
        
        return model_sparse
    else:
        return model


def build_edit_string_substitutions_only(seq, wt_seq):
    """Builds a 1-indexed edit string btw seq and wt_seq where there are assumed
    to be no indels.
    This function doesn't use nwalign and instead does a char by char comparison.

    """
    # Collect parts of the edit string then combine at the end.
    es_parts = []

    for i, wt_char in enumerate(wt_seq):
        if seq[i] != wt_char:
            one_indexed_pos = i + 1
            es_parts.append('{orig}{edit_pos}{new}'.format(
                    orig=wt_char,
                    edit_pos=one_indexed_pos,
                    new=seq[i]))

    return es_parts


## Code to simplify generalization dataframes.

PARENT_MIN_BRIGHTNESS = 1.5
VARIANT_MAX_BRIGHTNESS = 0.7

def generate_simplified_and_fused_gen_set(sn_df, fp_df):
    # See helper functions below.
    
    simple_sn_df = generate_simplified_syn_neigh_gen_set(sn_df)
    simple_fp_df = generate_simplified_fp_homologs_gen_set(fp_df)
    
    simple_sn_df = simple_sn_df.assign(gen_set = 'simple_syn_neigh')
    simple_fp_df = simple_fp_df.assign(gen_set = 'simple_fp_homologs')
    
    
    cols_to_keep = ['seq', 'quantitative_function', 'gen_set']
    
    fused_df = pd.concat([simple_sn_df[cols_to_keep], simple_fp_df[cols_to_keep]], axis=0)
    fused_df.reset_index(inplace=True, drop=True)
    
    return fused_df
    
    
    

# SYNTHETIC NEIGHBORHOODS 
def generate_simplified_syn_neigh_gen_set(sn_df):
    """
    See A003-009 notebook. Simplifies gen set by subsetting to
    highly functional parent sequences and a large number of 
    non-functional sequences. Thus there are a handful of
    high-functioning seqs in a sea of non-functional ones.
    
    Importantly, non-functional seqs were generated from the same
    process as the functional ones.
    
    sn_df: synthetic neighborhoods gen set dataframe. Should
        be a specific split (e.g. split 0), but can be the full 
        dataframe as well.
    """
    
    # Parents dataframe. 
    sn_parents_df = pd.read_csv(paths.SYNTHETIC_NEIGH_PARENTS_INFO_FILE)
    sn_parents_df = pd.merge(sn_parents_df, sn_df, how='inner', on='seq')
    
    # Subset
    
    parent_mask = sn_parents_df['quantitative_function'] >= PARENT_MIN_BRIGHTNESS
    bright_parent_seqs = list(sn_parents_df[parent_mask]['seq'])

    to_keep = []
    for i,r in sn_df.iterrows():
        to_keep.append(r['seq'] in bright_parent_seqs or r['quantitative_function'] <= VARIANT_MAX_BRIGHTNESS)
    to_keep = np.array(to_keep)

    subset_sn_df = sn_df[to_keep]
    
    return subset_sn_df


# FP HOMOLOGS
def generate_fp_homologs_parents_df():

    #cmd = ('aws s3 cp s3://mlpe-data/FP_homologs/20190401_selected_aequorea_FPs_for_synthesis_with_well_info.csv ' +
    #       '../../data/20190401_selected_aequorea_FPs_for_synthesis_with_well_info.csv') 
    #subprocess.check_output(cmd, shell=True)

    parent_df_file = 'intermediate_files/20190401_selected_aequorea_FPs_for_synthesis_with_well_info.csv'
    fp_parents_df = pd.read_csv(parent_df_file)[['name', 'tranlsated_ligated_seq_orf']]

    fp_parents_df['seq'] = [s.replace('*', '') for s in list(fp_parents_df['tranlsated_ligated_seq_orf'])]
    fp_parents_df.drop(columns=['tranlsated_ligated_seq_orf'], inplace=True)
    
    return fp_parents_df

def generate_count_summary_df_for_exp9():
    # This is EXP9_AA_SEQ_COUNT_DF = os.path.join(EXP4_8_EXP9_DATA_DIR, 'Exp9_aa_seq_count_table.csv') 
    # in A049 common from mlpe-gfp-pilot
    #cmd = ('aws s3 cp s3://mlpe-data/2018_12_21_Exp4-8_Exp9/Exp9_aa_seq_count_table.csv ' +
    #       '../../data/Exp9_aa_seq_count_table.csv')
    #subprocess.check_output(cmd, shell=True)

    exp9_count_df = pd.read_csv('intermediate_files/Exp9_aa_seq_count_table.csv')
    channels_fl2_all_lasers = ['aa_seq', 'barcodes'] + ['EXP4_8-9_NGS_' + str(i) for i in range(22,30)]
    
    exp9_count_df = exp9_count_df[channels_fl2_all_lasers]

    exp9_nbins = np.array(np.sum(exp9_count_df.iloc[:,2:] > 0, axis=1))
    exp9_coverage = np.array(np.sum(exp9_count_df.iloc[:,2:], axis=1))

    exp9_summary_df = pd.DataFrame()
    exp9_summary_df['seq'] = exp9_count_df['aa_seq']
    exp9_summary_df['barcodes'] = exp9_count_df['barcodes']
    exp9_summary_df['nbins'] = exp9_nbins
    exp9_summary_df['coverage'] = exp9_coverage
    
    return exp9_summary_df

def generate_simplified_fp_homologs_gen_set(fp_df):
    """
    See A003-009 notebook. Simplifies gen set by subsetting to
    highly functional sequences and a large number of 
    non-functional sequences. Thus there are a handful of
    high-functioning seqs in a sea of non-functional ones.
    
    Importantly, non-functional seqs were generated from the same
    process as the functional ones.
    
    See comments below for how high-functioning sequences are 
    selected.
    
    fp_df: FP homologs gen set dataframe. Should
        be a specific split (e.g. split 0), but can be the full 
        dataframe as well.
    """

    # Load information about shuffling parents.
    fp_parents_df = generate_fp_homologs_parents_df()
    
    # Subset to information about test set parents. 
    TESTING_PARENT_NAMES = list(FP_HOMOLOGS_TEST_SET_PARENTS)
    mask = fp_parents_df['name'].apply(lambda x : x in TESTING_PARENT_NAMES)

    fp_test_parents_df = fp_parents_df[mask]
    test_parent_seqs = list(fp_test_parents_df['seq'])
    test_parent_names = np.array(fp_test_parents_df['name'])
    
    # Load count information for all exp9 variants before imputation
    exp9_summary_df = generate_count_summary_df_for_exp9()
    
    # Merge gen set df with count information. Count info df should
    # be a superset of gen set df in terms of seqs they both carry.
    old_nvar = fp_df.shape[0]
    fp_df = pd.merge(fp_df.copy(deep=True), exp9_summary_df, how='inner', on='seq')
    assert fp_df.shape[0] == old_nvar
        
    # Define criteria for high functioning sequences to keep in the 
    # simplified generalization set.
    test_parent_mask = fp_df['seq'].apply(lambda x: x in test_parent_seqs)
    cov_nbin_mask = np.logical_and(fp_df['coverage'] > 1000, fp_df['nbins'] >= 3) # >1K reads, and in >=3 bins.
    func_brightness_mask = fp_df['quantitative_function'] > PARENT_MIN_BRIGHTNESS # Use same from syn_neigh.
    
    print('FP homolog parents contained in gen set:', np.sum(test_parent_mask))

    # Overall, we keep a variant on the basis of it being highly functional if: 
    # (its a test parent or (has sufficent coverage and is in enough bins)) and has >= a certain qfunc.
    func_seq_mask = np.logical_and(np.logical_or(test_parent_mask, cov_nbin_mask), func_brightness_mask)
    nonfunc_seq_mask = fp_df['quantitative_function'] < VARIANT_MAX_BRIGHTNESS # slight adjustment relative to syn_neigh
    
    subset_fp_df = fp_df[np.logical_or(func_seq_mask, nonfunc_seq_mask)]
    return subset_fp_df
