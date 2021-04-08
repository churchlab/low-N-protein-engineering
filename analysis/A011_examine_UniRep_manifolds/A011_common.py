import os
import sys
import warnings
import random
import pickle
import glob
import time

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import scipy
import sklearn

from matplotlib.colors import Normalize 
from scipy.interpolate import interpn
from matplotlib import cm
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA, MiniBatchSparsePCA, SparsePCA
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from scipy.stats import wilcoxon
from scipy.stats import gaussian_kde

sys.path.append('../common')
import data_io_utils
import paths
import utils
import constants
import plot_style_utils

sys.path.append('../A003_policy_optimization/')
import A003_common
import models

sys.path.append('../A006_simulated_annealing/')
from unirep import babbler1900 as babbler
from data_utils import aa_seq_to_int

def calc_seq_loglike(seq, logits, method='smart', plot=False):
    """
    seq: Amino acid sequence (str)
    logits: A [seq_length x vocab_size] array
    """
    
    # Convert to integer seq, and drop first token since 
    # logits are next char predictions.
    iseq = aa_seq_to_int(seq)[1:]
    iseq = [i-1 for i in iseq] # subtract 1 as logits dont consider pad.
    
    if plot:
        plt.figure(figsize=(20,4))
        plt.imshow(logits.T, aspect='auto')
        for i in range(len(iseq)):
            plt.plot(i, iseq[i], '.k')
        plt.show()
        
    if method == 'dumb':
        sm = scipy.special.softmax(logits, axis=1)
        
        log_like = 0
        for i in range(sm.shape[0]):
            log_like += np.log(sm[i, iseq[i]])
    
    if method == 'smart':
        
        lse = scipy.special.logsumexp(logits, axis=1)
        aa_logits = np.array([logits[i, iseq[i]] for i in range(logits.shape[0])])
        
        log_like = np.sum(aa_logits - lse)                   
            
    return log_like


def batch_avg_hidden(b, seq_list, sess, return_logits=False): # From A003/019
    start_time = time.time()
    
    if return_logits:
        hiddens, logits = b.get_all_hiddens(seq_list, sess, return_logits=return_logits)
    else:
        hiddens = b.get_all_hiddens(seq_list, sess)
        
    avg_hidden = np.stack([np.mean(s, axis=0) for s in hiddens],0)
    
    print('Time taken to do batch:', time.time() - start_time)
    
    if return_logits:
        return avg_hidden, logits
    else:
        return avg_hidden

def inference_seqs(seqs, model_weight_path, batch_size=500, return_loglikes=False):
    tf.reset_default_graph()
    
    b = babbler(batch_size=batch_size, model_path=model_weight_path)
    
    with tf.Session() as sess:
        print('Initializing variables')
        sess.run(tf.global_variables_initializer())

        avg_hiddens = []
        logits = []
        for i in range(0, len(seqs), batch_size):
            print()

            min_idx = i
            max_idx = min(i+batch_size,len(seqs))
            print(min_idx, max_idx)
            
            if return_loglikes:
                ah, lg = batch_avg_hidden(b, seqs[min_idx:max_idx], sess, return_logits=return_loglikes)
                avg_hiddens.append(ah)
                logits += lg
            else:
                avg_hiddens.append(
                    batch_avg_hidden(b, seqs[min_idx:max_idx], sess))
            
    all_seq_avg_hidden = np.vstack(avg_hiddens)
    
    if return_loglikes:
        assert len(seqs) == len(logits)
        for i in range(len(seqs)):
            assert len(seqs[i])+1 == logits[i].shape[0], (len(seqs[i]), logits[i].shape[0])
        
        log_likes = np.array([calc_seq_loglike(seqs[i], logits[i]) for i in range(len(seqs))])
        return all_seq_avg_hidden, logits, log_likes
    else:
        return all_seq_avg_hidden

def load_seq2rep_and_pca(protein, unirep_model, load_pca=True):
    
    #pickle_file = os.path.join(paths.DATASETS_DIR, 'A011', '%s_seq2rep_%s.p' % (protein, unirep_model))
    pickle_file = os.path.join(paths.DATASETS_DIR, 'A011', '%s_seq2rep_%s_v2.p' % (protein, unirep_model))
    print(pickle_file)
    with open(pickle_file, 'rb') as f:
        seq2rep = pickle.load(f)
        
    if load_pca:
        pca_file = '%s_%s_PCA.p' % (protein, unirep_model)
        print(pca_file)
        with open(pca_file, 'rb') as f:
            pca = pickle.load(f)
    else:
        pca = None

    return seq2rep, pca

def generate_reps(seqs, seq2rep, return_loglikes=False):
    reps = []
    log_likes = []
    for s in seqs:
        out = seq2rep[s]
        if type(out) is tuple:
            reps.append(out[0])
            log_likes.append(out[1])
        else:
            reps.append(out)
    
    if return_loglikes:
        return np.vstack(reps), np.array(log_likes)
    else:
        return np.vstack(reps)

def compute_PCs_for_seq(seqs, seq2rep, pca):
    reps = generate_reps(seqs, seq2rep)
    return pca.transform(reps), reps

def data2grid(s, z, ngrid=20):
    xlim, ylim = auto_set_xy_lims(s, None, apply_plot=False)
    
    xv = np.linspace(xlim[0], xlim[1], ngrid+1)
    yv = np.linspace(ylim[0], ylim[1], ngrid+1)
    
    M = np.zeros((ngrid, ngrid))
    for i in range(ngrid):
        for j in range(ngrid):
            mask = np.logical_and(
                np.logical_and(s[:,0] >= xv[i], s[:,0] < xv[i+1]),
                np.logical_and(s[:,1] >= yv[j], s[:,1] < yv[j+1])
            )
                        
            M[i,j] = np.mean(np.append(z[mask], [0]))
            
    return M.T



def compute_data_for_low_dim_manifold_viz(basis_seqs, vis_seqs, generate_reps):
    dr = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=100))])
    
    print('Generating basis reps')
    basis_reps, _ = generate_reps(basis_seqs)
    
    print('Generating viz reps')
    vis_reps, log_likes = generate_reps(vis_seqs)
    
    print('Fitting basis reps with PCA')
    dr.fit(basis_reps)
    
    print('Transforming viz reps with learned PCA')
    s = dr.transform(vis_reps)
    
    return s, dr, generate_reps, log_likes

def density_scatter( x , y, ax = None, sort = True, bins = 20, **kwargs )   :
    """
    Scatter plot colored by 2d histogram
    """
    if ax is None :
        fig , ax = plt.subplots()
    data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

    #To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter( x, y, c=z, **kwargs )

    norm = Normalize(vmin = np.min(z), vmax = np.max(z))
    #cbar = fig.colorbar(cm.ScalarMappable(norm = norm), ax=ax)
    #cbar.ax.set_ylabel('Density')

    return fig, ax

def log_like_scatter(s, log_likes):
    fig, ax = density_scatter(s[:,0], log_likes, cmap=ListedColormap(plot_style_utils.SEQ_PALETTE))
#     auto_set_xy_lims(s, ax, set_x_only=True)
    
#     xvals = plt.xlim()
#     mask = np.logical_and(s[:,0] > xvals[0], s[:,0] < xvals[1])
#     pct = np.percentile(log_likes,[1,99])
#     plt.ylim(pct)
    
    
    print('Pearson R', scipy.stats.pearsonr(log_likes, s[:,0])[0])
    print('Spearman rho', scipy.stats.spearmanr(log_likes, s[:,0])[0])
    
    return fig


def generate_pc_info_for_rep(s2r, rseqs, refseqs):
    np.random.seed(1)
    random.seed(1)
    s, dr, gr, log_likes = compute_data_for_low_dim_manifold_viz(
        rseqs,
        refseqs,
        lambda x: generate_reps(x, s2r, return_loglikes=True))

    s[:,0] = -s[:,0]
    
    return {
        's': s,
        'dr': dr,
        'gr': gr,
        'log_likes': log_likes
    }


def plot_single_variant(fig, seq, seq_name, gr, dr, color=[0, 1, 0], 
        markerfacecolor="None", markersize=10):
    
    s = dr.transform(gr([seq])[0].reshape((1,-1))) ##
    
    fig.gca().plot(-s[0,0], s[0,1], 'o', color=color, 
            markerfacecolor=markerfacecolor, markersize=markersize)
    
def grab_top_designs_for_model(design_df, model, prot, n=10):
    if prot == 'GFP':
        top_design_df = design_df[design_df['model'] == model].nlargest(n, 'qfunc')
    elif prot == 'BLAC':
        top_design_df = design_df[design_df['model'] == model]
        top_design_df = top_design_df[top_design_df['n_mut_rel_wt'] == 1].nlargest(n, 'lfe_1000')
        
        if top_design_df.shape[0] < 10:
            top_design_df = design_df[design_df['model'] == 'ET_Global_Init_1']
            top_design_df = top_design_df[top_design_df['n_mut_rel_wt'] == 1].nlargest(n, 'lfe_1000')
        
    return top_design_df