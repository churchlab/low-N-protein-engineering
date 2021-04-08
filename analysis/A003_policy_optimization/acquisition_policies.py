import os
import sys
import warnings
import multiprocessing as mp
import random
from io import StringIO

import numpy as np
import pandas as pd
import copy

sys.path.append('../common')
import constants


class RandomAcquisition(object):
    
    def __init__(self):
        pass
    
    def acquire_points(self, training_set_df, n_training_points, params):
        return training_set_df.sample(n=n_training_points)
    
    
def filter_for_ns_mutants(training_set_df, wt, ns):
    # return training_set_df only including sequence-function pairs within ns mutations away from wt seq
    # wt is str
    # ns is an array
    return training_set_df.loc[training_set_df.seq.map(lambda seq: count_mutations(wt,seq)).isin(ns)]
    
    
class Random_Singles_Acquisition(object):
    
    # sample random single mutants
    
    def __init__(self):
        pass
    
    def acquire_points(self, training_set_df, n_training_points, params):
        
        training_set_df_singles = filter_for_ns_mutants(training_set_df, wt=avGFP, ns=[1])
        
        return training_set_df_singles.sample(n=n_training_points)
    
###
# Structural acquisition policies utilities and classes:
###

def count_mutations(wt, mut_seq):
    # identify mutations in same length wt and seq2
    assert len(wt) == len(mut_seq)
    n_mut = 0
    for i,letter in enumerate(list(mut_seq)):
        if letter != wt[i]:
            n_mut+=1
    return n_mut



avGFP = constants.AVGFP_AA_SEQ # verified it was the same as the seq GK put here.

# verified this is the same as the copy paste GK put here.
sfGFP_seq_annotation = pd.read_csv('sfgfp_residues_annotated_with_secondary_structure.csv')
sfGFP_seq_annotation.set_index('pos', inplace=True)

fullsfGFP_seq_annotation = pd.Series(index=np.arange(0,238))

fullsfGFP_seq_annotation.loc[sfGFP_seq_annotation.index] = sfGFP_seq_annotation.SS_updated

fullsfGFP_seq_annotation.loc[
    np.setdiff1d(np.arange(0,238),sfGFP_seq_annotation.index.values)
] = "-"

def find_mutation_positions_and_annotate(wt, mut_seq):
    # identify mutations in same length wt and seq2
    assert len(wt) == len(mut_seq)
    pos_mut = []
    annot_mut = []
    for i,letter in enumerate(list(mut_seq)):
        if letter != wt[i]:
            pos_mut.append(i)
            annot_mut.append(fullsfGFP_seq_annotation.loc[i])
    return pos_mut, annot_mut


def filter_for_position_annotation(training_set_df, wt, allowed_annotations):
    return training_set_df.loc[training_set_df.seq.map(
        lambda seq: len(np.intersect1d(
            find_mutation_positions_and_annotate(wt,seq)[1], 
            allowed_annotations
        )) != 0 # finding the annotations of all mutations, checking if there is ANY overlap with allowed_annotations
    )]

def get_subset_sizes(n_training_points, n_splits):
    
    # get sizes of n_splits approximately equal splits of n_training_points
    
    if n_training_points % n_splits == 0:
            partial_n_training_pts = partial_n_training_pts_last = np.int(n_training_points / n_splits)
    else:
        partial_n_training_pts = np.round(n_training_points / n_splits)
        partial_n_training_pts_last = n_training_points - partial_n_training_pts* (n_splits-1)
        
    sizes = [partial_n_training_pts for i in range(n_splits - 1)] + [partial_n_training_pts_last]
    
    assert partial_n_training_pts % 1 == 0.0
    
    assert partial_n_training_pts_last % 1 == 0.0
    
    assert np.sum(sizes) == n_training_points
    
    return sizes

class Structural_12_Balanced_Acquisition(object):
    
    # out of the n_training points - sample 1/3 inner barrel position variants, 1/3 outer barrel and 1/3 loop.
    
    def __init__(self):
        pass
    
    def acquire_points(self, training_set_df, n_training_points, params):
        
        n_splits = 3
        
        training_set_df = filter_for_ns_mutants(training_set_df, avGFP, [1,2])
        
        tdfs = [filter_for_position_annotation(training_set_df, avGFP, ["E-"]),
            filter_for_position_annotation(training_set_df, avGFP, ["E+"]),
            filter_for_position_annotation(training_set_df, avGFP, ["S"])]
        ns = get_subset_sizes(n_training_points, n_splits)
        
        sampled_tdfs = []
        for i,tdf in enumerate(tdfs):
            replace = ns[i] > tdf.shape[0]
            sampled_tdfs.append(tdf.sample(n=int(ns[i]), replace=replace))
        
        return pd.concat(sampled_tdfs)
    
###
# Contact map-based acquisition policies
###   

def find_mutation_positions_and_aas(wt, mut_seq):
    # identify mutations in same length wt and seq2
    assert len(wt) == len(mut_seq)
    pos_mut = []
    aas_mut = []
    for i,letter in enumerate(list(mut_seq)):
        if letter != wt[i]:
            pos_mut.append(i)
            aas_mut.append(mut_seq[i])
    return pos_mut, aas_mut

gfp_contact_map = np.load("./1GFL_contact_map.npy")

non_trivial_cutoff = 5

class StructProximity_12_Acquisition(object):
    
    # sampling special doubles (1/3) and single variants in positions where the first (1/3) and second (1/3) mutations are in the double
    # special doubles are 1) far apart (5 or more aminoacids in between) 2) close in contact map (less than 9 A)
    
    def __init__(self):
        pass
    
    def acquire_points(self, training_set_df, n_training_points, params):

        training_set_df_doubles = filter_for_ns_mutants(training_set_df, wt=avGFP, ns=[2])

        training_set_df_singles = filter_for_ns_mutants(training_set_df, wt=avGFP, ns=[1])

        non_trivial_mask = training_set_df_doubles.seq.map(lambda mut_seq: find_mutation_positions_and_aas(avGFP, mut_seq)[0]).map(
            lambda x: np.abs(x[0] - x[1])
        ) >= non_trivial_cutoff 

        training_set_df_doubles = training_set_df_doubles.loc[non_trivial_mask]

        training_set_df_singles['pos'] = training_set_df_singles.seq.map(lambda mut_seq: find_mutation_positions_and_aas(avGFP, mut_seq)[0][0])
        training_set_df_singles['aa'] = training_set_df_singles.seq.map(lambda mut_seq: find_mutation_positions_and_aas(avGFP, mut_seq)[1][0])

        pairs = training_set_df_doubles.seq.map(lambda mut_seq: find_mutation_positions_and_aas(avGFP, mut_seq))

        matched_both = []

        for i in pairs.index:

            first_single = training_set_df_singles[
                (training_set_df_singles.pos == pairs[i][0][0])] #& (training_set_df_singles.aa == pairs[i][1][0])

            second_single = training_set_df_singles[
                (training_set_df_singles.pos == pairs[i][0][1])] # & (training_set_df_singles.aa == pairs[i][1][1])

            if (pairs[i][0][0] < 230) & (pairs[i][0][1] < 230):

                if (first_single.shape[0] >= 1) & (second_single.shape[0] >= 1):
                    matched_both.append(i)

        #print(f"matched {len(matched_both)} doubles out of {pairs.shape[0]} with corresponding singles")

        training_set_df_doubles = training_set_df_doubles.loc[matched_both]

        training_set_df_doubles['Angstrom_dist'] = training_set_df_doubles.seq.map(
            lambda mut_seq: gfp_contact_map[find_mutation_positions_and_aas(avGFP, mut_seq)[0][0],find_mutation_positions_and_aas(avGFP, mut_seq)[0][1]]
        )

        ns = get_subset_sizes(n_training_points, 3)

        #print(ns)

        # will add this throw ValueError if there is not enough points to sample
        doubles_sample = training_set_df_doubles.loc[training_set_df_doubles['Angstrom_dist'] < 9].sample(np.int(ns[0]))[['seq','quantitative_function']]


        pairs = doubles_sample.seq.map(lambda mut_seq: find_mutation_positions_and_aas(avGFP, mut_seq))

        singles_1 = []

        singles_2 = []

        for i in pairs.index:

            singles_1.append(
                training_set_df_singles[
                    (training_set_df_singles.pos == pairs[i][0][0])
                ].iloc[0]
            ) 

            singles_2.append(
                training_set_df_singles[
                    (training_set_df_singles.pos == pairs[i][0][1])
                ].iloc[0]
            )  

        singles_1 = pd.concat(singles_1, axis=1).T[['seq', 'quantitative_function']].sample(np.int(ns[1]))
        singles_2 = pd.concat(singles_2, axis=1).T[['seq', 'quantitative_function']].sample(np.int(ns[2]))

        return pd.concat([doubles_sample, singles_1, singles_2])
    
    
gfp_contact_map_without_diag = copy.deepcopy(gfp_contact_map)

for i in range(230):
    for j in range(230):
        if np.abs(i-j) <= 5:
            gfp_contact_map_without_diag[i,j] = 10000 # this is setting the distances on the diagonal very high so that they get filtered out
            
num_ineract = pd.Series(
    np.apply_along_axis(lambda x: np.count_nonzero(pd.Series(x) < 9), axis=1, arr=gfp_contact_map_without_diag)
)

class Num_interact_Positive_Acquisition(object):
    
    # Sample single mutants in proportion to how many non-trivial (i.e. more than 4 AA away in sequence) residues they are in contact with     # (< 9 A away from)
    
    def __init__(self):
        pass
    
    def acquire_points(self, training_set_df, n_training_points, params):
        
        training_set_df_singles = filter_for_ns_mutants(training_set_df, wt=avGFP, ns=[1])
        
        positions =  training_set_df_singles.seq.map(
                lambda mut_seq: find_mutation_positions_and_aas(avGFP, mut_seq)[0][0]
        )
        
        positions = positions[positions < 230]
        
        weights = positions.map(lambda x: num_ineract.loc[x])
        
        weights = weights / weights.max()

        return  training_set_df_singles.loc[positions.index].sample(
            n=n_training_points,
            weights = weights
        )

class Num_interact_Balanced_Acquisition(object):
    
    # Sample 10% high-contact residues (top 25% in number of far away residues within 9 A)
    # Sample 10% low-contact residues (bottom 25% in number of far away residues within 9 A)
    # Sample the rest randomly
    
    def __init__(self):
        pass
    
    def acquire_points(self, training_set_df, n_training_points, params):
        
        training_set_df_singles = filter_for_ns_mutants(training_set_df, wt=avGFP, ns=[1])
        
        positions =  training_set_df_singles.seq.map(
                lambda mut_seq: find_mutation_positions_and_aas(avGFP, mut_seq)[0][0]
        )
        
        positions = positions[positions < 230]
        
        weights = positions.map(lambda x: num_ineract.loc[x])
        
        return pd.concat([
            training_set_df_singles.loc[
                    weights[weights > weights.describe()['75%']].sample(np.int(np.round(n_training_points * 0.1))).index
                ],
            training_set_df_singles.loc[
                    weights[weights < weights.describe()['25%']].sample(np.int(np.round(n_training_points * 0.1))).index
                ],
            training_set_df_singles.loc[positions.index].sample(
                    n=n_training_points - np.int(np.round(n_training_points * 0.1))*2)
        ])

###
# Evolutionary acquisition policies utilities and classes:
###

    
gfp_pssm = pd.read_csv("./gfp_full_pssm.txt")

gfp_pssm = gfp_pssm.iloc[0:21, 0].map(lambda x: [np.float32(i) for i in x.split()])

gfp_pssm = np.vstack(gfp_pssm)

#one dimension for each amino acid, ordered alphabetically

from constants import AA_ALPHABET_STANDARD_ORDER

#len(AA_ALPHABET_STANDARD_ORDER)

gfp_pssm = pd.DataFrame(gfp_pssm)
gfp_pssm.index = list(np.sort(list(AA_ALPHABET_STANDARD_ORDER))) + ['inf_content']

def find_mutation_positions_and_pssm(wt, mut_seq):
    # identify mutations in same length wt and seq2
    assert len(wt) == len(mut_seq)
    pos_mut = []
    annot_mut = []
    for i,letter in enumerate(list(mut_seq)):
        if letter != wt[i]:
            pos_mut.append(i)
            annot_mut.append(gfp_pssm.loc[mut_seq[i],i])
    return pos_mut, annot_mut

class PSSM_Positive_Sample_Acquisition(object):
    
    # sample according to maximum PSSM value for mutations in a variant
    
    def __init__(self):
        pass
    
    def acquire_points(self, training_set_df, n_training_points, params):
        
        weights = training_set_df.seq.map(
                lambda x: np.max(find_mutation_positions_and_pssm(avGFP, x)[1])
        )
        
        weights = weights / weights.max()

        
        return  training_set_df.sample(
            n=n_training_points,
            weights = weights
        )
    
class PSSM_Quadratic_Sample_Acquisition(object):
    
    # sample according to square of maximum PSSM value for mutations in a variant (more weight to higher PSSM value variants)
    
    def __init__(self):
        pass
    
    def acquire_points(self, training_set_df, n_training_points, params):
        
        weights = training_set_df.seq.map(
                lambda x: np.max(find_mutation_positions_and_pssm(avGFP, x)[1])
        )
        
        weights = weights / weights.max()

        
        return  training_set_df.sample(
            n=n_training_points,
            weights = weights**2
        )