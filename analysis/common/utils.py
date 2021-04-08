import os
import sys

from Bio.Data import CodonTable
from Bio.Restriction import BsaI, BsmBI, BamHI, EcoRV
from Bio.Seq import reverse_complement
from Bio.Seq import Seq
from Bio import SeqIO 
from Bio.SeqUtils import GC 

import Levenshtein
import numpy as np
import pandas as pd

import constants

THIS_MODULE_PATH = os.path.dirname(os.path.abspath(__file__))

ECOLI_CODON_DF = pd.read_csv(os.path.join(THIS_MODULE_PATH, 'ecoli_aa_codon_df.csv'))

def fasta_read(fasta_file):
    headers = []
    seqs = []
    for seq_record in SeqIO.parse(fasta_file, 'fasta'):
        headers.append(seq_record.id)
        seqs.append(str(seq_record.seq))
    
    return headers, seqs

def levenshtein_distance_matrix(a_list, b_list=None, verbose=False):
    """Computes an len(a_list) x len(b_list) levenshtein distance
    matrix.
    """
    if b_list is None:
        single_list = True
        b_list = a_list
    else:
        single_list = False
    
    H = np.zeros(shape=(len(a_list), len(b_list)))
    for i in range(len(a_list)):
        if verbose:
            print(i)
        
        if single_list:  
            # only compute upper triangle.
            for j in range(i+1,len(b_list)):
                H[i,j] = Levenshtein.distance(a_list[i], b_list[j])
                H[j,i] = H[i,j]
        else:
            for j in range(len(b_list)):
                H[i,j] = Levenshtein.distance(a_list[i], b_list[j])

    return H

def decode_integer_array_to_seq(iarray):
    iarray = iarray.reshape(-1)
    sl = [constants.AA_ALPHABET_STANDARD_ORDER[int(i)] for i in iarray]
    return ''.join(sl)

def encode_seq_as_integer_array(seq, alphabet=constants.AA_ALPHABET_STANDARD_ORDER):
    encoding = np.zeros(len(seq))
    for i in range(len(seq)):
        encoding[i] = alphabet.find(seq[i])
        
    encoding = encoding.reshape((1,-1))
    return encoding


def encode_seq_list_as_integer_array(seq_list, alphabet=constants.AA_ALPHABET_STANDARD_ORDER):
    encoded_seqs = np.zeros((len(seq_list),len(seq_list[0])))
    for i in range(len(seq_list)):
        encoded_seqs[i] = encode_seq_as_integer_array(seq_list[i], alphabet=alphabet)
        
    return encoded_seqs


def encode_aa_seq_as_one_hot_vector(
        aa_seq, alphabet=constants.AA_ALPHABET_STANDARD_ORDER, 
        flatten=True, wildcard=None, pad_to_len=None):
    """Converts AA-Seq to one-hot encoding, setting exactly one 1 at every
    amino acid position.
    Returns:
        If flatten is True, return boolean np.array of length
            len(alphabet) * len(aa_seq). Else a matrix with
            dimensions (len(alphabet), len(aa_seq)).
    """
    # Optimization: Constant-time lookup. Empirically saves 30% compute time.
    alphabet_aa_to_index_dict = {}
    for i in range(len(alphabet)):
        alphabet_aa_to_index_dict[alphabet[i]] = i

    # Build as matrix.
    ohe_matrix = np.zeros((len(alphabet), len(aa_seq)), dtype=np.float32)
    for pos, aa in enumerate(aa_seq):
        if wildcard is not None:
            if aa == wildcard:
                ohe_matrix[:, pos] = 1/len(alphabet)
            else:
                ohe_matrix[alphabet_aa_to_index_dict[aa], pos] = 1
        else:
            ohe_matrix[alphabet_aa_to_index_dict[aa], pos] = 1
            
    if pad_to_len is not None:
        assert ohe_matrix.shape[1] <= pad_to_len
        
        npad = pad_to_len - ohe_matrix.shape[1]
        pad_mat = np.zeros((len(alphabet), npad))
        ohe_matrix = np.hstack((ohe_matrix, pad_mat))

    # Return flattened or matrix.
    if flatten:
        return ohe_matrix.reshape(-1, order='F')
    else:
        return ohe_matrix
    
### ONE HOT ENCODING FUNCTIONS FOR PUBLIC ###
def encode_aa_seq_as_one_hot(
        aa_seq, alphabet=constants.AA_ALPHABET_STANDARD_ORDER, flatten=True, 
        wildcard='X', pad_to_len=None):
    
    return encode_aa_seq_as_one_hot_vector(
            aa_seq, alphabet=alphabet, flatten=flatten, 
            wildcard=wildcard, pad_to_len=pad_to_len)

def encode_aa_seq_list_as_matrix_of_flattened_one_hots(
        aa_seq_list, alphabet=constants.AA_ALPHABET_STANDARD_ORDER,
        wildcard='X', pad_to_len=None):
    
    enc_seqs = [
        encode_aa_seq_as_one_hot(s, alphabet=alphabet, flatten=True,
                wildcard=wildcard, pad_to_len=pad_to_len).reshape((1,-1))
        for s in aa_seq_list
    ]
    
    return np.vstack(enc_seqs)
    
def encode_nt_seq_as_one_hot(
        nt_seq, alphabet=constants.NT_ALPHABET_STANDARD_ORDER, flatten=True, 
        wildcard='N'):
    
    return encode_aa_seq_as_one_hot_vector(
            nt_seq, alphabet=alphabet, flatten=flatten, 
            wildcard=wildcard)    


def aa_to_dna(
        wt_seq_aa, wt_seq_dna, mutant_seq_aa, codon_table_df=ECOLI_CODON_DF,
        forbidden_restriction_list=['XXXXXXXXXXXX']):
    """Takes an aa sequence, and returns the dna sequence. Uses the wild-type codon
    when possible.
    For mutated AA, if in first ten codons use low-GC (Goodman, 2013).
    Otherwise go by highest codon usage in E. coli.
    Args:
        wt_seq_aa: this is the wt aa sequence to compare to
        wt_dna_seq: will use these codons if the aa is the same
        mutant_seq_aa: this is the seq to convert to
        codon_table_df: this has the codons, their gc content, and e coli usage
        forbidden_restriciton_list: a list of strings of restriction sites to
            avoid when generating dna_seqs
    Returns:
        mutant_dna: a str of the dna sequence which is "codon optimized"
            (low gc | common & no res site)
    Raises:
        NoValidCodonFoundException if no codon swap can be found satisfying
        all contraints.
    """
    # Validate input.
    assert len(wt_seq_aa) == len(mutant_seq_aa)
    all([isinstance(x, str) for x in forbidden_restriction_list])

    forbidden_restriction_set_with_rc = set()
    for res_site in forbidden_restriction_list:
        forbidden_restriction_set_with_rc.add(res_site)
        forbidden_restriction_set_with_rc.add(reverse_complement(res_site))

    mutant_dna = ''

    for i in range(len(wt_seq_aa)):
        codon_start = i * 3 # i iterates through aa space, make a codon indexer
        codon_end = (i*3) + 3

        # If AA unchanged from wild-type, use same codon.
        wt_aa = wt_seq_aa[i]
        mut_aa = mutant_seq_aa[i]
        if wt_aa == mut_aa:
            wt_codon = wt_seq_dna[codon_start:codon_end]
            mutant_dna += wt_codon
            continue

        # Else AA changed. If in first ten codons use low-GC (Goodman, 2013).
        # Otherwise go by highest codon usage in E. coli.
        # If aa is 1-10, use low gc, else use most frequent from E. coli.
        if i <= 10:
            sort_by_key = 'gc'
            sort_ascending = True
        else:
            sort_by_key = 'usage'
            sort_ascending = False

        # Sort codons as determined.
        sorted_codon_options = (
                codon_table_df[codon_table_df['aa'] == mut_aa].sort_values(
                        by=sort_by_key, ascending=sort_ascending)['codon'])

        # Identify a codon that doesn't introduce unwanted restriction sites.
        codon_iter = 0
        is_valid_codon_choice = False
        while not is_valid_codon_choice:
            if codon_iter >= len(sorted_codon_options):
                raise NoValidCodonFoundException
            mut_codon = sorted_codon_options.iloc[codon_iter]
            mutant_test = mutant_dna + mut_codon
            is_valid_codon_choice = not any(
                    x in mutant_test
                    for x in forbidden_restriction_set_with_rc)
            codon_iter += 1
        mutant_dna += mut_codon

    assert len(mutant_dna) == 3 * len(mutant_seq_aa)
    return str(mutant_dna)

def build_edit_string(seq, wt_seq):
    """Builds a 1-indexed edit string btw seq and wt_seq where there are assumed
    to be no indels.
    This function doesn't use nwalign and instead does a char by char comparison.
    Returns a String consisting of:
        * An edit string of the form e.g. 'SA8G:SR97W'
        * 'WT', if alignment is perfect match.
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