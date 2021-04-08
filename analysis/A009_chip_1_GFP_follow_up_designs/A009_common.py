import os
import sys
import pickle

from Bio import SeqIO
import numpy as np
import pandas as pd

sys.path.append('../common')
import data_io_utils
import paths
import constants
import utils

# For manual flow data
SFGFP_WELL = 'E11'
AVGFP_WELL = 'F11'

def fasta_read(fasta_file):
    headers = []
    seqs = []
    for seq_record in SeqIO.parse(fasta_file, 'fasta'):
        headers.append(seq_record.id)
        seqs.append(str(seq_record.seq))
    
    return headers, seqs