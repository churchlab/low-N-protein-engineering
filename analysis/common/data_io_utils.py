"""
Utility functions for importing data.
"""

from enum import Enum
import hashlib
import json
import os
import pickle
import random
import re
import subprocess
import sys

from sklearn.utils import resample
from scipy.stats import ttest_ind
#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS_MODULE_PATH = os.path.dirname(os.path.abspath(__file__))

S3_DATA_ROOT = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '../../data', 's3')

DEFAULT_BUCKET = 'low-n-protein-engineering-public'
CMD_APPEND = '--no-sign-request'

################################################################################
# S3 Data Utility Functions
################################################################################

def generate_md5_checksum(file_path):
    """Generate checksum for the file.
    Only verify_file_md5_checksum() should use this programmatically.
    Otherwise users should use this interactively to enforce correct versions
    of files.
    """
    with open(file_path, 'rb') as fh:
        return hashlib.md5(fh.read()).hexdigest()


def verify_file_md5_checksum(file_path, expected_md5_checksum):
    """Allows verifying version is same.
    This is important since we generate data files and store them in s3 rather
    than versioning. It's possible we inadvertently overwrite files. Using
    this function correctly guarantees correcteness.
    """
    assert expected_md5_checksum == generate_md5_checksum(file_path)


def verify_md5_and_load_file(
        file_path, expected_md5_checksum, load_fn=pd.read_csv,
        load_fn_sep=',', skip_md5=False):
    """Verifies and loads file.
    NOTE: load_fn_sep only supported for load_fn=pd.read_csv.
    """
    if not skip_md5:
        verify_file_md5_checksum(
                file_path, expected_md5_checksum)

    if load_fn == pd.read_csv:
        return load_fn(file_path, sep=load_fn_sep)
    elif load_fn == pickle.load:
        with open(file_path, 'rb') as fh:
            return pickle.load(fh)
    else:
        return load_fn(file_path)


def verify_md5_and_pandas_read_csv(
        file_path, expected_md5_checksum):
    """Single function for verify a file md5 and reading csv.
    Returns pd.DataFrame.
    """
    return verify_md5_and_load_file(
            file_path, expected_md5_checksum, load_fn=pd.read_csv)


def _remove_local_root(full_local_path, local_root):
    """Removes the local root part of the full local path.
    # NOTE: Do not use strip() to do this.
    # That method doesn't exactly do what you think,
    # although sometimes it does, which makes it the devil.
    """
    assert full_local_path.find(local_root) == 0
    local_path_root_stripped = full_local_path[len(local_root):]
    local_path_root_stripped = local_path_root_stripped.lstrip('/')
    return local_path_root_stripped


def _assert_path_exists_on_s3(root_stripped_data_path, bucket=DEFAULT_BUCKET):
    """Makes sure the path is on s3.
    Path can be directory or file.
    Raises AssertionError if path doesn't exist on s3.
    """
    ls_wc_cmd = 'aws s3 ls s3://{bucket}/{path} | wc -l'.format(
            bucket=bucket,
            path=root_stripped_data_path)
    
    if len(CMD_APPEND) > 0:
        ls_wc_cmd = ls_wc_cmd.replace('aws s3 ls', 'aws s3 ls ' + CMD_APPEND)
        
    ls_wc_result = subprocess.check_output(ls_wc_cmd, shell=True)
    num_matches = int(re.search('[0-9]+', ls_wc_result.decode('utf-8')).group())
    assert num_matches > 0, (
            "Path not found on s3: %s" % ls_wc_cmd)


def sync_verify_and_load_dataset(
        file_path, expected_md5_checksum,
        load_fn=pd.read_csv, load_fn_sep=',', skip_md5=False,
        local_root=S3_DATA_ROOT):
    """Syncs the single file from s3 if not present, verifies the checksum,
    and returns as a pandas DataFrame.
    """

    sync_s3_path_to_local(
            file_path, local_root=local_root,
            is_single_file=True)

    return verify_md5_and_load_file(
            file_path, expected_md5_checksum,
            load_fn=load_fn, load_fn_sep=load_fn_sep,
            skip_md5=skip_md5)



def _core_s3_sync_command(local_path, remote_path, 
        direction, file_name=None):
    """Defines the core s3 sync command without optional flags
    
        Args:
            local_path: local path
            remote_path: remote path
            direction: A string ["local_to_remote" | "remote_to_local"] specifying
                the direction of the sync.
    """
    
    if direction == "local_to_remote":
        core_s3_sync_cmd = (
                'aws s3 sync '
                '{from_dir_path} '
                '{to_dir_path}'.format(
                        from_dir_path=local_path,
                        to_dir_path=remote_path))    
        
    elif direction == "remote_to_local":
        core_s3_sync_cmd = (
                'aws s3 sync '
                '{from_dir_path} '
                '{to_dir_path}'.format(
                        from_dir_path=remote_path,
                        to_dir_path=local_path))
        
    else:
        raise ValueError('direction must be "remote_to_local" '
                         'or "local_to_remote"')
    
    if file_name is not None:
        core_s3_sync_cmd = (core_s3_sync_cmd + ' '
                '--exclude="*" '
                '--include="{file_name}"'.format(file_name=file_name))
        
    if len(CMD_APPEND) > 0:
        core_s3_sync_cmd = core_s3_sync_cmd + ' ' + CMD_APPEND
        
    return core_s3_sync_cmd
    
def _build_full_s3_path(local_root_stripped_path, bucket=DEFAULT_BUCKET):
    return 's3://' + bucket + '/' + local_root_stripped_path
    
def sync_s3_path_to_local(
        full_local_path, local_root=S3_DATA_ROOT,
        is_single_file=False, additional_flags=None,
        verbose=False, bucket=DEFAULT_BUCKET):
    """Syncs a path (directory+contents or file) from s3 to the local machine,
    mirroring the full directory structure.
    Args:
        full_local_path: The full path to where the dir or file is locally.
            This will include the local_root where s3 data is stored.
        local_root: The local_root, which this function uses to know what
            part to strip off before querying s3.
        is_single_file: Must be set to True to sync single file.
        additional_flags: A list of strings. Specifies additional flags and
            their value (e.g. additional_flags=["--exclude */tensorboard/*"])
        verbose: print output of s3 sync command?
    Raises AssertionError if path doesn't exist on s3.
    """
    s3_sync(full_local_path, "remote_to_local",
            local_root=local_root, 
            is_single_file=is_single_file, 
            additional_flags=additional_flags,
            verbose=verbose,
            bucket=bucket)


def sync_local_path_to_s3(
        full_local_path, local_root=S3_DATA_ROOT,
        is_single_file=False, additional_flags=None,
        verbose=False, bucket=DEFAULT_BUCKET):
    """Syncs a path (directory+contents or file) from the local machiene to s3,
    mirroring the full directory structure.
    NOTE: s3 won't show an empty directory.
    Args:
        full_local_path: The full path to where the file would be locally.
            This will include the local_root where s3 data is stored.
        local_root: The local_root, which this function uses to know what
            part to strip off before querying s3.
        is_single_file: Must be set to True to sync single file.
        additional_flags: A list of strings. Specifies additional flags and
            their value (e.g. additional_flags=["--exclude */tensorboard/*"])
        verbose: print output of s3 sync command?
    Raises AssertionError if full_local_path doesn't exist locally.
    """
    s3_sync(full_local_path, "local_to_remote",
            local_root=local_root, 
            is_single_file=is_single_file, 
            additional_flags=additional_flags,
            verbose=verbose, 
            bucket=bucket)
    
def path_exists_on_s3(full_local_path, local_root=S3_DATA_ROOT, bucket=DEFAULT_BUCKET):
    
    root_stripped_data_path = _remove_local_root(
            full_local_path, local_root)
    
    ls_wc_cmd = 'aws s3 ls s3://{bucket}/{path} | wc -l'.format(
            bucket=bucket,
            path=root_stripped_data_path)
    
    if len(CMD_APPEND) > 0:
        ls_wc_cmd = ls_wc_cmd.replace('aws s3 ls', 'aws s3 ls ' + CMD_APPEND)
    
    ls_wc_result = subprocess.check_output(ls_wc_cmd, shell=True)
    num_matches = int(re.search('[0-9]+', ls_wc_result.decode('utf-8')).group())
    
    return num_matches > 0
    
def s3_sync(full_local_path, direction,
        local_root=S3_DATA_ROOT,
        is_single_file=False, 
        additional_flags=None,
        verbose=False,
        bucket=DEFAULT_BUCKET):
    """Syncs a path (directory+contents or file) from the local machiene to s3,
    or vice versa. Mirrors the full directory structure.
    NOTE: s3 won't show an empty directory.
    Args:
        full_local_path: The full path to where the file would be locally.
            This will include the local_root where s3 data is stored.
        direction: A string ["local_to_remote" | "remote_to_local"] specifying
            the direction of the sync.
        local_root: The local_root, which this function uses to know what
            part to strip off before querying s3.
        is_single_file: Must be set to True to sync single file.
        additional_flags: A list of strings. Specifies additional flags and
            their value (e.g. additional_flags=["--exclude */tensorboard/*"])
        verbose: print output of s3 sync command?
    Raises AssertionError if full_local_path doesn't exist locally.
    Raises AssertionError if path doesn't exist on S3.
    """

    
    if direction == "local_to_remote":
        assert os.path.exists(full_local_path), (
            "Path not found locally: %s" % full_local_path)
    elif direction == "remote_to_local":
        _assert_path_exists_on_s3(_remove_local_root(
            full_local_path, local_root), bucket=bucket)
    else:
        raise ValueError('direction must be "remote_to_local" '
                         'or "local_to_remote"')

    if is_single_file:
        # Define the local containing directory path.
        local_containing_dir_path, file_name = os.path.split(full_local_path)
        
        # Define the remote containing directory path
        root_stripped_containing_dir_path = _remove_local_root(
                local_containing_dir_path, local_root)
        remote_containing_dir_path = _build_full_s3_path( 
                root_stripped_containing_dir_path, bucket=bucket)    
        
        core_s3_sync_cmd = _core_s3_sync_command(
                    local_containing_dir_path,
                    remote_containing_dir_path,
                    direction,
                    file_name=file_name)

    else:        
        # Remove the local root part, and tack on the path to the
        # s3 bucket.
        remote_dir_path = _build_full_s3_path(
                _remove_local_root(full_local_path, local_root),
                bucket=bucket)
            
        local_dir_path = full_local_path
        
        core_s3_sync_cmd = _core_s3_sync_command(
                local_dir_path,
                remote_dir_path,
                direction)
        
    # Add additional flags (Optional arguments)
    s3_sync_cmd = core_s3_sync_cmd
    if additional_flags is not None:
        for flag_and_val in additional_flags:
            s3_sync_cmd = s3_sync_cmd + ' ' + flag_and_val
            
    # Execute the sync
    if verbose:
        print(s3_sync_cmd)
        
    output = subprocess.check_output(s3_sync_cmd, shell=True)
    
    if verbose:
        print(output.decode(sys.stdout.encoding))

