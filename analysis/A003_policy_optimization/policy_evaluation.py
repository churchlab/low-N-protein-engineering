import os
import sys
import warnings
import multiprocessing as mp
import copy
import random
import pickle

import numpy as np
import pandas as pd
import scipy

sys.path.append('../common')
import data_io_utils
import paths
import utils
import constants

import A003_common
import acquisition_policies
import models

def evaluate_model_and_acquisition_policy(
        training_set_df, 
        acq_policy_obj,
        n_training_points_schedule,
        acq_policy_params,
        model,
        generalization_set_dfs,
        generalization_set_names,
        generalization_set_sub_category_columns,
        generalization_set_calc_params,
        output_dir,
        force=False,
        verbose=False,
        save_sampled_training_set_dfs=False,
        save_models=False):
    """
    
    Run retrospective testing of a model and acquistion policy over a schedule
    of training points
    
    Parameters
    ----------    
    training_set_df : pandas DataFrame 
        must have 2 columns - 'sequence' with unencoded amino acid sequences 
        and 'quantitative_function' with quantitative function values and can 
        have additional metadata columns
    acq_policy_obj : Object
        Object that implements acq_policy_obj.acquire_points(...). Has the
        following interface: 
        
            acquire_points(training_set_df, n_training_points, acq_policy_params)
        
        Given training_set_df and acq_policy_params, returns a DataFrame that
        is a subset of training_set_df representing training points.             
    n_training_points_schedule : list of ints  
        Schedule of number of training points to use when evaluating the 
        acquisition policy and model.
    acq_policy_params : dict
        Dictionary of parameters that the acq_policy_obj needs.
    model : object
        An object from a user defined class. Required to implement a train and
        predict method. Their interfaces are as follows:
        
            model.train(aa_seqs, qfunc)
            model.predict(aa_seqs)
            
            aa_seqs are a list of amino acid sequences (strings) and qfunc is a
            one-dimensional numpy array of quantitative function values.
    generalization_set_dfs : A list of pandas DataFrames
        Each element of this list is a generalization set DataFrame that has at
        least a 'seq' and a 'quantitative_function' column. It may also have a
        specific column with categorical information (e.g. neighborhood ids)
        that can be used to subset the dataframe. Useful for calculating
        generalization stats on e.g. neighborhood subsets. See below.
    generalization_set_names : A list of strings
        Names for each generalization set.
    generalization_set_sub_category_columns : A list of strings
        A list of column names that specify categorical columns of each DataFrame
        in generalization_set_dfs. These categorical columns could specify the
        neighborhood each variant belongs, enabling sub-categorized generalization
        statistics. Note generalization_set_sub_category_columns[i] corresponds
        to generalization_set_dfs[i]
    generalization_set_calc_params : Dictionary
        Dictionary of parameters needed for calculating
        generalization statistics. Leaving this as a dictionary
        so it can organically grow as we add functionality.
        Required fields:
            'quantitative_function_threshold': Specifies threshold for
                binarizing actual and predicted quantitative function.
            'n_to_test_curve': Schedule of "N" to use for evaluating 
                generalization prioritization ("efficiency over random
                curves").
    output_dir : Path to output directory
    force : bool. Rerun pipeline regardless of saved progress
    verbose : bool. Print progress to stdout
    save_sampled_training_set_dfs: bool. Save dataframe of acquired training
        points to disk?
    save_models: bool. Save models used to train on acquired training points?
        
   
    Returns
    -------
    generalization_results : pandas DataFrame 
        index: sub-categories (e.g. neighborhoods) and 'overall'
        columns: Generalization metrics that include both continous
            "regression" metrics (e.g. MSE, Pearson R) and binary 
            "classification" metrics (e.g. FDR).
    """
    
    # Set up output/checkpoint files
    if 'get_name' in dir(model):
        model_name = model.get_name()
    else:
        model_name = model.__class__.__name__
    
    gen_set_name_str = '_'.join(generalization_set_names)
    output_file_name = (
        'generalization_results-%s-%s-%s.pkl') % (
        model_name, 
        acq_policy_obj.__class__.__name__,
        gen_set_name_str)

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, output_file_name)
    output_checkpoint_file = output_file + '.p'
    
    # Restore from checkpoint unless force
    # First check if we've already computed this before
    if not force and os.path.isfile(output_checkpoint_file):
        with open(output_checkpoint_file, 'rb') as f:
            gen_results_dfs = pickle.load(f)
        
        # Look at what n_training points we've done so far
        n_examined_so_far = []
        for gdf in gen_results_dfs:
            n_examined_so_far += list(np.unique(gdf['n_train']))

        n_examined_so_far = list(np.unique(n_examined_so_far))
        
        # Remove those from the schedule.
        n_training_points_schedule = [n for n in n_training_points_schedule 
                if n not in n_examined_so_far]
        
        if verbose:
            print('Already computed for n_training_points =', n_examined_so_far)
            print('n_training_points left to examine =', n_training_points_schedule)
    else:
        if verbose:
            print('Found no existing progress or being forced.')
        gen_results_dfs = []
        
    # Main computation loop
    for i,n_training_points in enumerate(n_training_points_schedule):
        
        #try:
        if verbose:
            print('n_train:', n_training_points)

        # Acquire training points
        if verbose:
            print('\tAcquiring training points')
        training_sample_df = acq_policy_obj.acquire_points(
                training_set_df, n_training_points, acq_policy_params)

        # model is an object with two methods `train` and `predict`.
        # Train model.
        if verbose:
            print('\tTraining/tuning model')
        aa_seqs = list(training_sample_df['seq'])
        qfunc = np.array(training_sample_df['quantitative_function'])
        model.train(aa_seqs, qfunc)

        # Evaluate generalization of the trained model.
        if verbose:
            print('\tEvaluating generalization')
        g_results_df = evaluate_generalization(
                model, 
                generalization_set_dfs,
                generalization_set_names,
                generalization_set_sub_category_columns,
                generalization_set_calc_params
            )

        g_results_df['n_train'] = n_training_points
        gen_results_dfs.append(g_results_df)
            
        #except Exception as e:
         #   print(repr(e))
        
        # Save progress
        if verbose:
            print('\tCheckpointing')
            
        with open(output_checkpoint_file, 'wb') as f:
            pickle.dump(obj=gen_results_dfs, file=f)
            
        if save_sampled_training_set_dfs:
            training_sample_outfile = os.path.join(output_dir, 
                    'acquired_training_points_%d.csv' % n_training_points)
            training_sample_df.to_csv(training_sample_outfile, index=False)
            
        if save_models:
            model_save_output_file = os.path.join(output_dir, 
                    'model_%d.p' % n_training_points)
            model.save(model_save_output_file)
    
    generalization_results_df = pd.concat(gen_results_dfs)
    generalization_results_df.to_pickle(output_file)

    return {
        'gen_results_df': generalization_results_df,
        'output_file': output_file
    }

def parallel_replicate_evaluate_model_and_acquisition_policy(
        n_replicates,
        n_processes,
        training_set_df, 
        acq_policy_obj,
        n_training_points_schedule,
        acq_policy_params,
        model,
        generalization_set_dfs,
        generalization_set_names,
        generalization_set_sub_category_columns,
        generalization_set_calc_params,
        output_dir,
        force=False,
        verbose=False):
    
    """
    Same interface as evaluate_model_and_acquisition_policy
    but has two additional arguments for n_replicates and
    n_processes.
    
    Results are saved in output_dir/rep_<replicate_number>
    """
    
    assert False, 'DEPRECATED'
        
    # Define worker tasks
    tasks = []
    for i in range(n_replicates):
        # Shuffle the n_training_points schedule
        # for better load balancing.
        ntp_sched = copy.deepcopy(n_training_points_schedule)
        random.shuffle(ntp_sched)
        
        rep_output_dir = os.path.join(output_dir, 'rep_'+str(i))

        tasks.append(
            (
                training_set_df,
                acq_policy_obj,
                ntp_sched,
                acq_policy_params,
                model,
                generalization_set_dfs,
                generalization_set_names,
                generalization_set_sub_category_columns,
                generalization_set_calc_params,
                rep_output_dir,
                force,
                verbose
            )
        )

    # Beast mode
    with mp.Pool(processes=n_processes) as pool:
        results = pool.starmap(
            evaluate_model_and_acquisition_policy, 
            tasks)
        
    # Aggregate results
    result_dfs = []
    for i in range(len(results)):
        df = results[i]['gen_results_df']
        df['rep'] = i
        result_dfs.append(df)
        
    master_result_df = pd.concat(result_dfs)
    
    # Output to disk if requested
    if output_dir is not None:
        gen_set_name_str = '_'.join(generalization_set_names)
        output_file_name = (
            'generalization_results-%s-%s-%s-%s.pkl') % (
            model.__class__.__name__, 
            acq_policy_obj.__class__.__name__,
            gen_set_name_str,
            'nrep_' + str(n_replicates))
    
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, output_file_name)
        master_result_df.to_pickle(output_file)
    else:
        output_file = None
        
    return {
        'gen_results_df': master_result_df,
        'output_file': output_file
    }
            
############################################################################
### INPUT BUILDING
############################################################################
            
# One master function to set up inputs for a data efficiency experiment.
# Minimizes how much fiddling we're doing inside notebooks, which will have
# copy paste issues.
N_TRAINING_POINTS_SCHEDULE = np.array(
    [8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 144, 192, 384, 768, 1536, 3072])
N_TESTING_POINTS_SCHEDULE = copy.deepcopy(N_TRAINING_POINTS_SCHEDULE)

def load_data_eff_inputs(split, training_set_name, acq_policy, model, experiment_name=''):
    
    ## Some constants
    n_training_points_schedule = N_TRAINING_POINTS_SCHEDULE
    
    ## Training set set up.
    data_io_utils.sync_s3_path_to_local(paths.DATA_DISTRIBUTIONS_DIR)
    if training_set_name == 'sarkisyan':
        leval_train = 'paths.SARKISYAN_SPLIT_%d_FILE' % split
    elif training_set_name == 'fp_homologs':
        leval_train = 'paths.FP_HOMOLOGS_DATA_DIST_SPLIT_%d_FILE' % split
    training_set_df = pd.read_csv(eval(leval_train))
    
    ## Acquisition policy set up.
    if acq_policy == 'random':
        acquisition_policy_obj = acquisition_policies.RandomAcquisition()
        acquisition_policy_params = {}
    elif acq_policy == 'struct_12_balanced':
        acquisition_policy_obj = acquisition_policies.Structural_12_Balanced_Acquisition()
        acquisition_policy_params = {}
    elif acq_policy == 'pssm_positive':
        acquisition_policy_obj = acquisition_policies.PSSM_Positive_Sample_Acquisition()
        acquisition_policy_params = {}
    elif acq_policy == 'pssm_quadratic':
        acquisition_policy_obj = acquisition_policies.PSSM_Quadratic_Sample_Acquisition()
        acquisition_policy_params = {}
        
        
    elif acq_policy == 'random_singles':
        acquisition_policy_obj = acquisition_policies.Random_Singles_Acquisition()
        acquisition_policy_params = {}
    elif acq_policy == 'struct_proximity_12_thirds':
        acquisition_policy_obj = acquisition_policies.StructProximity_12_Acquisition()
        acquisition_policy_params = {}
    elif acq_policy == 'num_interact_positive':
        acquisition_policy_obj = acquisition_policies.Num_interact_Positive_Acquisition()
        acquisition_policy_params = {}
    elif acq_policy == 'num_interact_balanced':
        acquisition_policy_obj = acquisition_policies.Num_interact_Balanced_Acquisition()
        acquisition_policy_params = {}
           
    ## Model set up
    if model == 'LassoLars':
        model_obj = models.LassoLarsModel()
    elif model == 'Ridge':
        model_obj = models.RidgeModel(do_sparse_refit=False)
    elif model == 'RidgeSparseRefit':
        model_obj = models.RidgeModel(do_sparse_refit=True)
    elif model == 'EnsembledRidgeSparseRefit':
        model_obj = models.EnsembledRidgeModel(do_sparse_refit=True)
        
    elif model == 'Doc2VecLassoLars':
        model_obj = models.Doc2VecLassoLarsModel()
    elif model == 'Doc2VecRidge':
        model_obj = models.Doc2VecRidgeModel(do_sparse_refit=False)
    elif model == 'Doc2VecRidgeSparseRefit':
        model_obj = models.Doc2VecRidgeModel(do_sparse_refit=True)
    elif model == 'Doc2VecEnsembledRidgeSparseRefit':
        model_obj = models.Doc2VecEnsembledRidgeModel(do_sparse_refit=True)
        
    elif model == 'UniRepLassoLars':
        model_obj = models.UniRepLassoLarsModel(do_sparse_refit=False, rep='global') 
    elif model == 'UniRepRidge':
        model_obj = models.UniRepRidgeModel(do_sparse_refit=False, rep='global') 
    elif model == 'UniRepRidgeSparseRefit':
        model_obj = models.UniRepRidgeModel(do_sparse_refit=True, rep='global')  
    elif model == 'UniRepEnsembledRidgeSparseRefit':
        model_obj = models.UniRepEnsembledRidgeModel(do_sparse_refit=True, rep='global')
        
    elif model == 'EvotunedUniRep_Random_Init_1_LassoLars':
        model_obj = models.UniRepLassoLarsModel(do_sparse_refit=False, rep='et_random_init_1') 
    elif model == 'EvotunedUniRep_Random_Init_1_Ridge':
        model_obj = models.UniRepRidgeModel(do_sparse_refit=False, rep='et_random_init_1') 
    elif model == 'EvotunedUniRep_Random_Init_1_RidgeSparseRefit':
        model_obj = models.UniRepRidgeModel(do_sparse_refit=True, rep='et_random_init_1')  
    elif model == 'EvotunedUniRep_Random_Init_1_EnsembledRidgeSparseRefit':
        model_obj = models.UniRepEnsembledRidgeModel(do_sparse_refit=True, rep='et_random_init_1')
      
    elif model == 'EvotunedUniRep_Global_Init_1_LassoLars':
        model_obj = models.UniRepLassoLarsModel(do_sparse_refit=False, rep='et_global_init_1') 
    elif model == 'EvotunedUniRep_Global_Init_1_Ridge':
        model_obj = models.UniRepRidgeModel(do_sparse_refit=False, rep='et_global_init_1') 
    elif model == 'EvotunedUniRep_Global_Init_1_RidgeSparseRefit':
        model_obj = models.UniRepRidgeModel(do_sparse_refit=True, rep='et_global_init_1')  
    elif model == 'EvotunedUniRep_Global_Init_1_EnsembledRidgeSparseRefit':
        model_obj = models.UniRepEnsembledRidgeModel(do_sparse_refit=True, rep='et_global_init_1')
        
    elif model == 'EvotunedUniRep_Global_Init_2_LassoLars':
        model_obj = models.UniRepLassoLarsModel(do_sparse_refit=False, rep='et_global_init_2') 
    elif model == 'EvotunedUniRep_Global_Init_2_Ridge':
        model_obj = models.UniRepRidgeModel(do_sparse_refit=False, rep='et_global_init_2') 
    elif model == 'EvotunedUniRep_Global_Init_2_RidgeSparseRefit':
        model_obj = models.UniRepRidgeModel(do_sparse_refit=True, rep='et_global_init_2')  
    elif model == 'EvotunedUniRep_Global_Init_2_EnsembledRidgeSparseRefit':
        model_obj = models.UniRepEnsembledRidgeModel(do_sparse_refit=True, rep='et_global_init_2')
        
        
    # Commented out elif blocks correspond to replacement elif blocks above.
    #elif model == 'UniRepLassoLars':
    #    model_obj = models.UniRepLassoLarsModel()
    #elif model == 'UniRepRidge':
    #    model_obj = models.UniRepRidgeModel()
    #elif model == 'UniRepRidgeSparseRefit':
    #    model_obj = models.UniRepRidgeModel(do_sparse_refit=True)
    #elif model == 'EvotunedUniRepLassoLars':
    #    model_obj = models.EvotunedUniRepLassoLarsModel()
    elif model == 'AminoBERTLassoLars':
        model_obj = models.AminoBERTLassoLarsModel()
    elif model == 'AminoBERTRidge':
        model_obj = models.AminoBERTRidgeModel()
    elif model == 'EvotunedAminoBERTLassoLars':
        model_obj = models.EvotunedAminoBERTLassoLarsModel()
    elif model == 'SparseRefitLassoLars':
        model_obj = models.SparseRefitLassoLarsModel()
    elif model == 'SparseRefitUniRepLassoLars':
        model_obj = models.SparseRefitUniRepLassoLarsModel()
    #elif model == 'EvotunedUniRepRidge':
    #    model_obj = models.EvotunedUniRepRidgeModel()
    #elif model == 'EvotunedUniRepRidgeSparseRefit':
    #    model_obj = models.EvotunedUniRepRidgeModel(do_sparse_refit=True)
    #elif model == 'EvotunedUniRep2Ridge':
    #    model_obj = models.EvotunedUniRep2RidgeModel()
    #elif model == 'EvotunedUniRep2RidgeSparseRefit':
    #    model_obj = models.EvotunedUniRep2RidgeModel(do_sparse_refit=True)
    elif model == 'EvotunedUniRepBayesianRidge':
        model_obj = models.EvotunedUniRepBayesianRidgeModel()
    elif model == 'EvotunedUniRepPessimisticBayesianRidge':
        model_obj = models.EvotunedUniRepBayesianRidgeModel(pessimistic=True)
    #elif model == 'EvotunedRandomInitRidge':
    #    model_obj = models.EvotunedRandomInitRidgeModel()
    #elif model == 'EvotunedRandomInitRidgeSparseRefit':
    #    model_obj = models.EvotunedRandomInitRidgeModel(do_sparse_refit=True)
    elif model == 'EvotunedUniRepKNN':
        model_obj = models.EvotunedUniRepKNNModel()
    elif model == 'GP_RBF':
        model_obj = models.GPRBFModel()
    elif model == 'EvotunedUniRepGP_Cosine':
        model_obj = models.EvotunedUniRepGPCosineModel()
    elif model == 'EvotunedUniRepGP_RBF':
        model_obj = models.EvotunedUniRepGPRBFModel()
    elif model == 'EvotunedUniRepGP_RBF+DotProduct':
        model_obj = models.EvotunedUniRepGPRBFPlusDotProductModel()
    elif model == 'EvotunedUniRepGP_RBFxDotProduct':
        model_obj = models.EvotunedUniRepGPRBFTimesDotProductModel()
    elif model == 'EvotunedAminoBERTRidge_1':
        model_obj = models.EvotunedAminoBERTRidgeModel(rep='1')
    elif model == 'EvotunedAminoBERTRidge_2_84K':
        model_obj = models.EvotunedAminoBERTRidgeModel(rep='2_84K')
    elif model == 'EvotunedAminoBERTRidge_2_188K':
        model_obj = models.EvotunedAminoBERTRidgeModel(rep='2_188K')
    elif model == 'EvotunedAminoBERTRidge_3_25K':
        model_obj = models.EvotunedAminoBERTRidgeModel(rep='3_25K')
    elif model == 'EvotunedAminoBERTRidge_3_25K_all_layer':
        model_obj = models.EvotunedAminoBERTRidgeModel(rep='3_25K_all_layer')
    elif model == 'EvotunedAminoBERTRidge_4_25K':
        model_obj = models.EvotunedAminoBERTRidgeModel(rep='4_25K')
    #elif model == 'EvotunedUniRep_Global_Init_1_EnsembledRidgeSparseRefit':
    #    model_obj = models.UniRepEnsembledRidgeModel(do_sparse_refit=True, rep='et_global_init_1')
    #elif model == 'EvotunedUniRep_Global_Init_2_EnsembledRidgeSparseRefit':
    #    model_obj = models.UniRepEnsembledRidgeModel(do_sparse_refit=True, rep='et_global_init_2')
    #elif model == 'EvotunedUniRep_Random_Init_1_EnsembledRidgeSparseRefit':
    #    model_obj = models.UniRepEnsembledRidgeModel(do_sparse_refit=True, rep='et_random_init_1')
    
    
    ## Generalization sets, set up.
    data_io_utils.sync_s3_path_to_local(paths.GEN_SETS_SPLITS_DIR)
    
    sn_df = pd.read_csv(
        eval('paths.SYNTHETIC_NEIGH_SPLIT_%d_FILE' % split))
    fp_df = pd.read_csv(
        eval('paths.FP_HOMOLOGS_GEN_SPLIT_%d_FILE' % split))
    fused_df = A003_common.generate_simplified_and_fused_gen_set(sn_df, fp_df)
    
    generalization_set_dfs = []
    generalization_set_dfs.append(sn_df)
    generalization_set_dfs.append(fp_df)
    generalization_set_dfs.append(fused_df)

    generalization_set_names = ['syn_neigh', 'fp_homologs', 'combined_simple']

    generalization_set_sub_category_columns = [
        'nearest_parent', # corresponds to synthetic neighborhoods df
        None, # corresponds to fp homologs, don't factorize by subcategory.
        'gen_set', # Corresponds to combined simplified gen sets.
    ]

    generalization_set_calc_params = {
        # brightness threshold for calc-ing binarized statistics.
        # avGFP brightness is 1.0. Anything with brightness less
        # than this threshold is considered a false discovery.
        'quantitative_function_threshold': 1.0,
        'n_to_test_curve': N_TESTING_POINTS_SCHEDULE
    }
        
    ## Root output directory, to put replicate subdirs in.
    if 'get_name' in dir(model_obj):
        model_name = model_obj.get_name()
    else:
        model_name = model_obj.__class__.__name__
    
    root_output_dir = os.path.join(
        paths.POLICY_EVAL_DIR,
        experiment_name,
        'split_' + str(split), 
        'train_' + training_set_name,
        model_name, 
        acquisition_policy_obj.__class__.__name__)
        
    
    return {
        'n_training_points_schedule': n_training_points_schedule,
        'training_set_df': training_set_df,
        'acquisition_policy_obj': acquisition_policy_obj,
        'acquisition_policy_params': acquisition_policy_params,
        'model_obj': model_obj,
        'generalization_set_dfs': generalization_set_dfs,
        'generalization_set_names': generalization_set_names,
        'generalization_set_sub_category_columns': generalization_set_sub_category_columns,
        'generalization_set_calc_params': generalization_set_calc_params,
        'root_output_dir': root_output_dir
    }

############################################################################
### GENERALIZATION EVALUATION
############################################################################
def evaluate_generalization(
        model, 
        generalization_set_dfs,
        generalization_set_names,
        generalization_set_sub_category_columns,
        generalization_set_calc_params):
    
    gen_stat_dfs = [
        evaluate_generalization_on_gen_set(
            model,
            generalization_set_dfs[i],
            generalization_set_sub_category_columns[i],
            generalization_set_calc_params
        ) for i in range(len(generalization_set_dfs))
    ]
    
    for i in range(len(gen_stat_dfs)):
        gen_stat_dfs[i]['gen_set_name'] = generalization_set_names[i]
        
    aggregated_gen_stat_df = pd.concat(gen_stat_dfs)
    
    return aggregated_gen_stat_df

def evaluate_generalization_on_gen_set(
        model, 
        generalization_set_df, 
        sub_category_column,
        gen_calc_params):
    
    """
    model: should have a predict method that takes in as input
        raw amino acid sequences and outputs a quantitative
        function prediction.
    generalization_set_df: A dataframe with at least two columns
        'seq' and 'quantitative_function'. Optionally has other
        metadata columns that can be used to evaluate generalization
        wrt subsets of the dataset.
    sub_category_column: Column name in generalization_set_df or None.
        A column with categorical features that specify subsets (e.g. 
        neighborhoods) of generalization_set_df on which to calculate 
        generalization statistics. If not applicable, set to be None.
    gen_calc_params: Dictionary of parameters needed for calculating
        generalization statistics. Leaving this as a dictionary
        so it can organically grow as we add functionality.
        Required fields:
            'quantitative_function_threshold': Specifies threshold for
                binarizing actual and predicted quantitative function.
    """
    
    assert 'quantitative_function_threshold' in gen_calc_params
    assert 'n_to_test_curve' in gen_calc_params
        
    gen_seqs = generalization_set_df['seq']
    gen_qfunc = np.array(generalization_set_df['quantitative_function'])
    
    gen_qfunc_hat = model.predict(gen_seqs)
    
    gen_stats = {}
    # Statistics across the whole generalization dataset.
    gen_stats['overall'] = calculate_generalization_statistics(
            gen_qfunc, gen_qfunc_hat, gen_calc_params)
    
    if sub_category_column is not None:
        sub_categories = list(
            np.unique(generalization_set_df[sub_category_column]))
        
        for sc in sub_categories:
            mask = np.array(generalization_set_df[sub_category_column] == sc)

            gen_stats[sc] = calculate_generalization_statistics(
                gen_qfunc[mask], gen_qfunc_hat[mask], gen_calc_params)
        
    gen_stats_df = pd.DataFrame(gen_stats).T
    gen_stats_df.fillna(0, inplace=True)
    return gen_stats_df
    

def calculate_generalization_statistics(y, yhat, params):
    # Suppress divide by zero warnings. Happens
    # when we e.g. predict the mean and the correlation
    # and binary statistics get messed up.
    warnings.filterwarnings('ignore')
    
    y = y.reshape(-1)
    yhat = yhat.reshape(-1)
    
    gen_stats = {}
    
    # Continuous, regression statistics
    gen_stats['mse'] = np.mean((y-yhat)**2)
    gen_stats['pearson'] = scipy.stats.pearsonr(y, yhat)[0]
    gen_stats['spearman'] = scipy.stats.spearmanr(y, yhat)[0]
    
    # Binary, classification statistics.
    y_bin = (y > params['quantitative_function_threshold']).astype(int)
    y_hat_bin = (yhat > params['quantitative_function_threshold']).astype(int)
    
    gen_stats['tpr'] = calc_tpr(y_bin, y_hat_bin)
    gen_stats['fpr'] = calc_fpr(y_bin, y_hat_bin)
    gen_stats['fdr'] = calc_fdr(y_bin, y_hat_bin)
    
    # Prioritization statistics
    prioritization_stats = calc_recall_and_max_qfunc_curves(y, yhat, 
            params['n_to_test_curve'], params['quantitative_function_threshold'])
    
    for k in prioritization_stats:
        gen_stats[k] = prioritization_stats[k]
    
    warnings.filterwarnings('default')
    
    return gen_stats

def break_ties(yhat, noise_scale):
    yhat_old = copy.deepcopy(yhat)
    itr = 0
    while len(np.unique(yhat)) != len(yhat):
        yhat = yhat_old + 1e-4*noise_scale*np.random.randn(len(yhat))
        itr += 1
        if itr > 100:
            break
            
    return yhat

def calc_recall_and_max_qfunc_curves(y, yhat, n_curve, qfunc_threshold):
    RANDOM_BASELINE_N = 100
    
    # If there are a lot of predictions that are numerically
    # equivalent, the model is randomly prioritizing among them.
    # add a small amount of random jitter to reflect this. Shouldn't
    # affect genuine differences between model predictions.
    yhat = break_ties(yhat, np.std(y))
    
    sidx = np.argsort(-yhat) # sort descending
    ys = y[sidx] # sorted according to predictions
    yhats = yhat[sidx]
    
    n_desirable = np.sum(y > qfunc_threshold) # currently not used
    max_qfunc_possible = np.max(y) # currently not used
    
    recall_curve = []
    max_qfunc_curve = []
    for n in n_curve:
        y_tested = ys[:n]
        recall_curve.append(np.sum(y_tested > qfunc_threshold))
        max_qfunc_curve.append(np.max(y_tested))
        
    recall_curve = np.array(recall_curve)
    max_qfunc_curve = np.array(max_qfunc_curve)
        
    # Calculate a random baseline
    random_recall_curves = []
    random_max_qfunc_curves = []
    for r in range(RANDOM_BASELINE_N):
        
        yhat = y[np.random.permutation(len(y))] # random prediction
        yhat = break_ties(yhat, np.std(y))
        
        sidx = np.argsort(-yhat) # sort descending
        ys = y[sidx] # sorted according to predictions
        yhats = yhat[sidx]
        
        rr = []
        rmq = []
        for n in n_curve:
            y_tested = ys[:n]
            
            rr.append(np.sum(y_tested > qfunc_threshold)+1e-6)
            rmq.append(np.max(y_tested))
            
        random_recall_curves.append(np.array(rr))
        random_max_qfunc_curves.append(np.array(rmq))
        
    random_recall_curve = np.mean(np.stack(random_recall_curves,0), axis=0)
    random_max_qfunc_curve = np.mean(np.stack(random_max_qfunc_curves,0), axis=0)
    
    return {
        'recall_vs_n': recall_curve,
        'max_qfunc_vs_n': max_qfunc_curve,
        'random_recall_vs_n': random_recall_curve,
        'random_max_qfunc_vs_n': random_max_qfunc_curve,
        'recall_ratio_over_random_vs_n': recall_curve/random_recall_curve,
        'max_qfunc_ratio_over_random_vs_n': max_qfunc_curve/random_max_qfunc_curve,
        'cum_mean_recall_ratio_over_random_vs_n': cum_mean(recall_curve/random_recall_curve),
        'cum_mean_max_qfunc_ratio_over_random_vs_n': cum_mean(max_qfunc_curve/random_max_qfunc_curve)
    }
        
    
def cum_mean(x):
    return np.cumsum(x)/np.arange(1,len(x)+1)

def assert_array_is_binary(x):
    unique_vals = set(x.reshape(-1))
    assert not len(unique_vals - set([0, 1]))
    
def calc_tpr(y_bin, y_hat_bin):
    """
    Among the things that are truly positive, 
    what percentage of those do you call positive?
    """
    assert_array_is_binary(y_bin)
    assert_array_is_binary(y_hat_bin)
    
    mask = y_bin == 1
    return float(np.sum(y_hat_bin[mask] == 1))/(float(len(y_hat_bin[mask])) + 1e-8)

def calc_fpr(y_bin, y_hat_bin):
    """
    Among the things that are truly negative,
    what percentage of those do you call positive?
    """
    assert_array_is_binary(y_bin)
    assert_array_is_binary(y_hat_bin)
    
    mask = y_bin == 0
    return float(np.sum(y_hat_bin[mask] == 1))/(float(len(y_hat_bin[mask])) + 1e-8)

def calc_fdr(y_bin, y_hat_bin):
    """
    Among things you predict are positive, what percentage of them are not?
    """
    assert_array_is_binary(y_bin)
    assert_array_is_binary(y_hat_bin)
    
    mask = y_hat_bin == 1 # things we predict are positive.
    return float(np.sum(y_bin[mask] == 0))/(float(len(y_bin[mask])) + 1e-8)



