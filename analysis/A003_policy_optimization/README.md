# Retrospective experiments for low-N engineering

This folder contains code to reproduce all retrospective modeling experiments, including Supplementary Figures 1 and 2. 
These retrospective modeling experiments were used to determine hyperparameters that were used for prospective sequence design.

The notebooks to reproduce analyses are as follows:
1. 001_split_synthetic_neighborhoods_into_3_part_generalization_set.ipynb - Splits the SynNeigh dataset into three datasplits as discussed in the paper's methods section.
2. 002_split_sarkisyan_into_3_part_data_distribution_set.ipynb - Same as 1. but for Sarkisyan
3. 003_split_FP_homologs_into_3_part_data_distribution_and_generalization_sets.ipynb - Same as above, but for FP Homologs
4. 004_split_2_retrospective_perf_runs_for_main_text.ipynb - Runs the retrospective evaluation end-to-end. This is a compute and storage intensive notebook.
5. 005_Supp_Fig_1.ipynb - Generates Supplementary Fig 1 plots
6. 006_Supp_Fig_2.ipynb - Generates Supplementary Fig 2 plots
