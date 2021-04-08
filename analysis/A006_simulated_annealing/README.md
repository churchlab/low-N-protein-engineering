# Simulated annealing for prospective design of GFP and beta-lactamase

This directory contains the code needed to perform prospective design of GFP and beta-lactamase. 

## Notebook organization
To reproduce analyses, run the following notebooks in order. Note some of the notebooks need to do large data syncs, so make sure you have enough disk space (~500 GB).

1. 001_generate_GFP_hyperborg_run_config_dicts.ipynb - Generates simulated annealing configuration dictionaries for GFP and outputs them as pickles. See "Running sequence design" section below.
2. 002_generate_BLAC_hyperborg_run_config_dicts.ipynb - Same as 1. but for beta-lactamase.
3. 003_select_sequences_from_GFP_and_BLAC_SA_runs.ipynb - Selects sequences from simulated annealing runs for testing. Assumes simulated annealing runs have already been performed, and draws from pre-saved output. To run the simulated annealing fresh, see "Running sequence design" section below.
4. 004_aggregate_SA_selected_seqs_into_chip_1_Twist_order.ipynb - Aggregates selected sequences into an order for Twist.
5. 005_validate_aggregated_and_in_silico_cloning_verified_seqs.ipynb - Runs an independent validation of selected sequences.
6. 006_finalize_chip_1_order_for_Twist.ipynb - Finalizes the order for Twist. Outputs the exact file used to submit sequence designs to Twist.


## Running sequence design

The above notebooks handle the entire sequence design process, except simulated annealing itself.

Prospective design was run by two scripts in the hyperborg sub-directory:
1. `BLAC_simulated_annealing.py`
2. `GFP_simulated_annealing.py`

Both of these scripts take in as input a pickled configuration dictionary. 
These are also contained in the `hyperborg` sub-directory. 
The config dictionaries are generated in notebooks 001 and 002, and each specifies
the parameters of a particular simulated annealing run, including for example,
which representation to use, the top model, target, protein, trust radius, etc.

To run the simulated annealing run specified by e.g. 
`hyperborg/BLAC_SA_config-ET_Random_Init_1-0024-07-245f5589.p`  do the following steps.

1. Start docker as described in the repository root README.
2. Note that by default this starts a jupyter notebook. If you ssh into the machine again in a separate shell
you can "ssh" into the docker environment as follows. `docker exec -it <container_name> /bin/bash`. Use
`docker ps` to list active containers and their names.
3. Once inside the container, navigate to the repository and into the `analysis/A006_simulated_annealing/hyperborg`
directory.
4. There enter `python BLAC_simulated_annealing.py BLAC_SA_config-ET_Random_Init_1-0024-07-245f5589.p`. This will
run simulated annealing for all configuration parameters specified in the passed pickle file.

**Note**
First, we have only tested the above on `p3.2xlarge` EC2 machines on Amazon AWS. The simulated annealing will take
impossibly long without a GPU. If using a GPU smaller than a V100, you will need to adjust the iteration batch size 
to accommodate the smaller memory.

Second, When we actually ran this analysis, the `*_simulated_annealing.py` scripts were run over all of the configs
in parallel using an internal Church lab utility, called Hyperborg (hence the name of the sub-directory).
This is a utility we wrote to spin-up EC2 GPU instances on Amazon AWS and run jobs
on them automatically. While we have validated reproducibility for a handful of runs using the strategy outlined above,
we have not done this for all possible runs. Everything should be reproducible, but if you notice any discrepancies 
please reach out.

Hyperborg is an involved codebase and depends on a number of Church Lab specific AWS configurations,
and is therefore not something we are not able to share easily. 

Third, we have disabled some automatic S3 syncing functionality at the end of each simulated annealing run. This 
won't change the output of the run at all. The buckets we sync-ed to in the writing of this paper are not available
for public syncs.

