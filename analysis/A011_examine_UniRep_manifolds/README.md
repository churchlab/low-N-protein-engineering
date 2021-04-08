# Visualize eUniRep, Local UniRep, and OneHot representations in 2 dimensions

Generate plots contained in Fig 4 and Supp Fig 17.

1. 001_compute_random_sequences_for_PC_computation - Generates the list of random sequences that are used for computing PCs. This notebook stores these sequences in a file already provided in this directory. Thus you do not need to run this notebook to run the ones below.
2. 002_inference_GFP_sequences_for_manifold_viz - Inferences all relevant GFP sequences and stores the result in a pickle file that lives on S3. The notebooks below load representation vectors from these pickles instead of computing them fresh.  Thus you do not need to run this notebook to run the ones below.
3. 003_inference_BLAC_sequences_for_manifold_viz - Same as 002, but for beta-lactamase.
4. 004_visualize_GFP_manifold - This notebook actually generates the PC plots used in Fig 4 using the computations from the notebooks above.
5. 005_visualize_BLAC_manifold - Same as 004, but for the beta-lactamase figure (Supp Fig 17).