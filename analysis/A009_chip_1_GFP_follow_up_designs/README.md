# Clonal (single-plex) flow characterization of eUniRep GFP designs, ancestral sequence reconstruction, and consensus sequence design.

Code to reproduce Figures 2c and 2d.

1. 001_analyze_and_format_FastML_ancestral_reconstruction_sequences - Analyzes FastML ASR sequences and finalizes list for synthesis and testing
2. 002_consensus_sequence_design - Performs consensus sequence design
3. 003_aggregate_UniRep_designs_ASR_consensus_seqs.ipynb - Aggregates ASR, consensus, and eUniRep designs into one file
4. 004_merge_twist_platemap_and_metadata - Merge's plate map from Twist with additional metadata. Required for downstream notebooks.
5. 005_analyze_single_plex_clonal_design_flow_data - Analyzes clonal (single-plex) flow measurements of eUniRep, Local UniRep, ASR, and Consensus sequences.
6. 006_link_singleplex_flow_with_flow_seq_data_and_visualize - Links single plex flow data with the FlowSeq data and visualizes. This notebook makes the panels in Fig 2c.
7. 007_make_2D_embedding_of_existing_ASR_consenus_eUniRep_GFPs - Makes the 2D sequence embedding shown in Fig 2d.

Unfortunately, in 001 there was a dictionary created that should have been created as an OrderedDict. Because the key order for a vanilla dictionary when iterating over it is not reproducible, this caused the file output at the end of the notebook to output rows in a different order. Though the conclusions are the exact same, this ends up changing some of the downstream visuals. We're sorry we weren't more careful in this situation! We've kept the original files used in the repository.