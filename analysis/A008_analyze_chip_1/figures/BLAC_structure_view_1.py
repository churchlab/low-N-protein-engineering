reinitialize

bg_color white

fetch 1XPB
remove solvent
remove resn eoh
remove resn ipa
remove resn SO4

select full_chain, all
color gray40, full_chain
set cartoon_transparency, 0.5, full_chain

# shifted by 2 on the PDB structure relative to our python indices
# due to Ambler numbering in the PDB structure.
select lib_region, resi 134-215 
color brightorange, lib_region
set cartoon_transparency, 0.0, lib_region
deselect

# Select key catalytic residues
# #+73+130+166
select catalytic, resi 70
show spheres, catalytic
set cartoon_transparency, 1.0, catalytic
deselect



rotate x, 270, chain A
rotate y, -30, chain A
rotate x, -30, chain A

ray 600, 600
png /Users/surgebiswas/GitHub/efficient-prot/analysis/A008_analyze_chip_1/figures/BLAC_structure_view1_lib_region.png, dpi=300

# https://pymolwiki.org/index.php/Publication_Quality_Images

