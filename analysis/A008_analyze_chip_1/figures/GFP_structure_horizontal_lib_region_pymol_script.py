reinitialize

#fetch 2WUR
fetch 1EMA
remove solvent
remove resn eoh
remove resn ipa

select full_chain, all
color gray40, full_chain
set cartoon_transparency, 0.7, full_chain

select lib_region, resi 29-110
color brightorange, lib_region
set cartoon_transparency, 0.0, lib_region

deselect

#show lines, lib_region

bg_color white



rotate x, 30, chain A
rotate y, 90, chain A

ray 600, 600
png /Users/surgebiswas/GitHub/efficient-prot/analysis/A008_analyze_chip_1/figures/GFP_structure_horizontal_lib_region.png, dpi=300

# https://pymolwiki.org/index.php/Publication_Quality_Images

