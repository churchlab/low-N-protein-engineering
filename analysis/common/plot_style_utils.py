"""
Utilities for plotting beautiful figures.
"""
import matplotlib.pyplot as plt
#import palettable as pal
import seaborn as sns
import pandas as pd

CAT_PALETTE = sns.color_palette('colorblind')
DIV_PALETTE = sns.color_palette("BrBG_r", 100)
SEQ_PALETTE = sns.cubehelix_palette(100, start=0.5, rot=-0.75)
GRAY = [0.5, 0.5, 0.5]

def prettify_ax(ax):
    """
    Nifty function we can use to make our axes more pleasant to look at
    """
    for i,spine in enumerate(ax.spines.values()):
        if i == 3 or i == 1:
            spine.set_visible(False)
    ax.set_frameon = True
    #ax.patch.set_facecolor('#eeeeef')
    #ax.grid('off', color='w', linestyle='-', linewidth=1)
    ax.tick_params(direction='out', length=3, color='k')
    ax.set_axisbelow(True)


def simple_ax(figsize=(6, 4), **kwargs):
    """
    Shortcut to make and 'prettify' a simple figure with 1 axis
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, **kwargs)
    prettify_ax(ax)
    return fig, ax

def set_pub_plot_context(colors='categorical', context="talk"):
    sns.set(style="white", context=context, font="Helvetica")


def save_for_pub(fig, path="../../data/default", dpi=300, include_vector=True):
    fig.savefig(path + ".png", dpi=dpi, bbox_inches='tight', transparent=True)
    #fig.savefig(path + ".eps", dpi=dpi, bbox_inches='tight')
    if include_vector:
        fig.savefig(path + ".pdf", dpi=dpi, bbox_inches='tight', transparent=True)
        fig.savefig(path + ".svg", dpi=dpi, bbox_inches='tight', transparent=True)
    #fig.savefig(path + ".emf", dpi=dpi, bbox_inches='tight')
   #fig.savefig(path + ".tif", dpi=dpi)
