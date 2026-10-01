import warnings
import logging
import math 
import matplotlib.pyplot as plt
import squidpy as sq 

def plot_celltypes(adata,celltype_col,n_cols=4,w=2.5,h=3,dot_size=5):
    

    ctypes = list(set(adata.obs[celltype_col]))
    n_rows = math.ceil(len(ctypes) / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(w * n_cols, h * n_rows))
    axes = axes.flatten()

    logging.disable(logging.WARNING)

    for i, ctype in enumerate(ctypes):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sq.pl.spatial_scatter(
                adata,
                color=celltype_col,
                groups=ctype,
                img_alpha=0.1,
                title=ctype,
                legend_loc=None,
                frameon=False,
                size=dot_size,
                alpha=0.8,
                ax=axes[i]
            )

    # hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    logging.disable(logging.NOTSET)
    plt.subplots_adjust(wspace=0.05, hspace=0.15)
    plt.tight_layout(pad=0.2)
    plt.show()