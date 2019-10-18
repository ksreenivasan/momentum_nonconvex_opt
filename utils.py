# %matplotlib inline
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import datetime
import math
# plt.style.use('seaborn-white')

def save_figure(ax, filename):
    fig = ax.get_figure()
    fig.savefig("data/{}".format(filename))
    fig.clear()
    plt.close()


def plot_aggregates(f_opt_df, dist_to_origin_df, dist_from_start_df, dist_moved_towards_origin_df):
    filenames = ['f_opt_aggregated.pdf', 'dist_to_origin_aggregated.pdf',
                    'dist_from_start_aggregated.pdf', 'dist_moved_towards_origin.pdf']
    for i, df in enumerate([f_opt_df, dist_to_origin_df, dist_from_start_df, dist_moved_towards_origin_df]):
        ax = df.plot.hist(alpha=0.5)
        save_figure(ax, 'hist_' + filenames[i])
        ax = df.plot.box()
        save_figure(ax, 'box_' + filenames[i])