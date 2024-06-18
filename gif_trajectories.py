# This script loads trajectories of ice superparticles
# and visualizes them as a gif

import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import gridspec
import imageio

# change the working directory to be able to use the load_trajectories functionality
workpath = '/glade/work/psturm/ice-mp-su24/'
os.chdir(workpath)
import load_trajectories as lt

# Get the timestamps
# dirpath = '/glade/derecho/scratch/klamb/superdroplets/outsdm_iceball_nowind_rhod_dist_min200_time_var_sgs_1024_poly_trj/SDM_trajs/'
dirpath = '/glade/derecho/scratch/klamb/superdroplets/outsdm_iceball_nowind_rhod_dist_min200_sgs_1024_poly_trj_5400_7200/'
timestamps = lt.get_timestamps(dirpath)
# coarse timestamps should be every minutes (every 2 timestamps) until the end
coarse_timestamps = timestamps[::4]
# coarse_timestamps = coarse_timestamps[:2]

class Bin_Superdroplets:
    """
    This class is used to store superdroplet data
    and perform binning on the data.
    The binned data can then be visualized as a histogram.
    """
    def __init__(self, data, bins=50):
        """Initialize with data and number of bins."""
        self.data = data
        self.bins = bins
        self.bin_counts = {}
        self.bin_edges = {}

    def calculate_histogram(self, key):
        """Calculate histogram for data associated with key."""
        weights = self.data['multiplicity[-]'] if 'multiplicity[-]' in self.data else None
        self.bin_counts[key], self.bin_edges[key] = np.histogram(self.data[key], bins=self.bins, weights=weights)

    def plot_histogram(self, key, ax, scale_factor=1, color=None, x_label=None):
        """
        Plot histogram for data associated with key.
        Raises KeyError if calculate_histogram has not been called for this key.
        """
        if key not in self.bin_counts or key not in self.bin_edges:
            raise KeyError(f"Histogram not calculated for key {key}")
        ax.bar(self.bin_edges[key][:-1]*scale_factor, self.bin_counts[key], 
               width=np.diff(self.bin_edges[key])*scale_factor, align="edge", color=color)
        ax.set_xlabel(x_label if x_label else key)

# Load the trajectories
for timestamp in coarse_timestamps:
    traj = lt.load_trajectories(dirpath, times=[timestamp])
    traj = traj.reset_index()

    # remove all entries with negative z values
    traj = traj[traj['z[m]'] > 0]

    xs = traj['x[m]'].values
    ys = traj['y[m]'].values
    zs = traj['z[m]'].values
    rads = 1e4*traj['radius_eq(ice)[m]'].values
    rhod = traj['rhod [kg/m3]'].values

    print("Plotting trajectories at timestamp", timestamp)

    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1]) 

    # First row of subplots
    ax0 = plt.subplot(gs[0, :])
    scatter = ax0.scatter(xs, zs, s=rads, c=rhod)
    ax0.set_ylabel("Height (m)")
    ax0.set_xlabel("Width (m)")
    fig.colorbar(scatter, ax=ax0, label='Deposition Density $\\rho_d$ [kg/m$^3$]')
    scatter.set_clim(0,1000)
    ax0.set_ylim(7000,11000)
    ax0.set_title(f'Trajectories at {timestamp} (minute {(timestamp-coarse_timestamps[0])/60})')

    # Second row of subplots
    sd2bin = Bin_Superdroplets(traj)
    sd2bin.calculate_histogram('radius_eq(ice)[m]')
    sd2bin.calculate_histogram('rhod [kg/m3]')
    ax1 = plt.subplot(gs[1, 0])
    sd2bin.plot_histogram('radius_eq(ice)[m]', ax1, scale_factor=1e6, x_label='Equivalent Radius [$\mu$m]')
    ax2 = plt.subplot(gs[1, 1],sharey=ax1)
    sd2bin.plot_histogram('rhod [kg/m3]', ax2, color='lightblue', x_label='Deposition Density $\\rho_d$ [kg/m$^3$]')
    fig.text(0.04, 0.5, 'Ice particle number', va='center', rotation='vertical')

    # save figure in the traj_png folder with the corresponding timestamp in its name
    plt.savefig(f'traj_png/traj_{timestamp}.png')
    plt.close(fig)  # Close the figure to free up memory

# Create a gif from the saved figures
images = []
for timestamp in coarse_timestamps:
    images.append(imageio.imread(f'traj_png/traj_{timestamp}.png'))
imageio.mimsave('traj_png/trajectories_poly_trj_5400_7200.gif', images, fps=1)

# find the min rhod in traj
traj["rhod [kg/m3]"].min()

