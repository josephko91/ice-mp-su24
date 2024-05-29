# Using the NPL 2024a kernel
import numpy as np
import xarray as xr
import os.path
import os
import pandas as pd

from matplotlib import pyplot as plt

# filepaths for trajectories
dirpath = '/glade/derecho/scratch/klamb/superdroplets/outsdm_iceball_nowind_rhod_dist_min200_time_var_sgs_1024_poly_trj/SDM_trajs/'



def load_trajectories(dirpath, num_timesteps=1):
    timestamps = []
    processors = []
    filelen = []
    filenames = []

    # filter trajfiles to only include "SD_output_ASCII"
    trajfiles = [file for file in os.listdir(dirpath) if 'SD_output_ASCII' in file]
    for file in trajfiles:
        filelen.append(len(file))
        filenames.append(file)
        timestamps.append(int(file[16:21]))
        processors.append(int(file[24:]))

    # these are the column names for the data in the ascii files
    colnames =['x[m]','y[m]','z[m]','vz[m]','radius(droplet)[m]','mass_of_aerosol_in_droplet/ice(1:01)[g]','radius_eq(ice)[m]','radius_pol(ice)[m]',
               'density(droplet/ice)[kg/m3]','rhod [kg/m3]','multiplicity[-]','status[-]','index','rime_mass[kg]','num_of_monomers[-]','rk_deact']

    times = np.unique(timestamps)
    trajs_list = []
    for t in range(num_timesteps):
        idx0 = np.where(timestamps==times[t])
        fn0 = [filenames[idx0[0][i]] for i in range(0,len(idx0[0]))]
        # this goes through all the files at the first time step
        # load the trajectory data 
        for fn in fn0: 
            filepath = os.path.join(dirpath,fn)
            traj=pd.read_csv(filepath,sep = '\s+',skiprows=1,header=None,delim_whitespace=False,names=colnames,index_col='rk_deact')
            trajs_list.append(traj)
    # concatenate it to the pandas dataframe
    # outside the for loop to be more efficient
    trajs = pd.concat(trajs_list)
    return trajs

# load the trajectories
# trajs = load_trajectories(dirpath)

def find_neighbs(trajs, midpoint, radius):
    # This function finds all superdroplets
    # within a certain radius of a given midpoint
    # midpoint is a tuple of 3 floats, position in meters
    # radius is a float, distance in meters
    squared_distance = (trajs['x[m]']-midpoint[0])**2 + (trajs['y[m]']-midpoint[1])**2 + (trajs['z[m]']-midpoint[2])**2
    neighbors = trajs[squared_distance < radius**2]
    return neighbors

# Want the whole domain? Set a 
neighbors = find_neighbs(trajs, [0,0,0], 2e5)


class Bin_Superdroplets:
    def __init__(self, data, bins=100):
        self.data = data
        self.bins = bins
        self.bin_counts = {}
        self.bin_edges = {}

    def calculate_histogram(self, key):
        self.bin_counts[key], self.bin_edges[key] = np.histogram(self.data[key], bins=self.bins)

    def plot_histogram(self, key, ax, scale_factor=1, color=None, x_label=None):
        ax.bar(self.bin_edges[key][:-1]*scale_factor, self.bin_counts[key], 
               width=np.diff(self.bin_edges[key])*scale_factor, align="edge", color=color)
        ax.set_xlabel(x_label if x_label else key)

# Usage:
sd2bin = Bin_Superdroplets(neighbors)
sd2bin.calculate_histogram('radius_eq(ice)[m]')
sd2bin.calculate_histogram('rhod [kg/m3]')

fig, axs = plt.subplots(1, 2, figsize=(10,5), sharey=True)
sd2bin.plot_histogram('radius_eq(ice)[m]', axs[0], scale_factor=1e6, x_label='Equivalent Radius [$\mu$m]')
sd2bin.plot_histogram('rhod [kg/m3]', axs[1], color='lightblue', x_label='Deposition Density $\\rho_d$ [kg/m$^3$]')
fig.text(0.04, 0.5, 'Count', va='center', rotation='vertical')
plt.show()

# Save the figure
# fig.savefig('./figs/bins_entire_domain.png')