# This script provides a function to load the environmental variables from a CM1 output file 
# and add them to a dataframe of superdroplet trajectories.

import os
import os.path
import numpy as np
import pandas as pd
import xarray as xr


#for calculating the air temperature
from metpy.units import units
from metpy.calc import temperature_from_potential_temperature

#time
from datetime import timedelta

# for loading the trajectories
workpath = '/glade/work/psturm/ice-mp-su24/'
os.chdir(workpath)
import load_trajectories as lt

def get_environmental_vars(nc, trajs, selected_vars):
    x_index_array = np.zeros(len(trajs), dtype=int)
    y_index_array = np.zeros(len(trajs), dtype=int)
    z_index_array = np.zeros(len(trajs), dtype=int)
    
    for index, row in trajs.iterrows():
        x_index_array[index] = (abs(row['x[m]']-nc['xh']* 1000).argmin())
        y_index_array[index] = (abs(row['y[m]']-nc['yh']* 1000).argmin())
        z_index_array[index] = (abs(row['z[m]']-nc['z']* 1000).argmin())

    trajs['xi gridbox'] = x_index_array
    trajs['yk gridbox'] = y_index_array
    trajs['zh gridbox'] = z_index_array

    for var in selected_vars:
        values = []
        for zk, yj, xi in zip(trajs['zh gridbox'].values, trajs['yk gridbox'].values, trajs['xi gridbox'].values):
            value = nc[var][0, zk, yj, xi].values * 1
            values.append(value)
        trajs[var] = values

    return trajs

# Example usage:

# get environmental file
dirpath = "/glade/derecho/scratch/klamb/superdroplets/outsdm_iceball_nowind_rhod_dist_min200_sgs_1024_poly_trj_5400_7200"
envfile = os.path.join(dirpath, "cm1out_only_upto35.nc")
nc = xr.open_dataset(envfile)

# get trajectory dataframe
timestamps = lt.get_timestamps(dirpath)
# load every 4 timestamps (every 1 minute)
coarse_timestamps = timestamps[0::4]

# only look at 10 superdroplets 
Ns = 10
unique_superdroplets = lt.get_unique_SDs(dirpath, coarse_timestamps[0])
first_Ns = unique_superdroplets[0:Ns]
trajs = lt.load_trajectories(dirpath, times=coarse_timestamps,
                         num_timesteps = 3, 
                         Ns_array = first_Ns)


# define the variables we want to append
selected_vars = ['rh', 'th', 'prs', 'uinterp', 'vinterp', 'winterp', 'out8', 'out9', 'out10', 'out11', 'out12', 'out13', 'out14', 'deactrat']



trajs = add_env_vars_to_traj(nc, trajs, selected_vars)