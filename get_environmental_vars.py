# This script provides a function to load the environmental variables from a CM1 output file 
# and add them to a dataframe of superdroplet trajectories.

import os
import os.path
import numpy as np
import pandas as pd
import xarray as xr


#time
from datetime import timedelta

# for loading the trajectories
workpath = '/glade/work/psturm/ice-mp-su24/'
os.chdir(workpath)
import load_trajectories as lt

def calculate_rh(T, P, qv, TYPE):
    # Constants for ice
    a0i = 6.11147274
    a1i = 0.503160820
    a2i = 0.188439774e-1
    a3i = 0.420895665e-3
    a4i = 0.615021634e-5
    a5i = 0.602588177e-7
    a6i = 0.385852041e-9
    a7i = 0.146898966e-11
    a8i = 0.252751365e-14

    # Constants for liquid
    a0 = 6.11239921
    a1 = 0.443987641
    a2 = 0.142986287e-1
    a3 = 0.264847430e-3
    a4 = 0.302950461e-5
    a5 = 0.206739458e-7
    a6 = 0.640689451e-10
    a7 = -0.952447341e-13
    a8 = -0.976195544e-15

    dt = np.maximum(-80., T - 273.16)
    rd = 287.04
    rv = 461.5
    eps = rd/rv
    if TYPE == 1:  # Ice
        polysvp = a0i + dt*(a1i+dt*(a2i+dt*(a3i+dt*(a4i+dt*(a5i+dt*(a6i+dt*(a7i+a8i*dt)))))))
    elif TYPE == 0:  # Liquid
        polysvp = a0 + dt*(a1+dt*(a2+dt*(a3+dt*(a4+dt*(a5+dt*(a6+dt*(a7+a8*dt)))))))
    polysvp = polysvp * 100.
    rh_out = qv / (eps * polysvp /(P-polysvp))
    return rh_out

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

    # calculate the air temperature
    rd = 287.0
    cp = 1004.0
    p00 = 100000.0
    rovcp  = rd/cp
    trajs['T [K]'] = trajs['th'].values * (trajs['prs'].values / p00 ) ** (rovcp)

    # calculate the relative humidity w.r.t ice
    # Apply the function to the dataframe
    trajs['RH_ice'] = calculate_rh(trajs['T [K]'], trajs['prs'], trajs['qv'], 1)
    trajs['RH_liquid'] = calculate_rh(trajs['T [K]'], trajs['prs'], trajs['qv'], 0)

    # make a new column that is the difference between RH_liquid and rh
    trajs['RH_diff'] = trajs['RH_liquid'] - trajs['rh']
    # note from Obin: if RH_diff is not zero, this might be because of how
    # the absolute temperature is given to the Fortran RH-calculating subroutine SAT_ICE3D_OUT
    # dum8(i,j,k)=(th0(i,j,k)+tha(i,j,k))*(pi0(i,j,k)+ppi(i,j,k))
    # call SAT_ICE3D_OUT(dum8,prs,qa(:,:,:,nqv),dum1,0)
    # where ppi is the perturbation nondimensional pressure ("Exner function")
    # and th0 and pi0 are base-state potential temperature and Exner function


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
selected_vars = ['rh', 'th', 'prs', 'qv', 'uinterp', 'vinterp', 'winterp', 'out8', 'out9', 'out10', 'out11', 'out12', 'out13', 'out14', 'deactrat']



trajs = get_environmental_vars(nc, trajs, selected_vars)

# find max absolute RH_diff in trajs
# max_abs_RH_diff = np.max(np.abs(trajs['RH_diff']))




