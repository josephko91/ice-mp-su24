# Description: 
# This script loads superdroplet data from a directory of ascii files,
# finds superdroplets within a certain radius of a given point, and
# bins the superdroplets by a given key (e.g. radius, density).
# The binned data is then plotted as a histogram.
# The class Bin_Superdroplets is used to store the data 
# perform the binning, and visualize the distributions.

import numpy as np
import os.path
import os
import pandas as pd


def get_timestamps(dirpath):
    timestamps = []

    # filter trajfiles to only include "SD_output_ASCII"
    trajfiles = [file for file in os.listdir(dirpath) if 'SD_output_ASCII' in file]
    for file in trajfiles:
        timestamps.append(int(file[16:21]))

    return np.unique(timestamps)

def get_unique_SDs(dirpath,timestep):
    # this function returns the unique superdroplets given a time step
    timestamps = []
    processors = []
    filelen = []
    filenames = []
    # these are the column names for the data in the ascii files
    colnames =['x[m]','y[m]','z[m]','vz[m]','radius(droplet)[m]','mass_of_aerosol_in_droplet/ice(1:01)[g]','radius_eq(ice)[m]','radius_pol(ice)[m]',
               'density(droplet/ice)[kg/m3]','rhod [kg/m3]','multiplicity[-]','status[-]','index','rime_mass[kg]','num_of_monomers[-]','rk_deact']


    # filter trajfiles to only include "SD_output_ASCII"
    trajfiles = [file for file in os.listdir(dirpath) if 'SD_output_ASCII' in file]
    for file in trajfiles:
        filelen.append(len(file))
        filenames.append(file)
        timestamps.append(int(file[16:21]))
        processors.append(int(file[24:]))
    idx0 = np.where(np.array(timestamps)==timestep)
    fn0 = [filenames[idx0[0][i]] for i in range(0,len(idx0[0]))]
    # this goes through all the files at the given time step
    # load the trajectory data
    trajs_list = []
    for fn in fn0:
        filepath = os.path.join(dirpath,fn)
        traj=pd.read_csv(filepath,sep = '\s+',skiprows=1,header=None,
                         delim_whitespace=False,names=colnames,index_col='rk_deact')
        trajs_list.append(traj)
    # concatenate it to the pandas dataframe
    # outside the for loop to be more efficient
    trajs = pd.concat(trajs_list)

    # find unique superdroplets
    unique_superdroplets = np.unique(trajs.index)
    return unique_superdroplets


def load_trajectories(dirpath,
                      num_timesteps=1,
                      times=None,
                      Ns_array = None):

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

    times = np.unique(timestamps) if times is None else times

    # add check to make sure the number of timesteps is less than the number of timestamps
    if num_timesteps > len(times):
        print(f'Error: num_timesteps {num_timesteps} is greater than the number of timestamps {len(times)}')
        return None

    trajs_list = []
    for t in range(num_timesteps):
        print(f'Loading trajectories for time {times[t]} ')
        idx0 = np.where(timestamps==times[t])
        fn0 = [filenames[idx0[0][i]] for i in range(0,len(idx0[0]))]
        # this goes through all the files at the current time step
        # load the trajectory data 
        for fn in fn0: 
            filepath = os.path.join(dirpath,fn)
            traj=pd.read_csv(filepath,sep = '\s+',skiprows=1,header=None,
                             delim_whitespace=False,names=colnames,index_col='rk_deact')
            # add the time step to the dataframe
            traj['time'] = times[t]

            # if Ns_array is not None, then filter the dataframe to only include them
            if Ns_array is not None:
                traj = traj[traj.index.isin(Ns_array)]

            trajs_list.append(traj)
    # concatenate it to the pandas dataframe
    # outside the for loop to be more efficient
    trajs = pd.concat(trajs_list)

    # remove all z[m] < 0
    trajs = trajs[trajs['z[m]'] > 0]

    # reset the index 
    trajs = trajs.reset_index()


    return trajs

# # # example usage of how to load the trajectories
dirpath = '/glade/derecho/scratch/klamb/superdroplets/outsdm_iceball_nowind_rhod_dist_min200_sgs_1024_poly_trj_5400_7200'
# trajs = load_trajectories(dirpath,num_timesteps=2)

# # # filter the trajectories to only include droplet index 304
# # # organized by timestep
# traj304 = trajs[trajs['index']==304]

# # # example of loading coarsened timestamps
# timestamps = get_timestamps(dirpath)
# # # # load every 4 timestamps (every two minutes)
# coarse_timestamps = timestamps[0::4]
# print(coarse_timestamps)
# trajs = load_trajectories(dirpath,num_timesteps=5,times=coarse_timestamps)


# # # example of loading unique superdroplets
# # # load the unique superdroplets at a given time step
# # # now of course, you could also filter unique superdroplets 
# # # by a certain radius or density or whatever you want
# unique_superdroplets = get_unique_SDs(dirpath,6165)
# Ns = 2^4
# first_Ns = unique_superdroplets[0:Ns]
# trajs = load_trajectories(dirpath,num_timesteps=4,Ns_array=first_Ns)