import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import gridspec
# import imageio

# change the working directory to be able to use the load_trajectories functionality
workpath = '/glade/work/psturm/ice-mp-su24/'
os.chdir(workpath)
import load_trajectories as lt

# Get the timestamps
dirpath = '/glade/derecho/scratch/klamb/superdroplets/outsdm_iceball_nowind_rhod_dist_min200_time_var_sgs_1024_poly_trj/SDM_trajs/'
# dirpath = '/glade/derecho/scratch/klamb/superdroplets/outsdm_iceball_nowind_rhod_dist_min200_sgs_1024_poly_trj_5400_7200/'
timestamps = lt.get_timestamps(dirpath)
# coarse timestamps should be every minutes (every 2 timestamps) until the end
coarse_timestamps = timestamps[::-10][:5][::-1]
# coarse_timestamps = coarse_timestamps[:2]

# Get the trajectories
trajs = lt.load_trajectories(dirpath, 
                             num_timesteps = 5,
                             times=coarse_timestamps)
# reset index
trajs = trajs.reset_index()
# get superdroplet id 989898 from rk_deact
# trajs[trajs["rk_deact"] == 310028008]

# remove all rows that have negative z values
trajs = trajs[trajs["z[m]"] >= 0]

# hard code rk_deact 543263375 at time 5100 to have a rhod of .23
# to make sure at least one superdroplet has a time varying rhod
trajs.loc[(trajs["rk_deact"] == 735521987) & (trajs["time"] == 6000), "rhod [kg/m3]"] = .23

# see if any superdroplet has a time varying rhod
unique_rhod_counts = trajs.groupby("rk_deact")["rhod [kg/m3]"].nunique()
sd_with_time_varying_rhod = unique_rhod_counts[unique_rhod_counts > 1].index
for sd in sd_with_time_varying_rhod:
    print("superdroplet", sd, "has a time varying rhod")


# remove all rows that have negative z values
# trajs = trajs[trajs["z[m]"] >= 0]