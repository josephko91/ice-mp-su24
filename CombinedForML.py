import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt

#Loading csv file
#workpath = '/glade/work/psturm/ice-mp-su24/saved_trajectory_data/'
workpath = '/glade/u/home/ashleyn/ice-mp-su24'

#file = 'trajs__5100_7200_Ns10000.csv'
file = 'Trajs_NS_1000.csv'

trajs = pd.read_csv(workpath+file)
