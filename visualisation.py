# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 15:45:38 2022

@author: kayva
"""



import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import struct

import nelpy as nel
import nelpy.io
import nelpy.plotting as npl
import spikeinterface.extractors as se 
import tempfile
os.environ['TEMPDIR'] = tempfile.gettempdir()
import matplotlib.pyplot as plt
import pandas as pd
from matplotvideo import attach_video_player_to_figure
import numpy as np
import cv2
from datetime import datetime, timedelta


# First, we load the TemplateModel.
sorting_check = se.PhySortingExtractor('C:\\Users\\kayva\\OneDrive\\Documents\\TestOverlaySpike\\output\\phy_MS')
print(f'Spike train of a unit: {sorting_check.get_unit_spike_train(1)}')
timestamps = sorting_check.get_unit_spike_train(1)
timestamps = timestamps  /30000
timestamps = np.round(timestamps,2)

tabx = []
taby =[]
timest = []
with open(r"\\genzel-srv.science.ru.nl\genzel\Rat\HM\Rat_HM_Ephys\Rat_HM_Ephys_Rat1_389236\Rat_HM_Ephys_Rat1_389236_Trackerfiles\Log\log_20200909_Rat1.log", 'r') as f:
    for line in f:   
        if "@" in line:
            linecut = line[:-1] 
            start = line.index('(')
            end = line.index(',',start+1)
            positionx = line[start+1:end]
            start = line.index(',')
            end = line.index(')',start+2)
            positiony = line[start+2:end]
            timestamp = linecut[7:-41]
            tabx.append(positionx)
            taby.append(positiony)
            utc_time = datetime.strptime(timestamp,"%H:%M:%S.%f")
            milliseconds = (utc_time - datetime(1900, 1, 1)) // timedelta(milliseconds=1)
            timest.append(milliseconds)
tabx = pd.DataFrame(tabx)
taby = pd.DataFrame(taby)
timest = pd.DataFrame(timest)/1000
tab = pd.concat([timest,tabx,taby],axis= 1)
tab.columns = ["timestamp", "x","y"]
timestamps = pd.DataFrame(timestamps)
timestamps.columns = ['timestamp']
merge = pd.merge(timestamps, tab, on = ['timestamp'],how = 'inner')
merge["timestamp"] = ((merge['timestamp']))
arr = merge.to_numpy().astype('int64')  
# assume default aesthetics
npl.setup()




st = nel.SpikeTrainArray(timestamps=spikes, support=session_bounds, fs=FS)
st.support

rest.domain


with npl.FigureManager(show=True, figsize=(30,4)) as (fig, ax):
    npl.utils.skip_if_no_output(fig)
    npl.rasterplot(st, lw=0.5, ax=ax)
    npl.epochplot(~rest, alpha=0.3)
    ax.set_xlim(*session_bounds.time)

sigma_100ms = 0.1
speed2d = nel.utils.dxdt_AnalogSignalArray(pos, smooth=True, sigma=sigma_100ms) / pixels_per_cm

run_epochs = nel.utils.get_run_epochs(speed2d.smooth(sigma=0.5), v1=2,v2=1) # original choice

run_epochs
with npl.FigureManager(show=True, figsize=(64,3)) as (fig, ax):
    npl.utils.skip_if_no_output(fig)
    plt.plot(pos.time, pos.asarray().yvals[0,:], lw=1, alpha=0.2, color='gray')
    plt.plot(pos.time, pos.asarray().yvals[1,:], lw=1, alpha=0.2, color='gray')
    npl.plot(pos[run_epochs], ax=ax, lw=1, label='run')
    
    plt.title('run activity')

st_run = st[run_epochs]

ds_run = 0.5 # 100 ms
ds_50ms = 0.05

# smooth and re-bin:
sigma = 0.3 # 300 ms spike smoothing
bst_run = st_run.bin(ds=ds_50ms).smooth(sigma=sigma, inplace=True).rebin(w=ds_run/ds_50ms)

sigma = 0.2 # smoothing std dev in cm
tc2d = nel.TuningCurve2D(bst=bst_run,
                         extern=pos, 
                         ext_nx=50, 
                         ext_ny=50, 
                         ext_xmin=190, 
                         ext_xmax=540, 
                         ext_ymin=90, 
                         ext_ymax=440, 
                         sigma=sigma, 
                         min_duration=0)

plt.matshow(tc2d.occupancy.T, cmap=plt.cm.Spectral_r)
plt.gca().invert_yaxis()


