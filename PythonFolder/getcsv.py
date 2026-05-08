import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from IPython import get_ipython

import mne
from utils import load_eeg_data, FREQ_BANDS, CHANNEL_GROUPS


my_path = Path('c:/Users/Hanvit Lee/Documents/School/EEGLabEnv')
const_path = os.path.join(my_path, 'binepochs_filtered_ICArej_P1AvgBOS5.set')
const_epochs = mne.io.read_epochs_eeglab(const_path, eog = (), uint16_codec = None, montage_units = 'auto', verbose = None)


for i in range(120):
    current_trial = np.transpose(np.column_stack((np.linspace(1, 32, num = 32), np.squeeze(const_epochs.get_data(item = i)))))
    np.savetxt(f"c:/Users/Hanvit Lee/Documents/School/EEGLabEnv/CISI_data/CISI{i+1}.csv", current_trial, delimiter = ",")



"""
my_path = Path('c:/Users/Hanvit Lee/Documents/School/EEGLabEnv')
const_path = os.path.join(my_path, 'binepochs_filtered_ICArej_P2AvgBOS5.set')
const_epochs = mne.io.read_epochs_eeglab(const_path, eog = (), uint16_codec = None, montage_units = 'auto', verbose = None)


for i in range(120):
    current_trial = np.transpose(np.column_stack((np.linspace(1, 32, num = 32), np.squeeze(const_epochs.get_data(item = i)))))
    np.savetxt(f"c:/Users/Hanvit Lee/Documents/School/EEGLabEnv/VISI_data/VISI{i+1}.csv", current_trial, delimiter = ",")
"""

