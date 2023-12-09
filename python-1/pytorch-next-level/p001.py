import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

import matplotlib.pyplot as plt

print("PyTorch version: ", torch.__version__)
print("Torch audio version: ", torchaudio.__version__)
print("CUDA available: ", torch.cuda.is_available())

print("Matplotlib version: ", plt.matplotlib.__version__)

yesno_data  = torchaudio.datasets.YESNO('./data', download=True)

print("Number of data points in YesNo dataset: ", len(yesno_data))

data_loader = torch.utils.data.DataLoader(yesno_data, batch_size=1, shuffle=True)

for data in data_loader:
    print("Data: ", data)
    print("Waveform: {}\nSample rate: {}\nLabels: {}".format(data[0], data[1], data[2]))
    break
