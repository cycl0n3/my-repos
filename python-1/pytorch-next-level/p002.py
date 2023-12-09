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

# Model definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2560, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = x.view(-1, 2560)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(self.fc3(x), dim=1)