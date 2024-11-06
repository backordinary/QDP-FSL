# https://github.com/Red2Je/IBMInternship/blob/3bd7034c1a4245c134b44c682c549491dfed3ce6/WSL%20works/torch%20gpu/Net.py
#La description de cette classe est disponible dans le fichier Hybrid neural network.ipynb
import torch.nn as nn
import numpy as np
from HybridFunction import Hybrid
import qiskit
from qiskit.providers.aer import AerSimulator
import torch.nn.functional as F
import torch
class Net(nn.Module):
    def __init__(self, device):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 10)
        self.device = device
#         self.hybrid = Hybrid(qiskit.Aer.get_backend('aer_simulator'), 100, np.pi / 2)
        if device == "GPU":
            backend = AerSimulator(method = 'density_matrix', device = 'GPU')
            self.hybrid = [Hybrid(backend, 100, np.pi / 2, device) for _ in range(10)]
        else : 
            backend = AerSimulator(method = 'density_matrix')
            self.hybrid = [Hybrid(backend, 100, np.pi / 2, device) for _ in range(10)]
       

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = x.view(1, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.chunk(x,10,dim=1)
        x = tuple([hy(x_) for hy,x_ in zip(self.hybrid, x)])
        return torch.cat(x,-1)
        