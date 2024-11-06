# https://github.com/stiandb/QuantumComputing/blob/f7273f435d963f9d0861dc2e1dfb876558a9e2c0/Scripts/main.py
import matplotlib.pylab as plt
from qcPairingModel import QCPairing
import qiskit as qk
from qiskit import IBMQ
from qiskit import Aer, IBMQ, execute
from qiskit.providers.aer import noise
import numpy as np
import pickle
from hamiltonian import *



with open("properties.txt", "rb") as fp:   #Get properties from IBMQ 16 qubit Melbourne
	properties = pickle.load(fp)


n_pairs = 1
n_basis = 2
H = hamiltonian(n_pairs,n_basis,1,1)
vals,vecs = np.linalg.eigh(H[0])
print('FCI eigenvalues for one pair and four spin orbitals',vals)

n_pairs = 2
n_basis = 2
H = hamiltonian(n_pairs,n_basis,1,1)
vals,vecs = np.linalg.eigh(H[0])
print('FCI eigenvalues for two pairs and four spin orbitals',vals)

for noise_bool in [False,True]:
	if noise_bool:
		print('With noise')
	else:
		print('Without noise')
		
	for n_work in [4,6,8]:
		if noise_bool:
			noise_model = noise.device.basic_device_noise_model(properties)
		else:
			noise_model = None
		n_simulation = 4
		Emax= 2
		dt = 0.005
		g = 1
		delta=1
		t= 100*dt
		Pairing1 = QCPairing(n_work,n_simulation,delta=delta,g=g,dt=dt,Emax=Emax,noise_model=noise_model) #Initialization
		measurements = Pairing1.run_simulation(t = t,shots = 1000) #Run simulation
		x,y = Pairing1.sort_measurements(measurements) #Calculates how many times a single eigenvalue is measured
		plt.plot(x,y)
		plt.xlabel('Eigenvalue')
		plt.ylabel('Times measured')
		plt.show()
		












