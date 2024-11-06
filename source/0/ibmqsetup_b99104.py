# https://github.com/Oelfol/Dynamics/blob/ed8b55297aa488444e7cd12e1352e4f1b88b525f/HeisenbergCodes/IBMQSetup.py
###########################################################################
# IBMQSetup.py
# Part of HeisenbergCodes
# Updated January '21
#
# Code to retrieve IBMQ account and initiate settings.
###########################################################################

from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer import noise
from qiskit import Aer, IBMQ

# Sometimes, this command is necessary (if console says no account in session):
# IBMQ.disable_account()
# Sometimes Provider: IBMQ.enable_account(TOKEN, hub='ibm-q', group='open', project='main')

#TOKEN = 

#provider =  
simulator = Aer.get_backend('qasm_simulator')

# 8192 is standard max for real runs
shots = 50000


class ibmqSetup():
    def __init__(self, sim=True, dev_name='', shots=shots):
        self.sim = sim
        self.shots = shots
        self.dev_name = dev_name

    def get_noise_model(self):
        # regular noise model from the backend

        device = provider.get_backend(self.dev_name)
        properties = device.properties()
        gate_lengths = noise.device.parameters.gate_length_values(properties)
        noise_model = NoiseModel.from_backend(properties, gate_lengths=gate_lengths)
        basis_gates = noise_model.basis_gates
        coupling_map = device.configuration().coupling_map
        return device, noise_model, basis_gates, coupling_map

    def get_device(self):
        return provider.get_backend(self.dev_name)

    def get_simulator(self):
        return Aer.get_backend('qasm_simulator')
