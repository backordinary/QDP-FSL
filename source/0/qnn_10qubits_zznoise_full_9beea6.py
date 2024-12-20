# https://github.com/ghuioio/effective_dimension_test/blob/e652138b1544ce8c403d53ad2ecd35ecb557674d/Noise_plots/eigenvalue_distribution/generate_data/qnn_10qubits_ZZnoise_full.py
from functions import QuantumNeuralNetwork, EffectiveDimension
from qiskit.aqua import QuantumInstance
from qiskit import IBMQ, Aer
from qiskit.providers.aer.noise.noise_model import NoiseModel
import numpy as np
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes

TOKEN = 'insert token here'
IBMQ.save_account(TOKEN, overwrite=True)
provider = IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q-internal', group='deployed', project='default')
backend_name = 'ibmq_montreal'
backend_ibmq = provider.get_backend(backend_name)
properties = backend_ibmq.properties()
coupling_map = backend_ibmq.configuration().coupling_map
noise_model = NoiseModel.from_backend(properties)
layout = [1, 2, 3, 5, 8, 11, 14, 13, 12, 10]  # might need to change
qi_ibmq_noise_model = QuantumInstance(backend=Aer.get_backend('qasm_simulator'),
                                       noise_model=noise_model, optimization_level=0, shots=8000,
                                       seed_transpiler=2, initial_layout=layout)
qi = qi_ibmq_noise_model
compile_config = {'initial_layout': layout,
                  'seed_transpiler': 2,
                  'optimization_level': 3
                  }
n = [1000, 2000, 8000, 10000, 40000, 60000, 100000, 150000, 200000, 500000, 1000000, 10000000, 10000000000, 10000000000000]
qubits = 10
fm = ZZFeatureMap(qubits, reps=2, entanglement='full')
varform = RealAmplitudes(qubits, reps=9, entanglement='full')
qnet = QuantumNeuralNetwork(fm, varform)
ed = EffectiveDimension(qnet, 100, 100)
fhat, _ = ed.get_fhat()
effdim = ed.eff_dim(fhat, n)
np.save('10qubits_fhats_noise_linearZZ_full.npy', fhat)
np.save('10qubits_effective_dimension_noise_linearZZ_full.npy', effdim)