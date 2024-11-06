# https://github.com/twoadave/dgayowsky-code-snippets/blob/47caced1fa38adceb40637f73bc4276eadbaa237/qiskit-tutorials/basic_noise.py
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 2 2023

@author: David

Qiskit tutorial: basic noise.
"""

#######################################################################

import numpy as np
import matplotlib.pyplot as plt
from qiskit import IBMQ, transpile
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.tools.visualization import plot_histogram

#######################################################################

#We will use real noise data for an IBM Quantum device using the data 
# stored in Qiskit Terra. Specifically, in this tutorial, the device is ibmq_vigo.
#The Qiskit Aer device noise model automatically generates a simplified noise model 
# for a real device. This model is generated using the calibration information 
# reported in the BackendProperties of a device and takes into account various params.

from qiskit.providers.fake_provider import FakeVigo
device_backend = FakeVigo()

#Now we construct a test circuit to compare the output of the real device with the 
# noisy output simulated on the Qiskit Aer AerSimulator.

# Construct quantum circuit
circ = QuantumCircuit(3, 3)
circ.h(0)
circ.cx(0, 1)
circ.cx(1, 2)
circ.measure([0, 1, 2], [0, 1, 2])

sim_ideal = AerSimulator()

# Execute and get counts
result = sim_ideal.run(transpile(circ, sim_ideal)).result()
counts = result.get_counts(0)
plot_histogram(counts, title='Ideal counts for 3-qubit GHZ state')
plt.show()

#Now we create our noise simulator:
#By storing the device properties in vigo_simulator, we ensure that the appropriate 
# basis gates and coupling map are used when compiling circuits for simulation, 
# thereby most closely mimicking the gates that will be executed on a real device.
sim_vigo = AerSimulator.from_backend(device_backend)

# Transpile the circuit for the noisy basis gates.
#If transpilation is skipped, noise from the device noise model will not be applied 
# to gates in the circuit that are supported by the simulator, but not supported by the 
# mimicked backend.
tcirc = transpile(circ, sim_vigo)

# Execute noisy simulation and get counts
result_noise = sim_vigo.run(tcirc).result()
counts_noise = result_noise.get_counts(0)
plot_histogram(counts_noise,
               title="Counts for 3-qubit GHZ state with device noise model")
plt.show()



