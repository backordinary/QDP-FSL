# https://github.com/paniash/progs/blob/b7f58efeb5b5c7942af6b8af12611bdbc1a52840/qiskit/Noise_Simulation.py
#!/usr/bin/env python
# coding: utf-8

import numpy as np
from qiskit import execute, QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Kraus, SuperOp
from qiskit.providers.aer import QasmSimulator
from qiskit.tools.visualization import plot_histogram

#%%
# Import from Qiskit Aer noise module
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise import QuantumError, ReadoutError
from qiskit.providers.aer.noise import pauli_error
from qiskit.providers.aer.noise import depolarizing_error
from qiskit.providers.aer.noise import thermal_relaxation_error


#%%
import qiskit
print(qiskit.__version__)
print(np.__version__)


# # Your Circuit

# This is an Example Circuit.. Here Your circuit will come in

#%%
n_qubits = 4
circ = QuantumCircuit(n_qubits, n_qubits)

#%%
# Test Circuit
circ.h(0)
for qubit in range(n_qubits - 1):
    circ.cx(qubit, qubit + 1)
circ.measure(range(4), range(4))
print(circ)


#%%
# Ideal simulator and execution
ideal_simulator = QasmSimulator()
job = execute(circ, ideal_simulator)
result_ideal = job.result()
plot_histogram(result_ideal.get_counts(0))


# # Noise Simulation _ 1

#%%
# Example error probabilities
p_reset = 0.03
p_meas = 0.1
p_gate1 = 0.05

#%%
# QuantumError objects
error_reset = pauli_error([('X', p_reset), ('I', 1 - p_reset)])
error_meas = pauli_error([('X',p_meas), ('I', 1 - p_meas)])
error_gate1 = pauli_error([('X',p_gate1), ('I', 1 - p_gate1)])
error_gate2 = error_gate1.tensor(error_gate1)

# Add errors to noise model
noise_bit_flip = NoiseModel()
noise_bit_flip.add_all_qubit_quantum_error(error_reset, "reset")
noise_bit_flip.add_all_qubit_quantum_error(error_meas, "measure")
noise_bit_flip.add_all_qubit_quantum_error(error_gate1, ["u1", "u2", "u3"])
noise_bit_flip.add_all_qubit_quantum_error(error_gate2, ["cx"])

print(noise_bit_flip)


# In[6]:

#%%
# Run the noisy simulation
noisy_simulator = QasmSimulator(noise_model=noise_bit_flip)
job = execute(circ, noisy_simulator)
result_bit_flip = job.result()
counts_bit_flip = result_bit_flip.get_counts(0)

#%%
# Plot noisy output
plot_histogram(counts_bit_flip)


# # More Realistic Noisy  Simulation

# ## Note:
# Numpy Version : 1.19.3 and if this cell gives you an error.. just run it again.. the error will be resolved.. there is some bug in the numpy library

# In[8]:

#%%
# T1 and T2 values for qubits 0-3
T1s = np.random.normal(50e3, 10e3, 4) # Sampled from normal distribution mean 50 microsec
T2s = np.random.normal(70e3, 10e3, 4)  # Sampled from normal distribution mean 50 microsec

# Truncate random T2s <= T1s
T2s = np.array([min(T2s[j], 2 * T1s[j]) for j in range(4)])

#%%
# Instruction times (in nanoseconds)
time_u1 = 0   # virtual gate
time_u2 = 50  # (single X90 pulse)
time_u3 = 100 # (two X90 pulses)
time_cx = 300
time_reset = 1000  # 1 microsecond
time_measure = 1000 # 1 microsecond

#%%
# QuantumError objects
errors_reset = [thermal_relaxation_error(t1, t2, time_reset)
                for t1, t2 in zip(T1s, T2s)]
errors_measure = [thermal_relaxation_error(t1, t2, time_measure)
                  for t1, t2 in zip(T1s, T2s)]
errors_u1  = [thermal_relaxation_error(t1, t2, time_u1)
              for t1, t2 in zip(T1s, T2s)]
errors_u2  = [thermal_relaxation_error(t1, t2, time_u2)
              for t1, t2 in zip(T1s, T2s)]
errors_u3  = [thermal_relaxation_error(t1, t2, time_u3)
              for t1, t2 in zip(T1s, T2s)]
errors_cx = [[thermal_relaxation_error(t1a, t2a, time_cx).expand(
             thermal_relaxation_error(t1b, t2b, time_cx))
              for t1a, t2a in zip(T1s, T2s)]
               for t1b, t2b in zip(T1s, T2s)]

#%%
# Add errors to noise model
noise_thermal = NoiseModel()
noise_thermal.add_quantum_error(errors_reset[0], "reset", [0])
noise_thermal.add_quantum_error(errors_measure[0], "measure", [0])
noise_thermal.add_quantum_error(errors_u1[0], "u1", [0])
noise_thermal.add_quantum_error(errors_u2[0], "u2", [0])
noise_thermal.add_quantum_error(errors_u3[0], "u3", [0])
noise_thermal.add_quantum_error(errors_cx[0][0], "cx", [0, 0])
noise_thermal.add_quantum_error(errors_cx[0][1], "cx", [0, 1])
noise_thermal.add_quantum_error(errors_cx[0][2], "cx", [0, 2])
noise_thermal.add_quantum_error(errors_cx[0][3], "cx", [0, 3])
noise_thermal.add_quantum_error(errors_reset[1], "reset", [1])
noise_thermal.add_quantum_error(errors_measure[1], "measure", [1])
noise_thermal.add_quantum_error(errors_u1[1], "u1", [1])
noise_thermal.add_quantum_error(errors_u2[1], "u2", [1])
noise_thermal.add_quantum_error(errors_u3[1], "u3", [1])
noise_thermal.add_quantum_error(errors_cx[1][0], "cx", [1, 0])
noise_thermal.add_quantum_error(errors_cx[1][1], "cx", [1, 1])
noise_thermal.add_quantum_error(errors_cx[1][2], "cx", [1, 2])
noise_thermal.add_quantum_error(errors_cx[1][3], "cx", [1, 3])
noise_thermal.add_quantum_error(errors_reset[2], "reset", [2])
noise_thermal.add_quantum_error(errors_measure[2], "measure", [2])
noise_thermal.add_quantum_error(errors_u1[2], "u1", [2])
noise_thermal.add_quantum_error(errors_u2[2], "u2", [2])
noise_thermal.add_quantum_error(errors_u3[2], "u3", [2])
noise_thermal.add_quantum_error(errors_cx[2][0], "cx", [2, 0])
noise_thermal.add_quantum_error(errors_cx[2][1], "cx", [2, 1])
noise_thermal.add_quantum_error(errors_cx[2][2], "cx", [2, 2])
noise_thermal.add_quantum_error(errors_cx[2][3], "cx", [2, 3])
noise_thermal.add_quantum_error(errors_reset[3], "reset", [3])
noise_thermal.add_quantum_error(errors_measure[3], "measure", [3])
noise_thermal.add_quantum_error(errors_u1[3], "u1", [3])
noise_thermal.add_quantum_error(errors_u2[3], "u2", [3])
noise_thermal.add_quantum_error(errors_u3[3], "u3", [3])
noise_thermal.add_quantum_error(errors_cx[3][0], "cx", [3, 0])
noise_thermal.add_quantum_error(errors_cx[3][1], "cx", [3, 1])
noise_thermal.add_quantum_error(errors_cx[3][2], "cx", [3, 2])
noise_thermal.add_quantum_error(errors_cx[3][3], "cx", [3, 3])

print(noise_thermal)


# In[9]:

#%%
# Run the noisy simulation
thermal_simulator = QasmSimulator(noise_model=noise_thermal)
job = execute(circ, thermal_simulator)
result_thermal = job.result()
counts_thermal = result_thermal.get_counts(0)

# Plot noisy output
plot_histogram(counts_thermal)
