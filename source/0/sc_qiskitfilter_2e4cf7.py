# https://github.com/stroblme/hqsp-stqft/blob/c2f8f8964648578755d3938bf8658e4c834548e8/sc_qiskitFilter.py
from qft import get_fft_from_counts, loadBackend, qft_framework
from fft import fft_framework
from frontend import frontend, signal, transform

from qiskit.circuit.library import QFT as qiskit_qft

# --- Standard imports

# Importing standard Qiskit libraries and configuring account
from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit.compiler import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import *
# Loading your IBM Q account(s)
provider = IBMQ.load_account()

# --- Imports
from qiskit import QuantumCircuit, execute, BasicAer
from qiskit.tools.monitor import job_monitor

import math
from numpy import linalg as LA
import numpy as np
#%config jupy = 'svg' # Makes the images look nice


# --- Computation of the calibration matrix

from qiskit.ignis.mitigation.measurement import (complete_meas_cal,CompleteMeasFitter)
from qiskit import *




nQubits   = 4
nShots    = 2048



qr = QuantumRegister(nQubits)
meas_calibs, state_labels = complete_meas_cal(qr=qr, circlabel='mcal')
_, backend = loadBackend('ibmq_quito', True)
job = execute(meas_calibs, backend=backend, shots=1000)
# job_monitor(job, interval = 3)
cal_results = job.result()

meas_fitter = CompleteMeasFitter(cal_results, state_labels, circlabel='mcal')
print(meas_fitter.cal_matrix)







q = QuantumRegister(4,'q')

qc = QuantumCircuit(q)
# Normalize ampl, which is required for squared sum of amps=1
ys = signal(samplingRate=1000, amplification=1, duration=0, nSamples=2**nQubits)

ys.addFrequency(125)
ys.addFrequency(250)
y = ys.sample()

plotData = ys.show(subplot=[1,4,1], title='signal')

print("Processing FFT")

fft = transform(fft_framework)

y_hat, f = fft.forward(ys)
y_hat_ideal_p, f_p = fft.postProcess(y_hat, f)
plotData = fft.show(y_hat_ideal_p, f_p, subplot=[1,4,2], title="FFT (ref)")



# y.addFrequency(250)
ampls = y / np.linalg.norm(y)

# for 2^n amplitudes, we have n qubits for initialization
# this means that the binary representation happens exactly here

qc.initialize(ampls, [q[i] for i in range(nQubits)])

qc += qiskit_qft(num_qubits=nQubits, approximation_degree=0, do_swaps=True, inverse=False, insert_barriers=False, name='qft')

qc.measure_all()
qc = transpile(qc, backend, optimization_level=1) # opt level 0,1..3. 3: heaviest opt
job = execute(qc, backend, shots = nShots)
#job = execute(qc, BasicAer.get_backend('qasm_simulator'), shots = shots)

result = job.result()
# print(result.get_counts())





genTransform = transform(None)





y_hat = np.array(get_fft_from_counts(result.get_counts(), nQubits))

f = genTransform.calcFreqArray(ys, y_hat)
y_hat_sim_p, f_p = genTransform.postProcess(y_hat, f)
plotData = genTransform.show(y_hat_sim_p, f_p, subplot=[1,4,3], title=f"qft_sim_n")

print(y_hat)








# Get the filter object
meas_filter = meas_fitter.filter

# Results with mitigation
mitigated_results = meas_filter.apply(result)
mitigated_counts = mitigated_results.get_counts(0)

# print(mitigated_counts)



y_hat = np.array(get_fft_from_counts(mitigated_counts, nQubits))

f = genTransform.calcFreqArray(ys, y_hat)
y_hat_sim_p, f_p = genTransform.postProcess(y_hat, f)
plotData = genTransform.show(y_hat_sim_p, f_p, subplot=[1,4,4], title=f"qft_sim_n_f")

print(y_hat)

frontend.primeTime()