# https://github.com/paniash/progs/blob/b7f58efeb5b5c7942af6b8af12611bdbc1a52840/qiskit/vqe.py
#%%
from qiskit import *
from qiskit.visualization import *
import numpy as np

#%%
np.random.seed(999999)
target_distr = np.random.rand(2)
target_distr /= sum(target_distr)

#%%
def get_var_form(params):
    qr = QuantumRegister(1, name='q')
    cr = ClassicalRegister(1, name='c')
    qc = QuantumCircuit(qr, cr)
    qc.u3(params[0], params[1], params[2], qr[0])
    qc.measure(qr, cr[0])
    return qc

#%%
backend = Aer.get_backend('qasm_simulator')
shots = 10000

def get_probability_distribution(counts):
    output_dis
