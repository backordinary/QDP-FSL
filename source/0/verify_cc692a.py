# https://github.com/Stephen-Campbell-UTD/NM_Project_Quantum_Computing/blob/676e0dfe7f04dfe36e687bb95b5479f752f14218/Qiskit/verify.py
#%%
from qiskit import Aer, QuantumCircuit
from qiskit.visualization.counts_visualization import plot_histogram
from qclib import qft, iqft, qpe
from qclib import plot_current_bloch_state
from numpy import pi


#%%
# Determine the QFT3 of 5
qc = QuantumCircuit(3)
qc.x(0)
# qc.x(2)
plot_current_bloch_state(qc)
qc.compose(qft(3), qubits=range(3), inplace=True)
plot_current_bloch_state(qc)
qc.draw()
qc.barrier()
qc.compose(iqft(3), qubits=range(3), inplace=True)
plot_current_bloch_state(qc)

# %%
# Determine the QFT3 of 5
# Guess the Phase 

qpe_test_psi = QuantumCircuit(1)
qpe_test_psi.x(0)
phase = 8 
def qpe_test_U(a : int):
  """
  control is 0th bit
  """
  qc = QuantumCircuit(1+1)
  qc.cp(2*pi*a/phase, 0, 1) 
  return qc

estimation_bits = 4 
qc_test =  qpe(estimation_bits,qpe_test_psi, qpe_test_U)
simulator = Aer.get_backend('aer_simulator')
results = simulator.run(qc_test).result()
counts = results.get_counts()
plot_histogram(counts)
# %%
qft(3).draw()

# %%
