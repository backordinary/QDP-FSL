# https://github.com/Stephen-Campbell-UTD/NM_Project_Quantum_Computing/blob/8a45f599c8c6d705d4435e253096ee18b1612c19/Qiskit/tutorials/qpe_examples.py
#%%
from qiskit.visualization.counts_visualization import plot_histogram
from tutorials.qpe import qpe
from qiskit import Aer, QuantumCircuit
from numpy import pi
# %%
simulator = Aer.get_backend('aer_simulator')
# %%
qpe_test_psi = QuantumCircuit(1)
qpe_test_psi.x(0)
def qpe_test_U(a : int):
  """
  control is 0th bit
  """
  qc = QuantumCircuit(1+1)
  qc.cp(a*pi/3, 0, 1) 
  return qc

qc_test =  qpe(12,qpe_test_psi, qpe_test_U)
# qc_test.draw()
results = simulator.run(qc_test).result()
counts = results.get_counts()
plot_histogram(counts)

# %%
