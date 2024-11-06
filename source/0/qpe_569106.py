# https://github.com/Stephen-Campbell-UTD/NM_Project_Quantum_Computing/blob/8a45f599c8c6d705d4435e253096ee18b1612c19/Qiskit/tutorials/qpe.py
#%%
from qiskit import QuantumCircuit
from numpy import pi
from qiskit.visualization import plot_histogram
from qiskit import Aer
from qft_playground import gen_iqft_circuit, iqft
#%%

#Exampek

def example3bit():
  # T gate 
  # T|1> -> exp(pi*i/4)|1>
  # T = cp(pi/4)
  # theta = 1/8
  qpe = QuantumCircuit(4,3)
  qpe.h(0)
  qpe.h(1)
  qpe.h(2)

  #target qubit
  qpe.x(3)
  for _ in range(2**0):
    qpe.cp(pi/4, 0,3)
  for _ in range(2**1):
    qpe.cp(pi/4, 1,3)
  for _ in range(2**2):
    qpe.cp(pi/4, 2,3)

  qpe.barrier()
  qpe = iqft(qpe,3)
  qpe.barrier()
  qpe.measure(range(3), range(3))
  return qpe



example3bit().draw()
# %%
# %%

sim = Aer.get_backend('aer_simulator')
results = sim.run(example3bit()).result()
counts = results.get_counts()
plot_histogram(counts)

# %%



def qpe(t : int, psi : QuantumCircuit, U) -> int: 
  """
  t : num counting qubits
  psi : quantum circuit in eigenstate (with proper number of qubits)
  U(a: int) : Returns quantum circuit that represent unitary operator to power a
  """
  #put counting bits into 0 degree phase (superposition)
  main_circuit = QuantumCircuit(t+psi.num_qubits, t)

  eigenstate_qubit_indices = list(range(t,t+psi.num_qubits))
  main_circuit.compose(psi,qubits=eigenstate_qubit_indices, inplace=True)


  for i in range(t):
    main_circuit.h(i)
  

  main_circuit.barrier() 
  #Apply Controlled Unitary Operations
  for i in range(t):
    op = U(2**i)
    main_circuit.compose(op,qubits=[i,*eigenstate_qubit_indices ], inplace=True)


  main_circuit.barrier() 
  #Inverse QFT
  iqft_circuit = gen_iqft_circuit(t)
  main_circuit.compose(iqft_circuit,qubits=list(range(t)), inplace=True)

  main_circuit.barrier() 
  # Measure Counting Bits
  main_circuit.measure(list(range(t)), list(range(t)))


  return main_circuit

qpe_test_psi = QuantumCircuit(1)
qpe_test_psi.x(0)
def qpe_test_U(a : int):
  """
  control is 0th bit
  """
  qc = QuantumCircuit(1+1)
  qc.cp(a*pi/4, 0, 1) 
  return qc

circ = qpe(3, qpe_test_psi, qpe_test_U)
circ.draw()
# %%

# sim = Aer.get_backend('aer_simulator')
# results = sim.run(circ).result()
# counts = results.get_counts()
# plot_histogram(counts)
# %%
