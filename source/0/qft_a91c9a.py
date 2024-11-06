# https://github.com/rsathyak/qml_masters_thesis/blob/2a6e4ce5db7e5908527f46acf05eb36b617c1652/asu_ke_project/experiments/qiskit_exp/qft.py
from decorators.compute_process_time import compute_process_time
from qiskit import execute, transpile
from qiskit.circuit.library import QFT
from qiskit.providers.aer import AerSimulator

EXECUTION_RUN = 5
SIM_EXEC_LOOP = 50
CRE_LOOP = 10
DEFAULT_SHOTS = 1024

def get_simulator(simulator,  shots = DEFAULT_SHOTS):
  return AerSimulator(method=simulator, shots=shots)

@compute_process_time(EXECUTION_RUN, CRE_LOOP)
def create_transpile_qft(qubits, simulator, inverse = False):
  qft = QFT(num_qubits=qubits, approximation_degree=0, do_swaps=True, inverse=inverse, insert_barriers=False)
  qft.measure_all()
  tqft = transpile(qft, simulator)
  return tqft
    


@compute_process_time(EXECUTION_RUN, SIM_EXEC_LOOP)
def simulate_qft(tqft, sim):
  counts = execute(tqft, sim).result().get_counts(0)
  return tqft.depth(), counts 
    
   