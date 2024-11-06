# https://github.com/KobeVK/qiskit--simple-circuits/blob/2c6a769fffc6f5c1496eb3ce7b1593ae4bed654d/qiskit/multi-qubit.py
# %%
import qiskit as q
from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit.compiler import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import *
from matplotlib import style
style.use("dark_background")
#provider = IBMQ.load_account()

def execute_circuit(quantum_circuit):
    simulator = Aer.get_backend('qasm_simulator')
    result = execute(quantum_circuit, simulator, shots=1024).result()
    results = result.get_counts(quantum_circuit)
    circuit_diagram = quantum_circuit.draw()
    histogram = plot_histogram(results)
    return results, circuit_diagram, histogram

def execute_circuit_sv(quantum_circuit):
    #Create a state vector simulator
    statevector_simulator = Aer.get_backend('statevector_simulator')
    #Execute the circuit on the simulator
    result = execute(quantum_circuit, statevector_simulator).result()
    #Assign state vector results
    statevector_results  = result.get_statevector(quantum_circuit)
    #Draw the circuit diagram
    circuit_diagram = quantum_circuit.draw()
    #Draw the Qsphere 
    q_sphere = plot_state_qsphere(statevector_results)
    #Return the results, circuit diagram, and QSphere		
    return statevector_results, circuit_diagram, q_sphere

#X-gate 
#Create the single qubit circuit
qc = QuantumCircuit(3)
qc.ccx(0,1,2)
qc.measure_all()
result, img, qsphere = execute_circuit(qc)

#decomposing:
# qc_decomposed = qc.decompose()
# qc_decomposed.draw()

# %%
# Statevector([1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
#              0.+0.j],
#             dims=(2, 2, 2))

#qasm_simulator {'000': 1024}
