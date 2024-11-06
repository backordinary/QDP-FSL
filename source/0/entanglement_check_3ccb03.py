# https://github.com/HypsoHypso/QuantumComputingScripts/blob/53134783facba7d5972449b7ebb479d9439f3a98/entanglement_check.py
##############################################################
#   Imports                                                  #
##############################################################

# %matplotlib inline ### to be enabled on qiskit notebook
import qiskit

from qiskit import (
    IBMQ,
    ClassicalRegister,
    QuantumCircuit,
    QuantumRegister,
    QuantumCircuit,
    execute,
    Aer,
)

from qiskit.visualization import plot_histogram


##############################################################
#   Circuit                                                  #
##############################################################

circuit = QuantumCircuit(2, 2)  # 2 quantum bits and 2 classical bits
circuit.h(0)                    # Hadamard Gate
circuit.cx(0, 1)                # CNOT Gate
circuit.measure([0, 1], [0, 1]) # Measurement

circuit.draw()
print(circuit)
# circuit.draw(output="mpl", filename="circuit.png") # to be enabled on qiskit notebook


##############################################################
#   Second Section :                                         #
#   Execution on a quantum simulator                         #
#   Perfectly quantum results                                #
#   Maximum entanglement                                     #
##############################################################

simulator = Aer.get_backend("qasm_simulator")  # Setting up the simulator
job = execute(circuit, simulator, shots=1000)  # Runs on 1000 trials
result = job.result()                # Get the result
counts = result.get_counts(circuit)  # Get the result

print("\nResults :", counts)
plot_histogram(counts)
for qubit, percentage in counts.items():
    percentage *= 0.1
    print(f"Qubit : {qubit}  =>  Percentage : {round(percentage, 3)}%")

print(("Non-maximal entanglement", "Maximum entanglement")[counts.get("01") == counts.get("10")])


##############################################################
#   Third Section :                                          #
#   Execution on a real quantum computer                     #
#   Only executable on IBM Quantum Lab                       #
#   Some result errors                                       #
##############################################################

provider = IBMQ.get_provider(group="open")
device = provider.get_backend(
    """YOUR CHOSEN QUANTUM COMPUTER NAME"""
)  # Declaration of a quantum computer located at a certain place
job_exp = execute(circuit, device, shots=1024)  # Runs on 1024 trials

result_exp = job_exp.result()                # Get the result
counts_exp = result_exp.get_counts(circuit)  # Get the result (qubits, percentage)
print("\nExperimental Results :", counts_exp)
plot_histogram(counts_exp)
for qubit, percentage in counts_exp.items():
    percentage *= 0.1
    print(f"Qubit : {qubit}  =>  Percentage : {round(percentage, 3)}%")

print(("Non-maximal entanglement", "Maximum entanglement")[counts_exp.get("01") == counts_exp.get("10")])
