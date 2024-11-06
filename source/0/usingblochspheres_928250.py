# https://github.com/woahdas/Programs-in-Quantum-Computing/blob/cca0e80371dc6233a3c17198b5de296fe8dbf988/UsingBlochSpheres.py
import qiskit as q
from qiskit.tools.visualization import plot_bloch_multivector
from qiskit.visualization import plot_histogram
from matplotlib import style
style.use("dark_background")

statevector_simulator = q.Aer.get_backend("statevector_simulator")
qasm_sim = q.Aer.get_backend("qasm_simulator")

def do_job(circuit):
    result = q.execute(circuit, backend=statevector_simulator).result()
    statevec = result.get_statevector()
    
    n_qubits = circuit.n_qubits #number of qubits
    
    circuit.measure([i for i in range(n_qubits)],[i for i in range(len(circuit.clbits))])
                                                                    ####^^^ number of classical bits
    qasm_job = q.execute(circuit, backend=qasm_sim, shots=1024).result()
    counts = qasm_job.get_counts()
    return statevec, counts

circuit = q.QuantumCircuit(2,2)
statevec, counts = do_job(circuit)

plot_bloch_multivector(statevec)
plot_histogram([counts], legend=["output"])#

circuit = q.QuantumCircuit(2,2)
circuit.h(0)
circuit.cx(0, 1)
statevec, counts = do_job(circuit)
plot_bloch_multivector(statevec)

circuit = q.QuantumCircuit(2,2)
circuit.h(0)
circuit.h(1)
circuit.cx(0, 1)
statevec, counts = do_job(circuit)
plot_bloch_multivector(statevec)
