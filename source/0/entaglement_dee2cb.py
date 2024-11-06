# https://github.com/keshavbharadwaj/Quantum-circuits-in-python/blob/0c620f78180e6515554855ae333de496f9670530/entaglement/entaglement.py
import qiskit as q
from qiskit import Aer,execute
from qiskit import IBMQ
from qiskit.tools.visualization import plot_histogram,plot_bloch_multivector
from qiskit.tools.monitor import job_monitor
import matplotlib
statevector=q.Aer.get_backend("statevector_simulator")
qasm_sim=q.Aer.get_backend("qasm_simulator")
def do_jobs(circuit):
    result=execute(circuit,backend=statevector).result()
    statevec = result.get_statevector()
    n_qubits=circuit.num_qubits
    circuit.measure([i for i in range(n_qubits)],[i for i in range(n_qubits)])

    result2=execute(circuit,backend=qasm_sim).result()
    counts = result2.get_counts()
    return statevec,counts

circuit=q.QuantumCircuit(3,3)
circuit.h(0)
circuit.cx(0,1)
circuit.cx(1,2)
print(circuit)
statevec,counts=do_jobs(circuit)
plot_bloch_multivector(statevec).show()
#plot_histogram(counts).show()
