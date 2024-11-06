# https://github.com/Thakkar-meet/Quantum-Computing/blob/1b921565abe4993cd8645531e1d7030f1b19f3bb/superdense_coding.py
from qiskit import QuantumCircuit, Aer, assemble
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt


def draw(qc):
    qc.draw(output='mpl')
    plt.show()


def create_entangled_state():
    qc = QuantumCircuit(2)
    qc.h(1)
    qc.cx(1,0)
    return qc


def encode_message(qc,qubit,message):
    if len(message)!=2 or not set(message).issubset({"0","1"}):
        raise ValueError(f"message '{message}' is invalid")
    if message[0] == "1":
        qc.z(qubit)
    if message[1] == "1":
        qc.x(qubit)
    return qc


def decode_message(qc):
    qc.cx(1,0)
    qc.h(1)
    return qc


qc = create_entangled_state()
qc.barrier()

message="11"
encode_message(qc,1,message)
qc.barrier()

decode_message(qc)

qc.measure_all()
draw(qc)

aer_sim = Aer.get_backend("aer_simulator")
q_obj = assemble(qc)
result = aer_sim.run(q_obj).result()
counts = result.get_counts(qc)
print(counts)
plot_histogram(counts)
plt.show()
