# https://github.com/hykelvinlee42/quantum-teleportation-circuit/blob/f8220855a10ef5856cee61f6c6b59b263e0e1d25/main.py
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.extensions import Initialize
from qiskit import IBMQ, Aer, transpile, assemble
from qiskit.visualization import plot_histogram, plot_bloch_multivector
import matplotlib.pyplot as plt


def entangle_qubits(circuit, qubits_ids):
    # create entangle pair of qubits (bell pair)
    circuit.h(qubits_ids[0])  # Put first qubit into a \+> state
    circuit.cx(qubits_ids[0], qubits_ids[1])  # Apply CNOT gate with first qubit as control and second qubit as target


def sender_gate(circuit, psi_qubit_id, sender_qubit_id):
    circuit.cx(psi_qubit_id, sender_qubit_id)
    circuit.h(psi_qubit_id)


def meansure_and_send(circuit, psi_qubit_id, sender_qubit_id):
    circuit.barrier()
    circuit.measure(psi_qubit_id, 0)  # measure and assign to first classical bit
    circuit.measure(sender_qubit_id, 1)  # measure and assign to second classical bit


def receiver_gate(circuit, receiver_qubit_id, classical_registerz, classical_registerx):
    circuit.x(receiver_qubit_id).c_if(classical_registerx, 1)
    circuit.z(receiver_qubit_id).c_if(classical_registerz, 1)


def create_teleportation_circuit(simulation="qasm"):
    # setup qubits and classical bits for quantum teleportation
    quantum_register = QuantumRegister(3, name="q")
    classical_registerz = ClassicalRegister(1, name="crz")
    classical_registerx = ClassicalRegister(1, name="crx")
    teleportation_circuit = QuantumCircuit(quantum_register, classical_registerz, classical_registerx)

    # sender owns q0 and receiver owns q1
    sender_qubit_id = 1
    receiver_qubit_id = 2
    third_party_qubit_id = 0

    psi = [0.45936609+0.56725257j, 0.40025413+0.55407937j]  # create qubit state
    # plot_bloch_multivector(psi)
    init_gate = Initialize(psi)
    init_gate.label = "init"
    inverse_init_gate = init_gate.gates_to_uncompute()

    teleportation_circuit.append(init_gate, [third_party_qubit_id])
    teleportation_circuit.barrier()

    # The Quantum Teleportation Protocol, step 1
    entangle_qubits(teleportation_circuit, [sender_qubit_id, receiver_qubit_id])  # entangle qubits q0 and q1

    # The Quantum Teleportation Protocol, step 2
    teleportation_circuit.barrier()
    sender_gate(teleportation_circuit, third_party_qubit_id, sender_qubit_id)

    # The Quantum Teleportation Protocol, step 3
    meansure_and_send(teleportation_circuit, third_party_qubit_id, sender_qubit_id)

    # The Quantum Teleportation Protocol, step 4
    receiver_gate(teleportation_circuit, receiver_qubit_id, classical_registerz, classical_registerx)

    if simulation == "sv":
        # Statevector Simulator
        statevector_simulation(teleportation_circuit)
    elif simulation == "qasm":
        # QASM qasm_simulator
        teleportation_circuit.append(inverse_init_gate, [receiver_qubit_id])
        classical_register_result = ClassicalRegister(1)
        teleportation_circuit.add_register(classical_register_result)
        teleportation_circuit.measure(receiver_qubit_id, 2)
        qasm_simulation(teleportation_circuit)

    teleportation_circuit.draw(output="mpl")
    plt.show()


def statevector_simulation(circuit):
    sv_sim = Aer.get_backend("statevector_simulator")
    qobj = assemble(circuit)
    out_vector = sv_sim.run(qobj).result().get_statevector()
    plot_bloch_multivector(out_vector)


def qasm_simulation(circuit):
    qasm_sim = Aer.get_backend("qasm_simulator")
    t_qc = transpile(circuit, qasm_sim)
    qobj = assemble(t_qc)
    counts = qasm_sim.run(qobj).result().get_counts()
    plot_histogram(counts)


if __name__ == "__main__":
    create_teleportation_circuit()
