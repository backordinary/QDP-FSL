# https://github.com/pdebus/IBMQuantumChallenge2020/blob/366552f9f8129f73d5483f8c9d7f9df1afb8d929/src/circuit_parts/qram_implementations.py
from qiskit import *
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import IBMQ, Aer, execute

import matplotlib.pyplot as plt


def qRAM_example(address_size, data_size):
    address = QuantumRegister(address_size)
    data = QuantumRegister(data_size)
    qc = QuantumCircuit(address, data)

    # address 0 -> data = 1
    qc.x([address[0], address[1]])
    qc.ccx(address[0], address[1], data[2])
    qc.x([address[0], address[1]])

    # address 1 -> data = 2
    qc.x(address[0])
    qc.ccx(address[0], address[1], data[1])
    qc.x(address[0])

    # address 2 -> data = 5
    qc.x(address[1])
    qc.ccx(address[0], address[1], data[2])
    qc.ccx(address[0], address[1], data[0])
    qc.x(address[1])

    # address 3 -> data = 7
    qc.ccx(address[0], address[1], data[2])
    qc.ccx(address[0], address[1], data[1])
    qc.ccx(address[0], address[1], data[0])

    qram = qc.to_gate()
    qram.name = "$qRAM$"
    return qram


def test_qram_write():
    print("\n===Write and measure QRAM===")
    address = QuantumRegister(2, name="address")
    data = QuantumRegister(3, name="data")
    c = ClassicalRegister(5, name="cbit")

    qc = QuantumCircuit(address, data, c)

    qc.h(address)
    qc.append(qRAM_example(2, 3), address[:] + data[:])

    # Check the qRAM　status
    qc.measure(address[0:2], c[0:2])
    qc.measure(data[0:3], c[2:5])

    # Reverse the output string.
    qc = qc.reverse_bits()

    # backend = provider.get_backend('ibmq_qasm_simulator')
    backend = Aer.get_backend('qasm_simulator')
    job = execute(qc, backend=backend, shots=8000, seed_simulator=12345)
    # job = execute(qc, backend=backend, shots=8192)
    result = job.result()
    count = result.get_counts()
    for string, counts in dict(sorted(count.items())).items():
        print(f"Memory Cell: {string[:2]} \tContent: {string[2:]} \t Counts:{counts}")

    qc.draw(output='mpl')
    plt.show()


def test_qram_grover_read():
    from diffusers import diffuser
    print("\n===Grover Read from QRAM===")
    print("Searching number 7 in RAM (Stored in 11)")
    address = QuantumRegister(2, name="address")
    data = QuantumRegister(3, name="data")
    output = QuantumRegister(1, name="output")
    c = ClassicalRegister(5, name="cbit")

    qc = QuantumCircuit(address, data, output, c)
    qram = qRAM_example(2, 3)

    # Initialization
    qc.h(address)  # |++>

    qc.x(output)
    qc.h(output)  # |->

    qc.append(qram, address[:] + data[:])

    # Oracle
    # qc.x([data[2],data[0]])
    # qc.mct(data, output)
    # qc.x([data[2],data[0]])
    qc.ccx(data[0], data[1], output)

    qc.append(qram, address[:] + data[:])

    # Diffuser
    qc.append(diffuser(2), address)

    # Check the qRAM　status
    qc.measure(address[0:2], c[0:2])
    qc.measure(data[0:3], c[2:5])

    # Reverse the output string.
    qc = qc.reverse_bits()

    # backend = provider.get_backend('ibmq_qasm_simulator')
    backend = Aer.get_backend('qasm_simulator')
    job = execute(qc, backend=backend, shots=1000, seed_simulator=12345)

    result = job.result()
    count = result.get_counts()
    for string, counts in dict(sorted(count.items())).items():
        print(f"Memory Cell: {string[:2]}\t Counts:  {counts}")

    qc.draw(output='mpl')
    plt.show()


if __name__ == "__main__":
    test_qram_write()
    test_qram_grover_read()
