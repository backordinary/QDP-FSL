# https://github.com/Simula-COMPLEX/qucat-paper/blob/04ec6c8d56849be432a17d64bbfccafe9d3fbefd/programs/BV.py
from qiskit import (
    # IBMQ,
    QuantumCircuit,
    QuantumRegister,
    ClassicalRegister,
    execute,
    Aer,
)
from qiskit.tools.visualization import circuit_drawer


def dec2bin(n):
    a = 1
    list = []
    while a > 0:
        a, b = divmod(n, 2)
        list.append(str(b))
        n = a
    s = ""
    for i in range(len(list) - 1, -1, -1):
        s += str(list[i])
    s = s.zfill(7)
    return s


def inverse(s):
    s_list = list(s)
    for i in range(len(s_list)):
        if s_list[i] == '0':
            s_list[i] = '1'
        else:
            s_list[i] = '0'
    s = "".join(s_list)
    return s


def BV(input, count_times):
    simulator = Aer.get_backend('qasm_simulator')
    oracle = QuantumRegister(7)
    register = QuantumRegister(7)
    c = ClassicalRegister(7)
    qc = QuantumCircuit(oracle, register, c)

    input_string = dec2bin(input)
    for i in range(len(input_string)):
        if input_string[6 - i] == '1':
            qc.x(oracle[i])

    qc.barrier(oracle)

    qc.h(register)
    for i in range(len(input_string)):
        qc.cz(oracle[i], register[i])

    qc.barrier(oracle)

    qc.h(register)
    qc.measure(register, c)

    #circuit_drawer(qc, filename='BV_circuit/BV_circuit.txt')

    job = execute(qc, simulator, shots=count_times * 100)
    result = job.result()
    counts = result.get_counts(qc)

    return counts


def BV_M1(input, count_times):
    simulator = Aer.get_backend('qasm_simulator')
    oracle = QuantumRegister(7)
    register = QuantumRegister(7)
    c = ClassicalRegister(7)
    qc = QuantumCircuit(oracle, register, c)

    input_string = dec2bin(input)
    for i in range(len(input_string)):
        if input_string[6 - i] == '1':
            qc.x(oracle[i])

    qc.mct([oracle[1],oracle[3]], oracle[6])  # M1

    qc.barrier(oracle)

    qc.h(register)
    for i in range(len(input_string)):
        qc.cz(oracle[i], register[i])

    qc.barrier(oracle)

    qc.h(register)
    qc.measure(register, c)

    from qiskit.tools.visualization import circuit_drawer

    #circuit_drawer(qc, filename='BV_circuit/BV_M1_circuit.txt')

    job = execute(qc, simulator, shots=count_times * 100)
    result = job.result()
    counts = result.get_counts(qc)

    return counts


def BV_M2(input, count_times):
    simulator = Aer.get_backend('qasm_simulator')
    oracle = QuantumRegister(7)
    register = QuantumRegister(7)
    c = ClassicalRegister(7)
    qc = QuantumCircuit(oracle, register, c)

    input_string = dec2bin(input)
    for i in range(len(input_string)):
        if input_string[6 - i] == '1':
            qc.x(oracle[i])

    qc.mct([oracle[0],oracle[2],oracle[4]], oracle[5])  # M2

    qc.barrier(oracle)

    qc.h(register)
    for i in range(len(input_string)):
        qc.cz(oracle[i], register[i])

    qc.barrier(oracle)

    qc.h(register)
    qc.measure(register, c)

    #circuit_drawer(qc, filename='BV_circuit/BV_M2_circuit.txt')

    job = execute(qc, simulator, shots=count_times * 100)
    result = job.result()
    counts = result.get_counts(qc)

    return counts


def BV_M3(input, count_times):
    simulator = Aer.get_backend('qasm_simulator')
    oracle = QuantumRegister(7)
    register = QuantumRegister(7)
    c = ClassicalRegister(7)
    qc = QuantumCircuit(oracle, register, c)

    input_string = dec2bin(input)
    for i in range(len(input_string)):
        if input_string[6 - i] == '1':
            qc.x(oracle[i])

    qc.mct([oracle[0],oracle[2],oracle[4],oracle[5]],oracle[3])  # M3

    qc.barrier(oracle)

    qc.h(register)
    for i in range(len(input_string)):
        qc.cz(oracle[i], register[i])

    qc.barrier(oracle)

    qc.h(register)
    qc.measure(register, c)

    #circuit_drawer(qc, filename='BV_circuit/BV_M3_circuit.txt')

    job = execute(qc, simulator, shots=count_times * 100)
    result = job.result()
    counts = result.get_counts(qc)

    return counts

def BV_specification(input):
    simulator = Aer.get_backend('statevector_simulator')
    oracle = QuantumRegister(7)
    register = QuantumRegister(7)
    qc = QuantumCircuit(oracle, register)

    input_string = dec2bin(input)
    for i in range(len(input_string)):
        if input_string[6 - i] == '1':
            qc.x(oracle[i])

    qc.barrier(oracle)

    qc.h(register)
    for i in range(len(input_string)):
        qc.cz(oracle[i], register[i])

    qc.barrier(oracle)

    qc.h(register)

    # from qiskit.tools.visualization import circuit_drawer
    #
    ##circuit_drawer(qc, filename='./BV_circuit')

    vector = execute(qc, simulator).result().get_statevector()

    return vector


def probabilityComputing(input):
    pt = []
    t = BV_specification(input)
    for i in range(128):
        temp = 0
        for j in range(128):
            temp += abs(t[i * 128 + j]) ** 2
        pt.append(temp)
    return pt

if __name__ == '__main__':
    print(BV(0,1))
    print(BV_M1(0,1))
    print(BV_M2(0,1))
    print(BV_M3(0,1))