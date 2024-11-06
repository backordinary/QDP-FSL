# https://github.com/Simula-COMPLEX/qucat-paper/blob/04ec6c8d56849be432a17d64bbfccafe9d3fbefd/programs/SM.py
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
    s = s.zfill(5)
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


def SM(input, count_times):
    simulator = Aer.get_backend('qasm_simulator')
    oracle = QuantumRegister(5)
    register = QuantumRegister(5)
    scratch = QuantumRegister(5)
    c = ClassicalRegister(5)
    qc = QuantumCircuit(oracle, register, scratch, c)

    input_string = dec2bin(input)
    for i in range(len(input_string)):
        if input_string[4 - i] == '1':
            qc.x(oracle[i])

    qc.barrier(oracle)

    qc.h(register)
    for i in range(len(input_string)):
        qc.x(oracle[i])
        qc.x(register[i])
        qc.toffoli(oracle[i], register[i], scratch[i])
        qc.x(oracle[i])
        qc.x(register[i])
        qc.x(scratch[i])

    qc.barrier(register)

    qc.h(register)

    qc.measure(register, c)

    #circuit_drawer(qc, filename='SM_circuit/Simon_circuit.txt')

    job = execute(qc, simulator, shots=count_times * 100)
    result = job.result()
    counts = result.get_counts(qc)

    return counts


def SM_M1(input, count_times):
    simulator = Aer.get_backend('qasm_simulator')
    oracle = QuantumRegister(5)
    register = QuantumRegister(5)
    scratch = QuantumRegister(5)
    c = ClassicalRegister(5)
    qc = QuantumCircuit(oracle, register, scratch, c)

    input_string = dec2bin(input)
    for i in range(len(input_string)):
        if input_string[4 - i] == '1':
            qc.x(oracle[i])

    qc.barrier(oracle)

    qc.mct([oracle[0],oracle[3]],oracle[4])

    qc.h(register)
    for i in range(len(input_string)):
        qc.x(oracle[i])
        qc.x(register[i])
        qc.toffoli(oracle[i], register[i], scratch[i])
        qc.x(oracle[i])
        qc.x(register[i])
        qc.x(scratch[i])

    qc.barrier(register)

    qc.h(register)

    qc.measure(register, c)

    # from qiskit.tools.visualization import circuit_drawer

    #circuit_drawer(qc, filename='SM_circuit/Simon_M1_circuit.txt')

    job = execute(qc, simulator, shots=count_times * 100)
    result = job.result()
    counts = result.get_counts(qc)

    return counts


def SM_M2(input, count_times):
    simulator = Aer.get_backend('qasm_simulator')
    oracle = QuantumRegister(5)
    register = QuantumRegister(5)
    scratch = QuantumRegister(5)
    c = ClassicalRegister(5)
    qc = QuantumCircuit(oracle, register, scratch, c)

    input_string = dec2bin(input)
    for i in range(len(input_string)):
        if input_string[4 - i] == '1':
            qc.x(oracle[i])

    qc.barrier(oracle)

    qc.mct([oracle[0], oracle[2],oracle[3]], oracle[4])

    qc.h(register)
    for i in range(len(input_string)):
        qc.x(oracle[i])
        qc.x(register[i])
        qc.toffoli(oracle[i], register[i], scratch[i])
        qc.x(oracle[i])
        qc.x(register[i])
        qc.x(scratch[i])

    qc.barrier(register)

    qc.h(register)

    qc.measure(register, c)

    # from qiskit.tools.visualization import circuit_drawer

    #circuit_drawer(qc, filename='SM_circuit/Simon_M2_circuit.txt')

    job = execute(qc, simulator, shots=count_times * 100)
    result = job.result()
    counts = result.get_counts(qc)

    return counts

def SM_M3(input, count_times):
    simulator = Aer.get_backend('qasm_simulator')
    oracle = QuantumRegister(5)
    register = QuantumRegister(5)
    scratch = QuantumRegister(5)
    c = ClassicalRegister(5)
    qc = QuantumCircuit(oracle, register, scratch, c)

    input_string = dec2bin(input)
    for i in range(len(input_string)):
        if input_string[4 - i] == '1':
            qc.x(oracle[i])

    qc.barrier(oracle)

    qc.mct([oracle[0],oracle[1], oracle[2],oracle[3]], oracle[4])

    qc.h(register)
    for i in range(len(input_string)):
        qc.x(oracle[i])
        qc.x(register[i])
        qc.toffoli(oracle[i], register[i], scratch[i])
        qc.x(oracle[i])
        qc.x(register[i])
        qc.x(scratch[i])

    qc.barrier(register)

    qc.h(register)

    qc.measure(register, c)

    # from qiskit.tools.visualization import circuit_drawer

    #circuit_drawer(qc, filename='SM_circuit/Simon_M3_circuit.txt')

    job = execute(qc, simulator, shots=count_times * 100)
    result = job.result()
    counts = result.get_counts(qc)

    return counts

def SM_specification(input):
    simulator = Aer.get_backend('statevector_simulator')
    oracle = QuantumRegister(5)
    register = QuantumRegister(5)
    scratch = QuantumRegister(5)
    qc = QuantumCircuit(oracle, register, scratch)

    input_string = dec2bin(input)
    for i in range(len(input_string)):
        if input_string[4 - i] == '1':
            qc.x(oracle[i])

    qc.barrier(oracle)

    qc.h(register)
    for i in range(len(input_string)):
        qc.x(oracle[i])
        qc.x(register[i])
        qc.toffoli(oracle[i], register[i], scratch[i])
        qc.x(oracle[i])
        qc.x(register[i])
        qc.x(scratch[i])

    qc.barrier(register)

    qc.h(register)

    vector = execute(qc, simulator).result().get_statevector()

    ##circuit_drawer(qc, filename='./SM_circuit')

    return vector


def probabilityComputing(input):
    pt = []
    t = SM_specification(input)
    for i in range(32):
        temp = 0
        for j in range(32):
            for k in range(32):
                temp += abs(t[i * 32 + j * 32 * 32 + k]) ** 2
        pt.append(temp)
    return pt
if __name__ == '__main__':
    print(SM(0,1))
    print(SM_M1(0,1))
    print(SM_M2(0,1))
    print(SM_M3(0,1))
