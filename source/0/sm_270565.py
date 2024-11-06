# https://github.com/QuantumPapersData/QuSBT/blob/209ee443fc94ebf433e7026186d8c044c280af87/code/programs/SM.py
from qiskit import (
    # IBMQ,
    QuantumCircuit,
    QuantumRegister,
    ClassicalRegister,
    execute,
    Aer,
)


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


def Simon(input, count_times):
    simulator = Aer.get_backend('qasm_simulator')
    oracle = QuantumRegister(7)
    register = QuantumRegister(7)
    scratch = QuantumRegister(7)
    c = ClassicalRegister(7)
    qc = QuantumCircuit(oracle, register, scratch, c)

    input_string = dec2bin(input)
    for i in range(len(input_string)):
        if input_string[6 - i] == '1':
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

    # circuit_drawer(qc, filename='./Simon_circuit')

    job = execute(qc, simulator, shots=count_times * 100)
    result = job.result()
    counts = result.get_counts(qc)

    return counts


def Simon_M1(input, count_times):
    simulator = Aer.get_backend('qasm_simulator')
    oracle = QuantumRegister(7)
    register = QuantumRegister(7)
    scratch = QuantumRegister(7)
    c = ClassicalRegister(7)
    qc = QuantumCircuit(oracle, register, scratch, c)

    input_string = dec2bin(input)
    for i in range(len(input_string)):
        if input_string[6 - i] == '1':
            qc.x(oracle[i])

    qc.barrier(oracle)

    qc.swap(oracle[6], oracle[0])  # M1

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

    # circuit_drawer(qc, filename='./Simon_M1_circuit')

    job = execute(qc, simulator, shots=count_times * 100)
    result = job.result()
    counts = result.get_counts(qc)

    return counts


def Simon_M2(input, count_times):
    simulator = Aer.get_backend('qasm_simulator')
    oracle = QuantumRegister(7)
    register = QuantumRegister(7)
    scratch = QuantumRegister(7)
    c = ClassicalRegister(7)
    qc = QuantumCircuit(oracle, register, scratch, c)

    input_string = dec2bin(input)
    for i in range(len(input_string)):
        if input_string[6 - i] == '1':
            qc.x(oracle[i])

    qc.barrier(oracle)

    qc.ccx(oracle[5], oracle[6], oracle[0])  # M2

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

    # circuit_drawer(qc, filename='./Simon_M2_circuit')

    job = execute(qc, simulator, shots=count_times * 100)
    result = job.result()
    counts = result.get_counts(qc)

    return counts


def Simon_M3(input, count_times):
    simulator = Aer.get_backend('qasm_simulator')
    oracle = QuantumRegister(7)
    register = QuantumRegister(7)
    scratch = QuantumRegister(7)
    c = ClassicalRegister(7)
    qc = QuantumCircuit(oracle, register, scratch, c)

    input_string = dec2bin(input)
    for i in range(len(input_string)):
        if input_string[6 - i] == '1':
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

    qc.cnot(register[6], register[0])

    qc.measure(register, c)

    # from qiskit.tools.visualization import circuit_drawer

    # circuit_drawer(qc, filename='./Simon_M3_circuit')

    job = execute(qc, simulator, shots=count_times * 100)
    result = job.result()
    counts = result.get_counts(qc)

    return counts


def Simon_M4(input, count_times):
    simulator = Aer.get_backend('qasm_simulator')
    oracle = QuantumRegister(7)
    register = QuantumRegister(7)
    scratch = QuantumRegister(7)
    c = ClassicalRegister(7)
    qc = QuantumCircuit(oracle, register, scratch, c)

    input_string = dec2bin(input)
    for i in range(len(input_string)):
        if input_string[6 - i] == '1':
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

    qc.cswap(register[5], register[6], register[0])  # M4

    qc.measure(register, c)

    # from qiskit.tools.visualization import circuit_drawer

    # circuit_drawer(qc, filename='./Simon_M4_circuit')

    job = execute(qc, simulator, shots=count_times * 100)
    result = job.result()
    counts = result.get_counts(qc)

    return counts


def Simon_M5(input, count_times):
    simulator = Aer.get_backend('qasm_simulator')
    oracle = QuantumRegister(7)
    register = QuantumRegister(7)
    scratch = QuantumRegister(7)
    c = ClassicalRegister(7)
    qc = QuantumCircuit(oracle, register, scratch, c)

    input_string = dec2bin(input)
    for i in range(len(input_string)):
        if input_string[6 - i] == '1':
            qc.x(oracle[i])

    qc.barrier(oracle)

    qc.h(register)

    qc.x(register[0])

    for i in range(len(input_string)):
        qc.x(oracle[i])
        qc.x(register[i])
        qc.toffoli(oracle[i], register[i], scratch[i])
        qc.x(oracle[i])
        qc.x(register[i])
        qc.x(scratch[i])

    qc.barrier(register)

    qc.h(register[0])

    qc.h(register)

    qc.measure(register, c)

    # from qiskit.tools.visualization import circuit_drawer

    # circuit_drawer(qc, filename='./Simon_M5_circuit')

    job = execute(qc, simulator, shots=count_times * 100)
    result = job.result()
    counts = result.get_counts(qc)

    return counts


def Simon_specification(input):
    simulator = Aer.get_backend('statevector_simulator')
    oracle = QuantumRegister(7)
    register = QuantumRegister(7)
    scratch = QuantumRegister(7)
    qc = QuantumCircuit(oracle, register, scratch)

    input_string = dec2bin(input)
    for i in range(len(input_string)):
        if input_string[6 - i] == '1':
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

    return vector


def probabilityComputing(input):
    pt = []
    t = Simon_specification(input)
    for i in range(128):
        temp = 0
        for j in range(128):
            for k in range(128):
                temp += abs(t[i * 128 + j * 128 * 128 + k]) ** 2
        pt.append(temp)
    return pt
