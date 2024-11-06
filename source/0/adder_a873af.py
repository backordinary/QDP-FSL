# https://github.com/enatheme/quantum-adder/blob/c1fa8012177e39ebfffd7d29aea34f176939dbe4/adder.py
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import IBMQ, Aer, execute

def simple_adder(lhs, rhs): # string of bits
    assert(len(lhs) == len(rhs))
    len_arg = len(lhs)
    assert(len_arg < 5)
    print(len_arg)
    q = QuantumRegister(len_arg * 4 + 1) # input1 + input2 + result + carry over + 1 (overflow)
    c = ClassicalRegister(len_arg + 1)
    qc = QuantumCircuit(q, c)

    input1_index = 0
    input2_index = len_arg
    carryover_index = input2_index + len_arg
    result_index = carryover_index + len_arg

    for ii, yesno in enumerate(lhs):
        if yesno == '1':
            qc.x(ii)
    for ii, yesno in enumerate(rhs):
        if yesno == '1':
            qc.x(ii + len_arg)
    qc.barrier()
    for i in range(0, len_arg):
        # XOR Gate
        qc.cx(q[input1_index + i], q[result_index + i])
        qc.cx(q[input2_index + i], q[result_index + i])

        # apply carry over
        if i > 0:
            qc.cx(q[carryover_index + i - 1], q[input2_index + i])

        # carry over (NAND Gate)
        qc.ccx(q[input1_index + i], q[input2_index + i], q[carryover_index + i])
        qc.ccx(q[input1_index + i], q[carryover_index + i - 1], q[carryover_index + i])
        qc.ccx(q[input2_index + i], q[carryover_index + i - 1], q[carryover_index + i])

    qc.barrier()
    for i in range(result_index, result_index + len_arg + 1):
        qc.measure(q[i], c[i - result_index])
    return qc

def add_circuit(lhs, rhs):
    # int to string of bit
    l = bin(lhs)[2:][::-1]
    r = bin(rhs)[2:][::-1]

    len_diff = len(l) - len(r)
    if len_diff > 0: # lhs > rhs
        r += '0' * len_diff
    elif len_diff < 0: # rhs > lhs
        l += '0' * -len_diff
    result = simple_adder(l, r)
    return result

def add(lhs, rhs):
    qc = add_circuit(lhs, rhs)
    simulator = Aer.get_backend('qasm_simulator')
    result = execute(qc, backend = simulator, shots = 100).result()
    for each in result.get_counts().keys():
        return int(each, 2)

lhs = 4
rhs = 0

#qc = add_circuit(lhs, rhs)
#qc.draw(output='mpl')
#simulator = Aer.get_backend('qasm_simulator')
#result = execute(qc, backend = simulator, shots = 100).result()
#print(result.get_counts())
#from qiskit.tools.visualization import plot_histogram
#plot_histogram(result.get_counts(qc))

print(f"{lhs} + {rhs} = {add(lhs, rhs)}")
