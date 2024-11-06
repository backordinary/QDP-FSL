# https://github.com/vic023/Task1_Submission/blob/8686ccbb82c2d344f4dcd55d715272604aa333c3/task1.py
from math import floor, sqrt, ceil, log, pi
from qiskit import *
from qiskit.circuit.library import *


"""
Calculates the first value by add alternating powers of 2 up to the data length.
Since the two values are complements of each other, we can just subract the
first value from 2^n - 1 to get the second value.

Input:  data_len - the length of the integer value in bits
Output: v1, v2   - the pair of integer values that have alternating bits
"""
def calcGoodVals(data_len):

    v1 = 0
    for i in range(data_len):
        if i % 2 == 0:
            v1 += 2 ** i

    v2 = 2**data_len - 1 - v1

    return v1, v2

"""
Helper function that changes integer values into binary and pads according to
the length of expected binary strings parameterized by n.

Inputs: x - the integer that has been representing the input bitstring x
        n - the length of the bitstring

Output: binx - processed and padded bitstring representation of x
               in n bits (reversed as per Qiskit's convention)
"""
def nbit_x(x, n):
    binx = bin(x)[2:]
    binx = ''.join(['0' for _ in range(n - len(binx))]) + binx
    return binx[::-1]

"""
Generates the circuit that loads the QRAM into memory.

Inputs: addr_len  - the length of the memory address
        data_len  - the length of each data elements
        input_vec - the data to be loaded into memory
"""
def make_QRAM(input_vec):

    # calculate how many qubits are needed
    addr_len = max(ceil(log(len(input_vec), 2)), 2)
    data_len = ceil(log(max(input_vec), 2))
    nqubits = addr_len + data_len + 1
    qram = QuantumCircuit(nqubits)

    # address bits controls must be exact, memory is assumed to be have been
    # 0'd out before load-in,
    for ind, val in enumerate(input_vec):

        # anticontrols
        for i, b in enumerate(nbit_x(ind, addr_len)):
            b = int(b)
            if not b:
                qram.x(i)

        # the address acts as controls, any '1' bit in the data is the target
        targetBits = [addr_len + t_ind for t_ind, bit in enumerate(nbit_x(val, data_len)) if int(bit)]
        mcmt = MCMT(XGate(), addr_len, len(targetBits))
        qram.append(mcmt, list(range(addr_len)) + targetBits)

        # anticontrols
        for i, b in enumerate(nbit_x(ind, addr_len)):
            b = int(b)
            if not b:
                qram.x(i)

    return qram

"""
Constructs the oracle function needed to mark the addresses which contain data
that has alternating bits. For the base problem of 4 elements in the input
vector, the addresses that satisify the condition are "marked", which are then
"extracted" via Hadamard's on the address space.

For the extended problem of having more than 4 elements, Grover's algorithm is
used instead. Phase Kickback is exploited to mark the addresses with an
auxilliary |-> qubit and then Grover's algorithm is used on the address space.
Only the initialization differs, so the oracles are the same.

Inputs: addr_len - the length of the addresses in bits
        data_len - the length of the data in bits
Ouptuts: oracle  - the quantum circuit that marks the the correct qubits
"""
def make_Oracle(addr_len, data_len):
    nqubits = addr_len + data_len + 1
    oracle = QuantumCircuit(nqubits)

    # alternating anticontrols for alternating bits
    oracle.x([i + addr_len for i in range(data_len) if i % 2 == 0])
    oracle.mct(list(range(addr_len, data_len + addr_len)), nqubits - 1)
    oracle.x([i + addr_len for i in range(data_len) if i % 2 == 0])
    # the complementary bit pattern
    oracle.x([i + addr_len for i in range(data_len) if i % 2 == 1])
    oracle.mct(list(range(addr_len, data_len + addr_len)), nqubits - 1)
    oracle.x([i + addr_len for i in range(data_len) if i % 2 == 1])

    return oracle

"""
The diffuser for Grover's. Operates on the address space to amplify the
amplitudes of the addresses that contain alternating qubits.

Inputs: addr_len - the length of the addresses in bits
        data_len - the length of the data in bits
Outputs: diffuser - the quantum circuit that performs amplitude amplification
"""
def make_Diffuser(addr_len, data_len):
    nqubits = addr_len + data_len + 1
    diffuser = QuantumCircuit(nqubits)
    diffuser.h(list(range(addr_len)))
    diffuser.x(list(range(addr_len)))
    diffuser.h(addr_len - 1)
    diffuser.mct(list(range(addr_len - 1)), addr_len - 1)
    diffuser.h(addr_len - 1)
    diffuser.x(list(range(addr_len)))
    diffuser.h(list(range(addr_len)))
    return diffuser



"""
Generates the quantum circuit that generates the superposition of the indices
(in qubits) that contain the integer values with alternating bit strings.

The bit length of the data is assumed to be the minimum bits needed to store the
largest value.

Input:  input_vec - the input integer vector
Output: qc        - the output circuit
"""
def solver(input_vec):

    # calculate address and data length in to prepare the circuit
    addr_len = max(ceil(log(len(input_vec), 2)), 2)
    data_len = ceil(log(max(input_vec), 2))

    # find the correct indices
    goodVals = list(calcGoodVals(data_len))
    indices = [input_vec.index(v) for v in goodVals]
    print(f"Correct indices: {indices}")

    # prepare the circuit
    nqubits = addr_len + data_len + 1
    qc = QuantumCircuit(nqubits, addr_len)
    for i in range(addr_len):
        qc.h(i)

    if addr_len > 2:
        qc.x(nqubits - 1)
        qc.h(nqubits - 1)

    # build the QRAM and oracle
    qram = make_QRAM(input_vec)
    oracle = make_Oracle(addr_len, data_len)

    # base problem, use simple solution
    if len(input_vec) <= 4:
        qc += qram
        qc += oracle
        qc += qram
        qc.h(list(range(addr_len)))

    # use Grover's if there are more than 4 elements in input
    else:
        #construct the diffuser
        diffuser = make_Diffuser(addr_len, data_len)

        # estimate iterations needed
        iters = floor(pi / 4 * sqrt(2**(addr_len - 1)))
        for _ in range(iters):
            qc += qram
            qc += oracle
            qc += qram
            qc += diffuser

    return qc

def main():
    # assume the first argument is a string of numbers
    try:
        input_vec = [int(s) for s in sys.argv[1].split(',')]
        assert(len(input_vec) >= 2)
    except:
        print("Invalid Input. Example correct input: '10,3,5,1'")
        return
    print(f"input vector: {input_vec}")
    print("Generating the circuit...")
    qc = solver(input_vec)
    print(qc)
    addr_len = max(ceil(log(len(input_vec), 2)), 2)
    qc.measure(list(range(addr_len)), list(range(addr_len)))
    print("Testing the circuit...")
    simulator = Aer.get_backend('qasm_simulator')
    job = execute(qc, simulator, shots=100)
    results = job.result()
    counts = results.get_counts(qc)

    # convert bitstrings to readable integer values, filter counts of 1
    processed_counts = {int(bitstring, 2): count for bitstring, count in counts.items() if count > 1}
    print(str(processed_counts))

if __name__ == '__main__':
    main()