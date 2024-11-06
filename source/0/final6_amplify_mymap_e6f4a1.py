# https://github.com/lukasszz/qiskit-exp/blob/ce14d53735870e7b6ace352629eb4049e9cd6740/ibm-challange/final6-amplify-mymap.py
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute, Aer, IBMQ

""" 
"""

q = QuantumRegister(12)
c = ClassicalRegister(12)
cmp_ancilla = QuantumRegister(1)
cmp_res = QuantumRegister(1)
sc = ClassicalRegister(1)
circ = QuantumCircuit()
circ.add_register(q)
circ.add_register(cmp_ancilla)
circ.add_register(cmp_res)
circ.add_register(c)
circ.add_register(sc)

mem_ancilla = QuantumRegister(1)  # anclia
mem_output = QuantumRegister(1)  # output
mem_output_c = ClassicalRegister(1)
circ.add_register(mem_ancilla, mem_output, mem_output_c)

oracle_q = QuantumRegister(1)
oracle_c = ClassicalRegister(1)
circ.add_register(oracle_q, oracle_c)

amp_ancilla = QuantumRegister(1)  # anclia
circ.add_register(amp_ancilla)


def mem(mem_input):
    circ.barrier()
    # cmp_res OR previous_output > ancila
    circ.cx(mem_input, mem_ancilla)
    circ.cx(mem_output, mem_ancilla)
    circ.ccx(mem_input, mem_output, mem_ancilla)

    # move ancila (OR result) output and clean ancila
    circ.swap(mem_ancilla, mem_output)
    circ.reset(mem_ancilla)


def init_map():
    # A: 00 red
    # B: 01 blue
    # C: 10 yellow
    # D: 11 green

    # 1 D
    circ.x(q[0])
    circ.x(q[1])

    # SUPERPOSITION
    # 2 A
    # q[2]
    # q[3]
    circ.h(q[2])
    circ.h(q[3])

    # 3 B
    # q[4]
    circ.x(q[5])

    # 4 D
    circ.x(q[6])
    circ.x(q[7])

    # SUPERPOSITION
    # 5 A
    # q[8]
    # q[9]
    circ.h(q[8])
    circ.h(q[9])

    # 6 C
    circ.x(q[10])
    # q[11]


def cmp(a, b):
    # compare into the second vertice
    circ.cx(a[0], b[0])
    circ.cx(a[1], b[1])
    circ.cx(b[0], cmp_ancilla)
    circ.cx(b[1], cmp_ancilla)
    circ.ccx(b[0], b[1], cmp_ancilla)  # q[4] = 0 the same BAD , 1 odd OK
    # OR if there was at least one adjenct
    circ.x(cmp_ancilla)
    circ.cx(cmp_ancilla, cmp_res)  # q[5] - if there were at least one the same adjenct chains q5 will by 1

    # revert scecond vertice
    circ.cx(a[1], b[1])
    circ.cx(a[0], b[0])

    circ.reset(cmp_ancilla)


def oracle():
    """ Oracle is for finding maps with no adjenct chains (mem_res = 0)
        So it checks if carry out (mem_res) is 0
    """
    circ.x(mem_output)
    circ.cz(mem_output, oracle_q)
    # # circ.h(oracle_q)
    # # revert mem
    circ.x(mem_output)


def run():
    circ.measure(q, c)
    # circ.measure(oracle_q, oracle_c)
    # circ.measure(mem_output, mem_output_c)
    # run
    # backend = Aer.get_backend('qasm_simulator')
    provider = IBMQ.load_account()
    backend = provider.get_backend('ibmq_qasm_simulator')
    job = execute(circ, backend=backend, shots=8000, seed_simulator=12345, backend_options={"fusion_enable":True})
    print(job.result().get_counts())

def amplitude_amplification():
    """
        diffusion operations
    """
    # diffusion part
    circ.h(q[2])
    circ.h(q[3])
    circ.h(q[8])
    circ.h(q[9])
    circ.barrier()
    circ.x(q[2])
    circ.x(q[3])
    circ.x(q[8])
    circ.x(q[9])
    circ.barrier()
    circ.h(q[9])
    #circ.cx(q[2], q[9])
    circ.mct([q[2], q[3], q[8]], q[9], amp_ancilla, mode='basic')
    circ.h(q[9])
    circ.barrier()
    circ.x(q[2])
    circ.x(q[3])
    circ.x(q[8])
    circ.x(q[9])
    circ.barrier()
    circ.h(q[2])
    circ.h(q[3])
    circ.h(q[8])
    circ.h(q[9])
    circ.barrier()


if __name__ == '__main__':
    init_map()
    circ.x(oracle_q)
    circ.h(oracle_q)
    circ.barrier()
    cmp((q[0], q[1]), (q[2], q[3]))  # 1 - 2
    mem(cmp_res)
    cmp((q[0], q[1]), (q[4], q[5]))  # 1 - 3
    mem(cmp_res)
    cmp((q[0], q[1]), (q[8], q[9]))  # 1 - 5
    mem(cmp_res)
    cmp((q[2], q[3]), (q[4], q[5]))  # 2 - 3
    mem(cmp_res)
    cmp((q[2], q[3]), (q[6], q[7]))  # 2 - 4
    mem(cmp_res)
    cmp((q[4], q[5]), (q[6], q[7]))  # 3 - 4
    mem(cmp_res)
    cmp((q[4], q[5]), (q[8], q[9]))  # 3 - 5
    mem(cmp_res)
    cmp((q[6], q[7]), (q[8], q[9]))  # 4 - 5
    mem(cmp_res)
    cmp((q[2], q[3]), (q[10], q[11]))  # 2 - 6
    mem(cmp_res)

    oracle()
    amplitude_amplification()
    circ.h(oracle_q)
    run()

    # Solution
    #   55    22   5 2
    # 010011100011 R R
    # 011011100011 Y R

# '1 0 0 010111100111': 390,
# '1 0 0 010011100011': 158,
# '1 0 0 011011100011': 548,
# '1 0 0 011111100111': 542,
# '0 0 0 011011101011': 43,
# '1 0 0 011111101011': 543,
# '0 0 0 010011101011': 27,
# '0 0 0 010111100011': 263,
# '1 0 0 010111100011': 164,
# '1 0 0 010011101111': 425,
# '1 0 0 011111100011': 528,
# '0 0 0 010111101111': 23,
# '0 0 0 010111101011': 32,
# '1 0 0 010011100111': 404,
# '1 0 0 011011101011': 538,
# '0 0 0 010011100011': 280,
# '0 0 0 011111100011': 29,
# '0 0 0 011111101011': 25,
# '1 0 0 011111101111': 540,
# '1 0 0 010111101011': 406,
# '0 0 0 011111101111': 35,
# '0 0 0 010011100111': 30,
# '1 0 0 010011101011': 451,
# '0 0 0 011011100111': 36,
# '0 0 0 011111100111': 25,
# '0 0 0 011011100011': 38,
# '0 0 0 011011101111': 34,
# '1 0 0 010111101111': 378,
# '1 0 0 011011100111': 495,
# '0 0 0 010011101111': 25,
# '0 0 0 010111100111': 22,
# '1 0 0 011011101111': 523}
