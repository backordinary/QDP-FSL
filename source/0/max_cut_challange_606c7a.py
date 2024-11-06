# https://github.com/lukasszz/qiskit-exp/blob/ce14d53735870e7b6ace352629eb4049e9cd6740/ibm-challange/max-cut-challange.py
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute, Aer

""" 
4 vertices:  (q0)-(q3) (3 connections - edges)
ce_checker:  [q4]-[q6]
sum cuts:    q7 - carry out 
             q8 - sum
             q9 - c1 for full adder / ancilla for dissfusion
oracle:      q10 - is one only if we have q7,q8 == 11

                    (q3)
                      \ __[q6]
                       \ 
                (q2)---(q0)---(q1)
                     |      |
                     |      |
                    [q5]   [q4]   

States |1000> and |0111> gives us the maximum probability, because it means, that
one vertice (q0) is in one zone, and all the others vertices are in the other zone.


                
"""
q = QuantumRegister(11)
c = ClassicalRegister(4)
circ = QuantumCircuit()
circ.add_register(q)
circ.add_register(c)


def prepare():
    """ Put vertices in the zones. Into the superposition state to check all the possibilites once.
    """
    circ.h(q[0:4])

    # Oracle phase flip
    circ.x(q[10])
    circ.h(q[10])


def cutter_edge_checker(a: QuantumRegister, b: QuantumRegister, res: QuantumRegister):
    """ Checks if two connected vertices are:
        - in the same zone   (00, 11)->(0) or
        - in different zones (01, 10)->(1)
                             (ab)->res

    """
    circ.cx(a, res)
    circ.cx(b, res)


def count_cuts():
    """ Counts how many cuts do we have by making the summation of the ce_checkers.
        It is done by the full adder
    """
    S = q[8]
    C = q[7]
    c1 = q[9]
    circ.cx(q[4], S)
    circ.cx(q[5], S)
    circ.cx(q[6], S)
    circ.ccx(q[4], q[5], c1)
    circ.barrier()

    circ.cx(q[6], C)
    circ.cx(c1, C)
    circ.ccx(q[6], S, C)
    circ.barrier()


def oracle():
    """ Oracle is for finding the cuts grater than 2.
        So it checks if carry out (q6) from the half adder is 1
    """
    circ.ccx(q[7], q[8], q[10])
    # circ.ccz(q[7], q[8], q[10])


def run():
    backend = Aer.get_backend('qasm_simulator')
    job = execute(circ, backend=backend, shots=1024)
    return job.result()


def clean_helper_registers():
    """ We need to clean helper registers count cut, and half adder so they dont have an input for the final result.
        (to eliminate unnecessary entanglements between the input qubits)
        We do that by inversing this part of circuit
    """

    # invert full adder
    S = q[8]
    C = q[7]
    c1 = q[9]

    circ.barrier()
    circ.ccx(q[6], S, C)
    circ.cx(c1, C)
    circ.cx(q[6], C)

    circ.barrier()
    circ.ccx(q[4], q[5], c1)
    circ.cx(q[6], S)
    circ.cx(q[5], S)
    circ.cx(q[4], S)

    # invert counting edges
    #   cutter_edge_checker(q[0], q[3], q[6])
    circ.cx(q[3], q[6])
    circ.cx(q[0], q[6])

    #   cutter_edge_checker(q[0], q[2], q[5])
    circ.cx(q[2], q[5])
    circ.cx(q[0], q[5])

    #   cutter_edge_checker(q[0], q[1], q[4])
    circ.cx(q[1], q[4])
    circ.cx(q[0], q[4])


def amplitude_amplification():
    """
        diffusion operations
    """
    # diffusion part
    circ.h(q[0:4])
    circ.barrier()
    circ.x(q[0:4])
    circ.barrier()
    circ.h(q[3])
    circ.mct([q[0], q[1], q[2]], q[3], [q[9]], mode='basic')
    circ.h(q[3])
    circ.barrier()
    circ.x(q[0:4])
    circ.barrier()
    circ.h(q[0:4])
    circ.barrier()


def max_cut_v3_challange_test_oracle():
    circ.h(q[0:4])  # prepare()
    circ.barrier()

    # Oracle part
    cutter_edge_checker(q[0], q[1], q[4])
    cutter_edge_checker(q[0], q[2], q[5])
    cutter_edge_checker(q[0], q[3], q[6])
    count_cuts()

    circ.ccx(q[7], q[8], q[10])  # oracle()
    circ.barrier()

    circ.measure(q[10], c[0])

    res = run()

    print("max_cut")
    print(circ.draw(output='text', line_length=200))
    print(res.get_counts())
    print("We have test the oracle the answear: {'0000': high, '0001': low}, so there is much"
          " more possibilities lower then 3 cuts - that show that our oracle works.")


def max_cut_v3_challange():
    prepare()
    circ.barrier()
    # Oracle part
    cutter_edge_checker(q[0], q[1], q[4])
    cutter_edge_checker(q[0], q[2], q[5])
    cutter_edge_checker(q[0], q[3], q[6])
    count_cuts()
    oracle()
    circ.barrier()

    clean_helper_registers()
    circ.barrier()

    amplitude_amplification()
    circ.barrier()
    # Oracle part
    cutter_edge_checker(q[0], q[1], q[4])
    cutter_edge_checker(q[0], q[2], q[5])
    cutter_edge_checker(q[0], q[3], q[6])
    count_cuts()
    oracle()
    circ.barrier()

    clean_helper_registers()
    circ.barrier()

    amplitude_amplification()
    circ.barrier()

    circ.measure(q[0:4], c[0:4])

    res = run()

    print("max_cut")
    print(circ.draw(output='text', line_length=200))
    print(res.get_counts())
    # print("We have found the answear: '0001': high, '1110': high,")

    # Decompose the circuit by using the Unroller
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import Unroller
    pass_ = Unroller(['u3', 'cx'])
    pm = PassManager(pass_)
    new_circuit = pm.run(circ)
    # new_circuit.draw(output='mpl')

    #show elementary gate counts
    new_circuit.count_ops()
    # create output text file with the gate counts
    import json
    dct = new_circuit.count_ops()
    with open('wk3_output-last-run.txt', 'w') as f:
        f.write(json.dumps(dct))
    print('0001 %s', res.get_counts()['0001'])
    print('1110 %s', res.get_counts()['1110'])

if __name__ == '__main__':
    # max_cut_v3_challange_test_oracle()
    # or
    max_cut_v3_challange()
