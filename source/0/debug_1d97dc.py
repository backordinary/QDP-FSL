# https://github.com/sh3rlock14/Intensive-Computation-2021-2022/blob/247a1cad19eac783c1d5144da07e309525e3a13f/debug.py
from qiskit import QuantumRegister, ClassicalRegister, AncillaRegister, QuantumCircuit, assemble, Aer, transpile
from qiskit.providers.aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
from qiskit.circuit.library import SXdgGate

# Create a controlled version of V+ gate
csxdg = SXdgGate(label="V+").control(1)

def V_Gate(inp):
    """A V-Gate

    Parameters:
        inp (str): Input, encoded in qubit 0

    Returns:
        QuantumCircuit: Output V_Gate circuit
        str: Output value measured from qubit 1.
    """

    qc = QuantumCircuit(2,2)
    qc.reset(0)
    qc.reset(1)

    # encode the input - if it is 0, don't do anything
    if inp == '1':
        qc.x(0)
    
    # barrier between input state and gate operation
    qc.barrier()

    # apply the controlled V-gate
    qc.csx(0, 1)
    qc.barrier()


    # measure the output
    qc.measure(0, 0)
    qc.measure(1,1)
    qc.draw(output='mpl')


    # We'll run the program on a simulator
    backend = AerSimulator() #Aer.get_backend('aer_simulator')
    
    # transpile the circuit
    qc_compiled = transpile(qc, backend)

    # execute the circuit
    job_sim = backend.run(qc_compiled, shots=1024)
    result_sim = job_sim.result()
    counts = result_sim.get_counts(qc_compiled)

    return qc, counts

def V_h_Gate(inp):
    """A V-inverse-Gate

    Parameters:
        inp (str): Input, encoded in qubit 0

    Returns:
        QuantumCircuit: Output V-Inverse_Gate circuit
        str: Output value measured from qubit 1.
    """

    qc = QuantumCircuit(2,2)
    qc.reset(0)
    qc.reset(1)

    # encode the input - if it is 0, don't do anything
    if inp == '1':
        qc.x(0)
    
    # barrier between input state and gate operation
    qc.barrier()

    # apply the controlled V-gate
    qc.csx(0, 1).inverse()
    qc.barrier()


    # measure the output
    qc.measure(0, 0)
    qc.measure(1,1)
    qc.draw(output='mpl')


    # We'll run the program on a simulator
    backend = AerSimulator() #Aer.get_backend('aer_simulator')
    
    # transpile the circuit
    qc_compiled = transpile(qc, backend)

    # execute the circuit
    job_sim = backend.run(qc_compiled, shots=1024)
    result_sim = job_sim.result()
    counts = result_sim.get_counts(qc_compiled)

    return qc, counts


# Define the Peres gate
qc_peres = QuantumCircuit(3)
qc_peres.append(csxdg, qargs=[0,2])
qc_peres.append(csxdg, qargs=[1,2])
qc_peres.cnot(0,1)
qc_peres.csx(1,2)

Peres_Gate = qc_peres.to_gate(label="PG")


qc = QuantumCircuit()

RCSA_inputs = ["a{}", "b{}", "a'_{}", "b_{}_2", "c_in_{}", "a_{}_2", "b_{}_3", "ancilla_{}_0", "a'_{}_2", "b'_{}", "ancilla_{}_1", "ancilla_buffer_{}_0"] # replace the hardcoded '0' with for i in range(n_bit) -> {}.fomrat(i)
n_bit = 2




# Add the quantum register for the inputs
for b in range(n_bit):
    #if b > 0:
    #    RCSA_inputs.pop(4)
    for i in range (len(RCSA_inputs)):
        qc.add_register(QuantumRegister(1, RCSA_inputs[i].format(b)))



# Encode the input
a = "10"
b = "10"
c_in = "0"


def encode_input(a,b,c_in, n_bit):

    a = a[::-1] # read from LSB to MSB
    b = b[::-1]


    # encode a
    a_pos = [0, 5]
    a_not_pos = [2, 8]
    b_pos = [1,3,6]
    b_not_pos = [9]

    for n in range(n_bit):

        for bit in a:
            for ix in a_pos:
                #qc.reset(ix)
                if bit ==  "1":
                    qc.x(ix + 12 * n ) # 12 = len(ECSA_input)

        # encode a'
        for bit in a:
            for ix in a_not_pos:
                qc.x(ix + 12 * n)
                qc.cnot(0 + 12 * n, ix + 12 * n)

        # encode b
        
        for bit in b:
            for ix in b_pos:
                #qc.reset(ix)
                if bit ==  "1":
                    qc.x(ix + 12 * n)
        
        # encode b'
        
        for bit in b:
            for ix in b_not_pos:
                qc.x(ix + 12 * n)
                qc.cnot(0 + 12 * n, ix + 12 * n)
                


        # encode c_in
        if c_in == "0":
            qc.reset(4 + 12 * n)
        else:
            qc.x(4 + 12 * n)

        # encode ancilla_0
        qc.reset(7 + 12 * n)

        # encode ancilla_1
        qc.x(10 + 12 * n)

        # encode ancilla_buffer_0
        qc.reset(11 + 12 * n)


encode_input(a,b,c_in, n_bit)


qc.draw(output='mpl')