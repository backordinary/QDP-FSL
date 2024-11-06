# https://github.com/lbadams2/csc_qc/blob/c9894fa587a331e44583717341013abd3a5d6875/hw4/dj.py
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute
from qiskit import IBMQ, BasicAer
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

# flip output qubit for half of possible qubits
def create_balanced_oracle(n):
    qc = QuantumCircuit(n+1)

    # don't have to use x gates but do it for fun
    b_str = '101' # dictates where x gates go
    for qubit in range(len(b_str)):
        if b_str[qubit] == '1':
            qc.x(qubit)

    # Use barrier as divider
    qc.barrier()

    # add cnot from every input to single output qubit
    for qubit in range(n):
        qc.cx(qubit, n)
    
    qc.barrier()

    # add another set of x gates to make it reversible
    for qubit in range(len(b_str)):
        if b_str[qubit] == '1':
            qc.x(qubit)

    return qc

    
# always either flips or doesn't flip the output qubit
def create_constant_oracle(n):
    qc = QuantumCircuit(n+1)
    #a = QuantumRegister(n, 'a')  # 3 inputs
    #b = QuantumRegister(1, 'b')  # 1 output    
    #qc.i(b[0]) # apply I gate to output for f(x) = 0
    return qc


def create_dj(n, oracle_type):
    dj_circuit = QuantumCircuit(n+1, n)
    
    # apply H to all inputs to put them in |+>
    for qubit in range(n):
        dj_circuit.h(qubit)

    # Put output in state |->
    dj_circuit.x(n)
    dj_circuit.h(n)

    # append constant orable to circuit
    if oracle_type == 'constant':
        oracle = create_constant_oracle(n)
    else:
        oracle = create_balanced_oracle(n)
    dj_circuit += oracle

    # put another set of H gates to make circuit reversible
    for qubit in range(n):
        dj_circuit.h(qubit)
    dj_circuit.barrier()

    # Measure input qubits
    for i in range(n):
        dj_circuit.measure(i, i)

    dj_circuit.draw(output='mpl')

    backend = BasicAer.get_backend('qasm_simulator')
    shots = 1024
    results = execute(dj_circuit, backend=backend, shots=shots).result()
    answer = results.get_counts()
    print(answer)
    plot_histogram(answer)
    plt.show()


if __name__ == '__main__':
    n = 3
    create_dj(n, 'constant')
    create_dj(n, 'balanced')
