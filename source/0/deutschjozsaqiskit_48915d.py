# https://github.com/junmendoza/QuantumPlayground/blob/903e2b61268810b05674d3e2a5c1fb9b41f38afe/Algorithms/DeutschJozsaQiskit.py
#
# Deutsch Jozsa Algorithm
# Implemented in Qiskit
#

# Qiskit core packages
from qiskit import BasicAer, IBMQ
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute
from qiskit.tools.monitor import job_monitor

# Visualisation tools
from qiskit.tools.visualization import plot_histogram

def deutschJozsaQiskit(oracle_type, input_qubits) :
    print("Running: deutschJozsaQiskit")

    # psi0: Init input qubits |001>
    #==============================
    print("psi0: Setting up " + str(input_qubits) + " input qubits")

    # Num input qubits
    n = input_qubits

    # Num oracle control bits
    m = 1

    # Oracle control qubit index
    oracle_control_qubit = n + m - 1

    # Total qubits in circuit
    nqubits = n + m + 1

    # Output qubit index
    output_qubit = nqubits - 1

    # Create the input + oracle qubits + 1 output qubit registers
    qregs = QuantumRegister(nqubits)

    # Create the bits that will store the final measurement
    regs = ClassicalRegister(n)

    # Create the quantum cirquit that contains the input qubits and measurement bits
    qcircuit = QuantumCircuit(qregs, regs)

    # Set the last qubit to be the |1>
    # This creates an initial state of |001>
    qcircuit.x(qregs[output_qubit])

    qcircuit.barrier()

    # psi1: Place all qubits in superposition
    #========================================
    print("psi1: Apply hadamard gates")

    # Apply hadamard to input and output qubits
    # This creates the state |+>|+>|->
    for i in range(nqubits):
        if i != oracle_control_qubit :
            qcircuit.h(qregs[i])

    qcircuit.barrier()

    # psi2: Apply the oracle function to the superposition of input qubits
    #=====================================================================
    print("psi2: Apply oracle function")

    # Start of oracle function
    if oracle_type == 'constant':
        constant_type = 1
        print("Running oracle function: Constant ", constant_type)
        if constant_type == 1:
            # Setup oracle function to always return 1
            # Constant 1 oracle function
            qcircuit.x(qregs[oracle_control_qubit])
    else: # oracle == BALANCED:
        print("Running oracle function: Balanced 1,-1,-1, 1")
        for i in range(n):
            qcircuit.cx(qregs[i], qregs[oracle_control_qubit])

    # CNOT the output qubit based on the oracle output qubit
    # i.e. this is the CNOT of the oracle function into the output qubit
    qcircuit.cx(qregs[oracle_control_qubit], qregs[output_qubit])
    qcircuit.barrier()


    # psi3: Prepare for measurement
    #==============================
    print("psi3: Apply hadamard gates")

    # Apply hadamard to in qubits
    # Apply hadamard to out qubit
    for i in range(nqubits):
        if i != oracle_control_qubit :
            qcircuit.h(qregs[i])

    qcircuit.barrier()

    # Final measurement
    #==================
    print("Final measurement")

    for i in range(n):
        qcircuit.measure(qregs[i], regs[i])

    # Draw the circuit
    print("Drawing circuit")
    print(qcircuit)

    image_filename = 'DeutschJozsaCircuitMPL.png';
    print("Saving circuit: ", image_filename)
    image = qcircuit.draw(output='mpl')
    image.savefig(image_filename)

    # Start simulation
    simulator = 'qasm_simulator'
    print("Start BasicAer simulator: ", simulator)
    backend = BasicAer.get_backend(simulator)
    shots = 100
    job = execute(qcircuit, backend=backend, shots=shots)
    results = job.result()
    results_count = results.get_counts()

    # Generate simulation results
    histogram_filename = 'DeutschJozsaHistogram.png';
    print("Plotting histogram: ", histogram_filename)
    histogram = plot_histogram(results_count)
    histogram.savefig(histogram_filename)

    print("Done")
