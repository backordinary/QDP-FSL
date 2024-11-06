# https://github.com/DBasoco/Quantum-Computation/blob/0630ae4d3ecab85326484de9879ed8ef8c8f29f7/Gates.py
# Here we are simply building up our gate relationships from one library to the other.
# In each case, I will write the cirq version first then the qiskit version I would use.

import cirq
import qiskit

# this will be the toolkit that we need

# before we can see the outputs of gates we need to first create circuits and qubits for those gates to operate on'


# For cirq: here we are measuring the output of a single qubit acted on by the NOT gate(discussed later)

try:       # using try because the literature I'm using is calling a class function that isn't available
    # first we pick a qubit and establish its state this one is |0 >
    qubit = cirq.GridQubit(0, 0)

    # next we create a circuit
    circuit = cirq.Circuit.from_ops([
        cirq.X(qubit),  # this is the NOT gate
        cirq.measure(qubit, key='m')  # measure the qubit after NOT gate
    ])

    # display circuit
    print('Circuit:')
    print(circuit)

    # since we, presumedly, don't have access to QC we simulate the results
    simulator = cirq.Simulator()

    # now execute the code using 10 particles
    results = simulator.run(circuit, repetitions=10)

    # print the results
    print('Results:')
    print(results)
except AttributeError:
    print('Sorry, looks like you accidentally made black hole...')


# For qiskit: here will do the same process but in qiskit so that way you can see the difference in syntax

# first we create a quantum and classical registers to store the one qubit
qreg = qiskit.QuantumRegister(1, name='qreg')
creg = qiskit.ClassicalRegister(1, name='creg')

# create the circuit with the above registers
circ = qiskit.QuantumCircuit(qreg, creg)

# add NOT gate
circ.h(qreg[0])

# add measure on the qubit
circ.measure(qreg, creg)

# print the circuit
print(circ.draw())

# now we need to get something to simulate this on the back end
backend = qiskit.BasicAer.get_backend('qasm_simulator')

# execute the code on the backend for 10 particles
job = qiskit.execute(circ, backend, shots=10)
result = job.result()

# print measurement
print(result.get_counts())


# Conclusion: As you can see (if your code was able to run) the results were the same,
#       1 : 10.
# This means that all 10 particles returned in the |1> state. This is the expected result.
# Firstly, both languages it is customary to have each new qubit start in the |0> state. How do we do this?
#
# In cirq you have to manually adjust row and column placement of the qubit to establish it's initial state. In
# qiskit you simply determine how many you want to have in the register and it'll default to the customary notation.
#
# In cirq when you create the cirquit you have to include all operators that you want to act on the qubits during
# instantiation in the order you want them executed. In qiskit you make the circuit and then add the operators in
# down the thread in the order you want them executed. Now that we have out circuit let's look at the NOT gate we
# used or the X gate:
#
# A single qubit is either |0> or |1>, for those not familiar with dirac notation this means the same as their binary
# representation 0 or 1. In matrix notation these are both 2X1 matrices. The NOT or X gate is what is known as an
# operator. It is a 2X2 matrix that changes the state of the thing that it operates on. The NOT gate is also known as
# the flip gate. This is because it's operator switches a 0 to 1 and vice versa. Thus the reason why we expect to
# only measure 1's. Since, the particles all started as 0, after the X gate they all became 1.
#
# Now if you noticed when we called measure in cirq we didn't include a mapping to a classical bit, like in qiskit.
# This is because we already mapped them when we made the circuit. The only other difference is actually running the
# simulator to make this happen. You can actually gain computing time on both Google and IBM's QC but it is very
# limited so for non-research related projects we simply simulate the physics.
