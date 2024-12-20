# https://github.com/oreilly-qc/oreilly-qc.github.io/blob/a8dbfd638480c6cfc908c29822da104acdf8703b/samples/Qiskit/ch07_08_qft_rotating_phases.py
## Programming Quantum Computers
##   by Eric Johnston, Nic Harrigan and Mercedes Gimeno-Segovia
##   O'Reilly Media
##
## More samples like this can be found at http://oreilly-qc.github.io
##
## A complete notebook of all Chapter 6 samples (including this one) can be found at
##  https://github.com/oreilly-qc/oreilly-qc.github.io/tree/master/samples/Qiskit

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer, IBMQ, BasicAer
import math
## Uncomment the next line to see diagrams when running in a notebook
#%matplotlib inline

## Example 7-8: QFT rotating phases

# Set up the program
signal = QuantumRegister(4, name='signal')
qc = QuantumCircuit(signal)

def main():

    ## Rotate kth state in register by k times 20 degrees
    phi = 20;

    ## First HAD so that we can see the result for all k values at once
    qc.h(signal);
    val = 1 << 0
    for j in range(val):
        qc.rz(math.radians(phi), signal[0]);
    val = 1 << 1
    for j in range(val):
        qc.rz(math.radians(phi), signal[1]);
    val = 1 << 2
    for j in range(val):
        qc.rz(math.radians(phi), signal[2]);
    val = 1 << 3
    for j in range(val):
        qc.rz(math.radians(phi), signal[3]);

def QFT(qreg):
    ## This QFT implementation is adapted from IBM's sample:
    ##   https://github.com/Qiskit/qiskit-terra/blob/master/examples/python/qft.py
    ## ...with a few adjustments to match the book QFT implementation exactly
    n = len(qreg)
    for j in range(n):
        for k in range(j):
            qc.cu1(-math.pi/float(2**(j-k)), qreg[n-j-1], qreg[n-k-1])
        qc.h(qreg[n-j-1])
    # Now finish the QFT by reversing the order of the qubits
    for j in range(n//2):
        qc.swap(qreg[j], qreg[n-j-1])

main()

## That's the program. Everything below runs and draws it.

backend = BasicAer.get_backend('statevector_simulator')
job = execute(qc, backend)
result = job.result()

outputstate = result.get_statevector(qc, decimals=3)
for i,amp in enumerate(outputstate):
    if abs(amp) > 0.000001:
        prob = abs(amp) * abs(amp)
        print('|{}> {} probability = {}%'.format(i, amp, round(prob * 100, 5)))
qc.draw()        # draw the circuit

