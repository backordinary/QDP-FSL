# https://github.com/fizlip/quantum-signal-processing/blob/398fc5d42f0a129327a52046ab4360d52f95aa12/qsp_operation.py
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.opflow import CircuitOp, CircuitStateFn

import matplotlib.pyplot as plt
import numpy as np

import math

from qiskit.opflow import X, Y, Z, I

# We chart the interval between [-pi, pi]
start = -3.4
end = 3.4

# This is a constant used in one of the examples below
eta = 0.5*math.acos(-0.25)

"""
Calculate the expectation value an angle phi
@param phi: an angle in radians
@return the expectation value for the angle.
"""
def Evaluate(phi):

    current = start
    inc = 0.01

    data = []
    while current <= end:
        q           = QuantumRegister(1)
        circuit     = QuantumCircuit(q)

        for i in range(1,len(phi)):
            circuit.rz(-2*phi[i], q)
            circuit.rx(current, q)

        op = CircuitOp(circuit)

        psi = QuantumCircuit(2)
        psi.x(0)
        psi.x(1)

        psi = CircuitStateFn(psi)

        expectation = psi.adjoint().compose(op).compose(psi).eval().real

        data.append(expectation**2)

        current += inc
    return data

def PlotResponse(data):
    phi_trivial = (0,0)
    data_trivial    = Evaluate(phi_trivial)

    # Plot the two expectation distributions
    x = np.linspace(start, end, len(data_trivial))
    plt.plot(x, data_trivial)
    plt.plot(x, data)

    plt.xlabel("theta")
    plt.ylabel("Expectation (<0|U|0>)")
    plt.title("Value of expectation as a function of theta")

    plt.show()

#eta = 0.5*math.acos(-0.25)

#phi_bb1     = (math.pi/2, -eta, 2*eta, 0, -2*eta, eta)

#data_trivial    = calculateExpectation(phi_trivial)
#data_bb1        = calculateExpectation(phi_bb1)

# Plot the two expectation distributions
#x = np.linspace(start, end, len(data_trivial))
#plt.plot(x, data_trivial)
#plt.plot(x, data_bb1)

#plt.xlabel("theta")
#plt.ylabel("Expectation (<0|U|0>)")
#plt.title("Value of expectation as a function of theta")

#plt.show()
