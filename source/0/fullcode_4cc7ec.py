# https://github.com/berkeayevi/Phys343/blob/b3bfa12e92a5227697f4dfdfb29060e84f089d3f/TermProject/FullCode.py
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import IBMQ, Aer, transpile, assemble
from qiskit.visualization import plot_histogram, plot_bloch_multivector
from qiskit.extensions import Initialize
from qiskit_textbook.tools import random_state, array_to_latex


def encoder(qc, encodedmessage):
    stringtobin = ''.join(format(ord(i), '08b') for i in encodedmessage)
    listforbin = []
    r = float(0.0)

    for i in stringtobin:
        listforbin.append(float(i))

    for i in range(len(listforbin)):
        r = r + (1.0 / (2 ** (i + 1))) * listforbin[i]

    rotation = r * np.pi * 2
    qc.ry(rotation, 0)
    return rotation


# Teamon Creates Bell Pair
def Bellpair_Teamon(qc, a, b):
    qc.h(a)  # puts the qubit in superposition
    qc.cx(a, b)  # Apply CNOT gate and produces Bell Pair


# Allice
def Allice(qc, Phase, a):
    qc.cx(Phase, a)
    qc.h(Phase)


def Measure(qc, a, b):
    qc.measure(a, 0)
    qc.measure(b, 1)


# Bob
def Bob(qc, q2, C1, C2):
    qc.x(q2).c_if(C2, 1)  # If the channel 1 ==1 then applys X gate
    qc.z(q2).c_if(C1, 1)  # If the channel 1 ==1 then applys Z gate


def decoder(qc):
    sv_sim = Aer.get_backend('statevector_simulator')
    qobj = assemble(Circuit)
    out_vector = sv_sim.run(qobj).result().get_statevector()

    teleporttedrot = np.sort(out_vector)
    teleporttedrot = np.real(np.arctan(a[-1] / a[-2]))
    rotat = teleporttedrot
    count = 0
    decodedlist = []
    while rotat > 0.0000000000000001:
        divider = (1.0 / (2.0 ** (count + 1)))
        if (rotat - divider) < 0:
            decodedlist.append(0)
            count = count + 1
        else:
            rotat = rotat - divider
            decodedlist.append(1)
            count = count + 1

    while True:
        if (len(decodedlist) % 8) == 0:
            break
        decodedlist.append(0)

    binstr = ""
    for elem in range(len(l)):
        if elem % 8 == 0:
            binstr += " "
        binstr += str(l[elem])

    son = binstr[1:]

    binary_values = son.split()
    ascii_string = ""
    for binary_value in binary_values:
        an_integer = int(binary_value, 2)
        ascii_character = chr(an_integer)
        ascii_string += ascii_character

    return print(ascii_string)


def textdivider(message):
    d = 0
    dividedlist = []
    for i in range(len(message)):
        if (message[5 * i:5 * (i + 1)] == ''):
            d = 0
        else:
            dividedlist.append(message[5 * i:5 * (i + 1)])

    return dividedlist


def check(qc, rotation):
    sv_sim = Aer.get_backend('statevector_simulator')
    qobj = assemble(qc)
    out_vector = sv_sim.run(qobj).result().get_statevector()

    teleporttedrot = np.sort(out_vector)
    teleporttedrot = np.abs(np.real(np.arctan(teleporttedrot[-1] / teleporttedrot[-2])))

    if (teleporttedrot == (rotation * 0.5)):

        return print('Message Sent!!')
    else:
        return print('Error Occured')


def Teleportation(messageFromAllice):
    C1, C2 = ClassicalRegister(1), ClassicalRegister(1)
    Q = QuantumRegister(3)
    Circuit = QuantumCircuit(Q, C1, C2)  # Create 3 Qubits and 2 Measurement channel

    message = messageFromAllice  # The Message that Allice want to send

    rotation = encoder(Circuit, message)  # Encodes the given message and Rotates the Allice's qubit accordingly
    Circuit.barrier()

    Bellpair_Teamon(Circuit, 1, 2)  # Creates Bellpair
    Circuit.barrier()

    Allice(Circuit, 0, 1)  # Allice's action to first and Second Qubit
    Circuit.barrier()

    Measure(Circuit, 0, 1)  # Measuring Qubit 1 and 2 and sending trought classical channel to the bob

    Bob(Circuit, 2, C1, C2)  # Bob's action and gets the Encoded qubit created by the Allice

    check(Circuit, rotation)


message = 'helloworld'

divtex = textdivider(message)

for i in range(len(divtex)):
    Teleportation(divtex[i])
