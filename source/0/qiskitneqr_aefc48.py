# https://github.com/anjalia16/quantum-aia/blob/50feaa21a3f2601c25fb372e8fc30348b9471d76/QuantumAIA/QiskitNEQR.py
import math
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from typing import List

def NEQR(image : List[List[int]]) -> QuantumCircuit:

    intensity = QuantumRegister(8)
    y_register = QuantumRegister(math.ceil(math.log(len(image)) / math.log(2)))
    x_register = QuantumRegister(math.ceil(math.log(len(image[0])) / math.log(2)))
    i_measurement = ClassicalRegister(8)
    y_measurement = ClassicalRegister(len(y_register)) 
    x_measurement = ClassicalRegister(len(x_register))
    circuit = QuantumCircuit(intensity, y_register, x_register, i_measurement, y_measurement, x_measurement)

    circuit.h(y_register)
    circuit.h(x_register)

    for y in range(0, len(image)):
        for x in range(0, len(image[0])):
            binIntensity = "{0:b}".format(image[y][x])
            binX = "{0:b}".format(x)
            binY = "{0:b}".format(y)
            storeAlteredX = []
            storeAlteredY = []

            while len(binIntensity) < 8:
                binIntensity = "0" + binIntensity

            while len(binX) < len(x_register):
                binX = "0" + binX

            while len(binY) < len(y_register):
                binY = "0" + binY

            for num in range(0, len(binX)):
                if binX[num] == '0':
                    circuit.x(x_register[num])
                    storeAlteredX.append(True)
                else:
                    storeAlteredX.append(False)
            
            for num in range(0, len(binY)):
                if binY[num] == '0':
                    circuit.x(y_register[num])
                    storeAlteredY.append(True)
                else:
                    storeAlteredY.append(False)

            for num in range(0, len(binIntensity)):
                if binIntensity[num] == '1':
                    circuit.mcx(y_register[:] + x_register[:], intensity[num])

            #makeshift adjoint
            for i in range(0, len(storeAlteredX)):
                if storeAlteredX[i]:
                    circuit.x(x_register[i])
            for i in range(0, len(storeAlteredY)):
                if storeAlteredY[i]:
                    circuit.x(y_register[i])


    circuit.measure(intensity, i_measurement)
    circuit.measure(x_register, x_measurement)
    circuit.measure(y_register, y_measurement)
    
    return circuit







