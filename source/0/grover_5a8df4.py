# https://github.com/PierreHandtschoewercker/QuantumNoteFinder/blob/bbcc20759a5d06f99640df49c6fc5544228f6524/Grover.py
from qiskit import *
from qiskit.aqua.algorithms import Grover
from qiskit.aqua.components.oracles import TruthTableOracle
from qiskit.aqua import QuantumInstance
import math


#          0000        0001        0010        0011       0100       0101       0110       1000      1001       1010      1011         1100

def buildTruthTable(notes, freq):
    truthTable = ''
    for n in notes:
        if float(notes[n] - 3) < float(freq) < float(notes[n] + 3):
            truthTable += '1'
        else:
            truthTable += '0'
    return truthTable


def grover(freq):
    notes = {'C': 65.41, 'C#': 69.30, 'D': 73.42, 'D#': 77.78, 'E': 82.41, 'F': 87.31, 'F#': 92.50, 'G': 98,
             'G#': 103.83,
             'A': 110, 'A#': 116.54, 'B': 123.5, '0': 0, '1': 0, '2': 0, '3': 0
             }
    truthtable = buildTruthTable(notes, freq)
    oracle = TruthTableOracle(truthtable)
    grover = Grover(oracle)
    result = grover.run(QuantumInstance(BasicAer.get_backend('qasm_simulator'), shots=1024))
    indice_note = int(result['top_measurement'], 2)
    key = list(notes)[indice_note]
    print(freq,key)
