# https://github.com/Gcuencam/full-tomography/blob/331217f6a97038f0e2e74cd63a230c6aec1ea583/src/common/quantum_commons.py
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 19:42:48 2022

@author: Pedro
"""
import collections
import os

from qiskit import transpile
from qiskit.providers.aer import QasmSimulator


def simulate(qc, shots):
    simulator = QasmSimulator()
    compiled_circuit = transpile(qc, simulator)
    return simulator.run(compiled_circuit, shots=shots)


def debug_circuit(qc, counts, probs):
    print(qc)
    print(counts)
    print('probs: {}'.format(probs))


def isDebugEnabled():
    if "DEBUG" in os.environ:
        return os.environ['DEBUG'] == 'True'
    else:
        return False

def countPovmIndividual(counts):
    qbits_count = {}
    count_keys = list(counts.keys())
    for bits in count_keys:
        for q_index in range(0, len(bits)):
            q_value = str(bits[q_index])
            if q_index in qbits_count:
                q_counts = qbits_count[q_index]
                if q_value in q_counts:
                    q_counts[q_value] = q_counts[q_value] + counts[bits]
                else:
                    q_counts[q_value] = counts[bits]
            else:
                q_counts = {}
                qbits_count[q_index] = q_counts
                if q_value in q_counts:
                    q_counts[q_value] = q_counts[q_value] + counts[bits]
                else:
                    q_counts[q_value] = counts[bits]

    return qbits_count

def countPovm(counts):
    qbits_count = {}
    count_keys = list(counts.keys())
    for bits in count_keys:
        for q_index in range(0, len(bits), 2):
            if q_index in qbits_count:
                q_counts = qbits_count[q_index]
                q_value = str(bits[q_index]) + str(bits[q_index + 1])
                if q_value in q_counts:
                    q_counts[q_value] = q_counts[q_value] + counts[bits]
                else:
                    q_counts[q_value] = counts[bits]
            else:
                q_counts = {}
                qbits_count[q_index] = q_counts
                q_value = str(bits[q_index]) + str(bits[q_index + 1])
                if q_value in q_counts:
                    q_counts[q_value] = q_counts[q_value] + counts[bits]
                else:
                    q_counts[q_value] = counts[bits]

    return qbits_count


def getSingleQubitProbabilities(qbits_count, shots):
    probs = {}
    for q_index in qbits_count:
        q_bit_count = qbits_count[q_index]
        for q_bit_measurement in q_bit_count:
            counts = q_bit_count[q_bit_measurement]
            if q_index in probs:
                probs[q_index][q_bit_measurement] = counts / shots
            else:
                probs[q_index] = {}
                probs[q_index][q_bit_measurement] = counts / shots
    return probs

def getFrequencies(counts, shots):
    probs = {}
    for state_index, state in enumerate(counts):
        probs[state] = counts[state] / shots

    return collections.OrderedDict(sorted(probs.items()))


def expandCounts(counts):
    measurements = []
    meas_out = list(counts.keys())
    for meas in meas_out:
        measurement = []
        for bit in meas:
            measurement.append(int(bit))
        for i in range(counts[meas]):
            measurements.append(measurement.copy())

    return measurements
