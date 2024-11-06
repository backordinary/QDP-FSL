# https://github.com/Karen-Shekyan/Variational-Quantum-Eigensolver/blob/a28525f7c5ea8bafb318044eeddcd26ad02b4522/qiskit%20code%20(Python)/UnitTest.py
# Copyright 2021 The MITRE Corporation. All Rights Reserved.

import unittest

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute
from qiskit import Aer
import random
from qiskit.opflow import I, X, Y, Z
import main
import numpy as np


class VQETests(unittest.TestCase):

    def test1(self):
        electrons = 1
        orbitals = 4

        thetaArray = np.array([])
        for i in range(orbitals*2):
            thetaArray = np.append(thetaArray, [1])

        # print(thetaArray)
        standardError = 0.5
    #     hamiltonian =  (-1.0523732 * I^I) + (0.39793742 * I^Z) + (-0.3979374 * Z^I) \
    # + (-0.0112801 * Z^Z) + (0.18093119 * X^X)
        hamiltonian = (-0.27293036432559103 * I^I) + (0.03963943913641628 * I^Z) + (0.03963943913641629 * I^Z) + (0.19365148605275848 * Z^Z)

        result = main.VQE(electrons, orbitals, standardError, thetaArray, hamiltonian)
        print("\n")
        print(result)

    def test2(self):
        
        # print("Test 2 passed!")
        return


    def test3(self):
        
        # print("Test 3 passed!")
        return


if __name__ == '__main__':
    unittest.main()