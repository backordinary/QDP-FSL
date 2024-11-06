# https://github.com/RLogik/qiskit-course/blob/4c5ec35cfd0084b51429da944d8964a03ab668e3/src/thirdparty/quantum.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# IMPORTS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import qiskit as qk;
from qiskit import Aer as QkBackendAer;
from qiskit import ClassicalRegister;
from qiskit import IBMQ;
from qiskit import QuantumCircuit;
from qiskit.circuit.gate import Gate as QkGate;
from qiskit import QuantumRegister;
from qiskit import assemble as QkAssemble;
from qiskit import execute as QkExecute;
from qiskit import transpile as QkTranspile;
from qiskit import visualization as QkVisualisation;
from qiskit.providers import ibmq;
from qiskit.providers import Backend as QkBackend;
from qiskit.providers.ibmq.job.ibmqjob import IBMQJob;
from qiskit.providers.ibmq.ibmqbackend import IBMQSimulator;
from qiskit.providers.ibmq.accountprovider import AccountProvider as QkAccountProvider;
# from qiskit.providers.ibmq import least_busy;
from qiskit.quantum_info import Statevector as QkStatevector;
from qiskit.quantum_info import Operator as QkOperator;
from qiskit.result.result import Result as QkResult;
from qiskit.tools import jupyter as QkJupyter;
from qiskit_textbook import problems as QkProblems;


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# EXPORTS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

__all__ = [
    'qk',
    'ClassicalRegister',
    'ibmq',
    'IBMQ',
    'IBMQJob',
    'IBMQSimulator',
    'QkAccountProvider',
    'QkAssemble',
    'QkBackend',
    'QkBackendAer',
    'QkExecute',
    'QkGate',
    'QkJupyter',
    'QkOperator',
    'QkProblems',
    'QkResult',
    'QkStatevector',
    'QkTranspile',
    'QkVisualisation',
    'QuantumCircuit',
    'QuantumRegister',
];
