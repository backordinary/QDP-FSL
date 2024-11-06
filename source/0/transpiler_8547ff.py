# https://github.com/UTokyo-ICEPP/aqcel/blob/e61f22223131a9ddc81f9e219a597e922f6d74e7/icepp/compiler/transpiler.py
from qiskit import *
from qiskit.compiler import transpile
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Unroller

from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit


class transpiler():
    
    def __init__(self, qc, backend, backend_tket, level):
        self.qc = qc
        self.backend = backend
        self.backend_tket = backend_tket
        self.level = level
        
    def transpile(self):
        
        if self.level != 0:
            qc = transpile(self.qc, basis_gates=['id','x','sx','rz','cx','reset'])
            tket_qc = qiskit_to_tk(qc)
            self.backend_tket.compile_circuit(tket_qc, optimisation_level=2)
            qc = tk_to_qiskit(tket_qc)
        else:
            qc = self.qc
        
        transpiled_qc = transpile(circuits=qc, backend=self.backend, basis_gates=None, seed_transpiler=1, optimization_level=self.level)
        
        return transpiled_qc