# https://github.com/tjong-sc/GAOP/blob/1d323592c85336f5f401e95541d097ecdf9fae98/GAOP/QiskitDrivers/OpenQASMAdapter.py
from qiskit import QuantumCircuit, Aer, execute
from typing import List, Tuple
import itertools
import re
import numpy as np
class OpenQASMAdapter:
    """
    This class serves as the OpenQASM runtime for our GP programs
    """
    REGISTER_PATTERN = re.compile(r'qreg q\[(\d?)\];')

    @classmethod
    def printCircuit(cls, program: str) -> None:
        qc = QuantumCircuit.from_qasm_str(program)
        print(qc)

    @classmethod
    def execute(cls, program : str) -> Tuple[dict, dict]:
        """
        This method executes a given string of program instrutions to be executed by the backend.
        Tentatively, it outputs a state vector with the corresponding broch complex number representation.
        namely: sqrt(2) <-- 100%, sqrt(2)/2 <--- 50%, etc

        TODOs: put executions on a thread with a timer to terminate them. We need to also decide on
        ways to assign the fitness values if the program never terminates.  
        Outputs: 
            dict1: raw data from the state vector of qiskit
            dict2: the computed probability from the state vector (more useful.)
        """

        #this re retrieves the number of registers for a particular openQASM program
        match = cls.REGISTER_PATTERN.search(program)
        regs = int(match.group(1))

        backend = Aer.get_backend("statevector_simulator")
        qc = QuantumCircuit.from_qasm_str(program)

        # Execute the circuit and show the result.
        job = execute(qc, backend)
        result = job.result()

        return result.data(qc) , cls.__generatePermutations(regs, result.get_counts())

    @classmethod
    def __generatePermutations(cls, registers: int, counts:dict) -> dict:
        #creates a list consisting of permutations of 0 and 1s based on the size of registers
        binaries = ["".join(seq) for seq in itertools.product("01", repeat=registers)]
        #converts the list into set
        it =  iter(binaries)
        binaries = dict(zip(it, np.empty(len(binaries))))
        #includes the probabilities into the set
        for key, val in counts.items():
            binaries[str(key)] = val
        return binaries