# https://github.com/AmandaStromdahl/DD2480_CodeComplexity_Coverage/blob/9f94b7fbfce6cf56afa07a0ecd0b061e1c5d8c69/complex_functions/deutsch_jozsa.py
import numpy as np
import qiskit as q

# This algorithm is taken from the GitHub repository: https://github.com/TheAlgorithms/Python

def dj_oracle(case: str, num_qubits: int) -> q.QuantumCircuit:
    """
    Returns a Quantum Circuit for the Oracle function.
    The circuit returned can represent balanced or constant function,
    according to the arguments passed
    """
    # This circuit has num_qubits+1 qubits: the size of the input,
    # plus one output qubit
    oracle_qc = q.QuantumCircuit(num_qubits + 1)

    # First, let's deal with the case in which oracle is balanced
    
    # Decision 1
    if case == "balanced":
        # First generate a random number that tells us which CNOTs to
        # wrap in X-gates:
        b = np.random.randint(1, 2**num_qubits)
        # Next, format 'b' as a binary string of length 'n', padded with zeros:
        b_str = format(b, f"0{num_qubits}b")
        # Next, we place the first X-gates. Each digit in our binary string
        # correspopnds to a qubit, if the digit is 0, we do nothing, if it's 1
        # we apply an X-gate to that qubit:
        
        # Decision 2
        for index, bit in enumerate(b_str):
            
            # Decision 3
            if bit == "1":
                oracle_qc.x(index)
        # Do the controlled-NOT gates for each qubit, using the output qubit
        # as the target:
        
        # Decision 4
        for index in range(num_qubits):
            oracle_qc.cx(index, num_qubits)
        # Next, place the final X-gates
        
        # Decision 5
        for index, bit in enumerate(b_str):
            #Decision 6
            if bit == "1":
                oracle_qc.x(index)

    # Case in which oracle is constant
    
    # Decision 7
    if case == "constant":
        # First decide what the fixed output of the oracle will be
        # (either always 0 or always 1)
        output = np.random.randint(2)
        
        # Decision 8
        if output == 1:
            oracle_qc.x(num_qubits)

    oracle_gate = oracle_qc.to_gate()
    oracle_gate.name = "Oracle"  # To show when we display the circuit
    
    # Exit 1
    return oracle_gate

    """
    Lizard CCN: 9
    Manual CCN: 9
        8 decisions
        1 exit
        CCN = 8 - 1 + 2
    """