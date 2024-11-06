# https://github.com/AlejandroGomezFrieiro/NQubitALU/blob/cdcaa38f2b252a261977865739b0eb56e95461f0/NQubitALU.py
import numpy as np
import qiskit
from qiskit.visualization import plot_state_city
# import qiskit.circuits.library as lib
from qiskit import QuantumRegister
from typing import Optional, List

class HalfAdder(qiskit.circuit.Gate):

    
    def __init__(self) -> None:
        super().__init__('halfAdder', 4, [])

    def _define(self):
        """
        gate halfAdder(a, b, sum, carry)
        {
            cx a, sum;
            cx b, sum;
            ccx a, b, carry;
        }
        """
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        q = qiskit.QuantumRegister(4, name='q')
        qc = qiskit.QuantumCircuit(q, name=self.name)
        # XOR q0, q1 to q2
        qc.cx(0, 2)
        qc.cx(1, 2)

        # AND q0, q1 to q3
        qc.ccx(0, 1, 3)

        self.definition = qc

        # if self.num_variable_qubits < 4:
        #     raise ValueError

class FullAdder(qiskit.circuit.Gate):
    def __init__(self) -> None:
        super().__init__('fullAdder', 8, [])

    def _define(self):
        """
        gate halfAdder(a, b, cin, cout, sum, ancilla1, ancilla2, ancilla3)
        {
            cx a, ancilla1;
            cx b, ancilla1;
            cx cin, cout;
            cx sum, cout;
            ccx a, b, ancilla2;
            ccx cin, ancilla1, ancilla3;
            cx ancilla3, cout;
            cx ancilla2, cout;
            ccx ancilla2, ancilla3, cout;
        }
        """
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        q = qiskit.QuantumRegister(8, name='q')
        qc = qiskit.QuantumCircuit(q, name=self.name)
        # XOR q0, q1 to q5
        qc.cx(0, 5)
        qc.cx(1, 5)

        # XOR q2, q5 to q4
        qc.cx(2, 4)
        qc.cx(5, 4)

        # AND q0, q1 to q6
        qc.ccx(0, 1, 6)

        # AND q2, q5 to q7
        qc.ccx(2, 5, 7)

        # OR q7, q6, q3
        qc.cx(7, 3)
        qc.cx(6, 3)
        qc.ccx(7, 6, 3)

        self.definition = qc

class TwoQubitALU(qiskit.QuantumCircuit):
    def __init__(self,
                 num_qubits: int = 12) -> None:
        """Return a circuit implementing a Two Qubit ALU, with input qubits in the form (a0, a1, b0, b1, s0, s1, c0, c1, sb, ancilla1, ancilla2, ancilla3)

        Args:
            num_qubits: the width of circuit.

        Raises:
            ValueError: If the number of qubits is not right.
        """
        super().__init__(num_qubits, name="ALU(2)")

        if num_qubits != 12:
            raise ValueError("ALU(2) requires 13 bits")

        q = self.qregs[0]
        # B0 XOR SB to B0 (use ancilla 1, swap b0, ancilla1, then reset ancilla1)
        self.cx(2, 9)
        self.cx(8, 9)
        self.swap(2, 9)
        self.reset([9]*10)

        # B1 XOR SB TO B1 (use ancilla 1, swap b1, ancilla2, then reset ancilla2)
        self.cx(3, 10)
        self.cx(8, 10)
        self.swap(3, 10)
        self.reset([10]*10)

        # (a0, a1, b0, b1, s0, s1, c0, c1, sb, ancilla1, ancilla2, ancilla3)
        # a, b, cin, cout, sum, ancilla1, ancilla2, ancilla3
        # SUM A0, B0, reset ancillas
        self.append(FullAdder(), [q[i] for i in [0, 2, 8, 6, 4, 9, 10, 11]]) #a0, b0, sb, c0, s0, ancillas
        self.reset([9]*10)
        self.reset([10]*10)
        self.reset([11]*10)

        # SUM A1, B1, reset ancillas
        self.append(FullAdder(), [q[i] for i in [1, 3, 6, 7, 5, 9, 10, 11]])
        self.reset([9]*10)
        self.reset([10]*10)
        self.reset([11]*10)

class NQubitALU(qiskit.circuit.library.BlueprintCircuit):
    def __init__(self,
                registerA: qiskit.QuantumRegister,
                registerB: qiskit.QuantumRegister,
                registerS: qiskit.QuantumRegister,
                registerSB: qiskit.QuantumRegister,
                registerC: qiskit.QuantumRegister,
                registerAncilla: qiskit.QuantumRegister,
                num_qubits: Optional[int] = None,
                name: str = 'ALU(N)') -> None:
        """Return a circuit implementing a simple N Qubit ALU (named ALU(N)), with input qubits in the form (a0, ... an, b0, b ...bn, s0...sn, c0...cn, sb, ancilla1, ancilla2, ancilla3)

        Args:
            registerA: First Quantum register of size N, which will be the first string to operate on.
            registerB: Second Quantum register of size N, which will be the second string to operate on.
            registerS: Third Quantum register of size N, which will contain the output string.
            registerSB: Single qubit quantum register which controls between operations ADD and SUB.
            registerC: Fourth Quantum register of size N, which contains the carry bits.
            registerAncilla: Quantum register of size 3 to operate Ancillas.
        Raises:
            CircuitError: if the xor bitstring exceeds available qubits.

        Reference Circuit:
            .. jupyter-execute::
                :hide-code:

                from qiskit.circuit.library import XOR
                import qiskit.tools.jupyter
                circuit = XOR(5, seed=42)
                %circuit_library_info circuit
        """
        self.registerA = registerA
        self.registerB = registerB
        self.registerS = registerS
        self.registerSB = registerSB
        self.registerC = registerC
        self.registerAncilla = registerAncilla
        self._num_qubits = None
        self.num_qubits = registerA.size # Size of the Nbits to operate on
        self._name = f'ALU({self.registerA.size})'
        self.qregs = [self.registerA, self.registerB, self.registerS, self.registerSB, self.registerC, self.registerAncilla]
        super().__init__(*self.qregs, name=self._name)
        self.qregs = [self.registerA, self.registerB, self.registerS, self.registerSB, self.registerC, self.registerAncilla]

        
        

    @property
    def num_qubits(self) -> int:
        """The number of qubits to be summed.

        Returns:
            The number of qubits per main register.
        """
        return self._num_qubits

    @num_qubits.setter
    def num_qubits(self, num_qubits: int) -> None:
        """Set the number of qubits in the registers of the ALU operation.

        Args:
            num_qubits: The new number of qubits.
        """
        if self._num_qubits is None or num_qubits != self._num_qubits:
            self._invalidate()
            self._num_qubits = num_qubits
            self._reset_registers()

    def _reset_registers(self):
        qr_A = self.registerA
        qr_B = self.registerB
        qr_S = self.registerS
        qr_SB = self.registerSB
        qr_C = self.registerC
        qr_An = self.registerAncilla
        self.qregs = [qr_A, qr_B, qr_S, qr_SB, qr_C, qr_An]


    @property
    def num_ancilla_qubits(self) -> int:
        """The number of ancilla qubits required to implement the ALU(operation).

        Returns:
            The number of ancilla qubits in the circuit.
        """
        return 3
            
    @property
    def num_carry_qubits(self) -> int:
        """The number of carry qubits required to compute the ALU.

        Note that this is not necessarily equal to the number of ancilla qubits, these can
        be queried using ``num_ancilla_qubits``.

        Returns:
            The number of carry qubits required to compute the sum.
        """
        return self.num_qubits

    def _check_configuration(self, raise_on_failure=True):
        valid = True
        if self._num_qubits is None:
            valid = False
            if raise_on_failure:
                raise AttributeError('The input register has not been set.')
        if not (self.registerA.size and self.registerB.size and self.registerS.size and self.registerC.size):
            valid = False
            if raise_on_failure:
                raise ValueError('Register sizes are not equal.')
        if self.registerSB.size != 1:
            valid = False
            if raise_on_failure:
                raise ValueError('Control qubit register must be of size 1.')
        if self.registerAncilla.size < 3:
            valid = False
            if raise_on_failure:
                raise ValueError('Ancilla register needs at least three qubits.')
        return valid

    def _build(self):
        super()._build()

        qr_A = self.registerA
        qr_B = self.registerB
        qr_S = self.registerS
        qr_SB = self.registerSB
        qr_carry = self.registerC
        qr_ancilla = self.registerAncilla

        for qubit in qr_ancilla:
            self.reset([qubit]*1)

        for B_index in range(qr_B.size):
            self.cx(qr_B[B_index], qr_ancilla[0])
            self.cx(qr_SB[0], qr_ancilla[0])
            self.swap(qr_B[B_index], qr_ancilla[0])
            self.reset([qr_ancilla[0]]*1)
        
        for A_index in range(qr_A.size):
            if A_index == 0:
                self.append(FullAdder(), [qubit for qubit in [qr_A[A_index], qr_B[A_index], qr_SB, qr_carry[A_index],
                                                        qr_S[A_index], qr_ancilla[0], qr_ancilla[1], qr_ancilla[2]]]) #a0, b0, sb, c0, s0, ancillas
                self.reset([qr_ancilla[0]]*1)
                self.reset([qr_ancilla[1]]*1)
                self.reset([qr_ancilla[2]]*1)
            else:
                self.append(FullAdder(), [qubit for qubit in [qr_A[A_index], qr_B[A_index], qr_carry[A_index - 1], qr_carry[A_index],
                                                        qr_S[A_index], qr_ancilla[0], qr_ancilla[1], qr_ancilla[2]]]) #a0, b0, sb, c0, s0, ancillas
                self.reset([qr_ancilla[0]]*1)
                self.reset([qr_ancilla[1]]*1)
                self.reset([qr_ancilla[2]]*1)

def set_register_from_classical_register(quantumRegister: qiskit.QuantumRegister, classicalRegister: qiskit.ClassicalRegister):
    '''
    Sets the qubits in a quantum register to the classical values of a classic register.
    Cannot be implemented as of current Qiskit version without measuring a circuit.

    Input:
        quantumRegister: A quantum register in its ground state.
        classicalRegister: Target register to set. 
    '''
    if quantumRegister.size != classicalRegister.size:
        raise ValueError("Classic register and quantum register are not of same size")
    pass

def set_quantum_register_from_string(circuit: qiskit.QuantumCircuit,
                                    quantumRegister: qiskit.QuantumRegister,
                                    input_string: str,
                                    n_resets: int = 1):
    """
    Resets quantumRegister and sets it to the values of the input string. Appends the necessary gates into circuit.
    """
    N = len(input_string)
    if quantumRegister.size != N:
        raise ValueError("Classic register and quantum register are not of same size")
    circuit.reset([qubit for qubit in quantumRegister]*n_resets)
    for characters in range(N-1, -1, -1):
        if input_string[characters] == '1':
            circuit.x(quantumRegister[N - characters - 1])

if __name__ == "__main__":
    from qiskit import Aer
    import qiskit

    import itertools
    from qiskit.result.utils import count_keys

    # Define length of strings, as well as the strings you would like to sum and the control bit
    N = 2
    stringA = '10'
    stringB = '01'
    stringSB = '0'

    # Initialize quantum and classic registers
    registerA = qiskit.QuantumRegister(N)
    registerB = qiskit.QuantumRegister(N)
    registerS = qiskit.QuantumRegister(N)
    registerSB = qiskit.QuantumRegister(1)
    registerC = qiskit.QuantumRegister(N)
    registerAnc = qiskit.QuantumRegister(3)
    measurementRegister = qiskit.ClassicalRegister(4+4*N)
    regs = [registerA, registerB, registerS, registerSB, registerC, registerAnc]
    flat_regs = list(itertools.chain(*[register[:] for register in regs]))

    # Create a circuit
    circuit = qiskit.QuantumCircuit(*regs, measurementRegister,  name = 'ALU')
    
    # Set input registers
    set_quantum_register_from_string(circuit, registerA, stringA)
    set_quantum_register_from_string(circuit, registerB, stringB)

    # Set SB to choose between add or sub
    # If dding a Hadamard gate, then calculate both ADD and SUB
    set_quantum_register_from_string(circuit, registerSB, stringSB)
    circuit.h(registerSB)

    # Add NQubitALU to the circuit
    circuit.append(NQubitALU(*regs).to_instruction(), flat_regs)
    
    # Add measurement operations to the circuit
    circuit.barrier()
    circuit.measure(range(4+4*N), range(4+4*N))

    # Display the circuit
    print(circuit)

    # Use Aer's qasm_simulator
    simulator = Aer.get_backend('qasm_simulator')
    n_shots = 1024

    # Execute the circuit on the qasm simulator
    job = qiskit.execute(circuit, simulator, shots=n_shots, meas_return = 'single')

    # Grab results from the job
    result = job.result()

    # Return counts
    counts = result.get_counts(circuit)

    bit_results = list(counts.keys())
    print(f"\nInputs are A = {stringA}, B = {stringB}, SB = {stringSB}")
    print("\nTotal counts are:", counts)
    print("\nSum of A + (SB XOR B) is: \n")
    for result in bit_results:
        print(f"{result[4+N:4+2*N]} with a number of counts {counts.get(result)}/{n_shots}")