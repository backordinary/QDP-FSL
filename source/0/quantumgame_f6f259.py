# https://github.com/TimVroomans/Quantum-Mastermind/blob/b3c814c35e16f697c0fdb291a6c3a10ed6036a06/src/mastermind/game/quantumgame.py
from abc import ABC
import numpy as np
import mastermind.game.algorithms.Mastermind_Oracle as oracle
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit

from experiment.qiskit_experiment import QiskitExperiment
from .game import Game

class QuantumGame(Game, ABC):
    def __init__(self, turns=10, num_slots=4, colour_amount=6, ask_input=True, do_v2_b_oracle=True):
        # Get some relevant numbers
        self.amount_colour_qubits = int(np.ceil(np.log2(colour_amount)))
        self.amount_answer_qubits = int(np.ceil(np.log2(num_slots))) + 1
        
        # Save input setting
        self.do_v2_b_oracle = do_v2_b_oracle
        
        # Query register
        self.q = QuantumRegister(self.amount_colour_qubits * num_slots, 'q')
        
        # Answer pin registers
        self.a = QuantumRegister(self.amount_answer_qubits, 'a')
        self.b = QuantumRegister(self.amount_answer_qubits, 'b')
        if self.do_v2_b_oracle:
            self.c = QuantumRegister(self.amount_answer_qubits, 'c')
            self.d = QuantumRegister(1, 'd')
        self.classical_a = ClassicalRegister(self.amount_answer_qubits, 'ca')
        self.classical_b = ClassicalRegister(self.amount_answer_qubits, 'cb')
        
        # Build circuit from registers
        if self.do_v2_b_oracle:
            self.circuit = QuantumCircuit(self.q, self.a, self.b, self.c, self.d, self.classical_a, self.classical_b)
        else:
            self.circuit = QuantumCircuit(self.q, self.a, self.b, self.classical_a, self.classical_b)
        
        # Set up qiskit experiment
        self.experiment = QiskitExperiment()
        
        # Initialise Mastermind
        super(QuantumGame, self).__init__(turns, num_slots, colour_amount, ask_input)


    def check_input(self, query, secret_sequence):
        # If there is no check circuit:
        if self.circuit.size() == 0:
            # Build check circuit
            oracle.build_mastermind_a_circuit(self.circuit, self.q, self.a, secret_sequence)
            if self.do_v2_b_oracle:
                oracle.build_mastermind_b_circuit_v2(self.circuit, self.q, self.b, self.c, self.d, secret_sequence)
            else:
                oracle.build_mastermind_b_circuit(self.circuit, self.q, self.b, secret_sequence)
            # Measure registers a and b
            self.circuit.measure(self.a, self.classical_a)
            self.circuit.measure(self.b, self.classical_b)
        
        # Prepare q register in query
        binary_query = [bin(q)[2:].zfill(self.amount_colour_qubits) for q in query]
        if self.do_v2_b_oracle:
            prep_q = QuantumCircuit(self.q, self.a, self.b, self.c, self.d, self.classical_a, self.classical_b)
        else:
            prep_q = QuantumCircuit(self.q, self.a, self.b, self.classical_a, self.classical_b)
        for (i,binary) in enumerate(binary_query):
            for (j,bit) in enumerate(binary[::-1]):
                if bit == '1':
                    prep_q.x(self.q[i*self.amount_colour_qubits + j])
                else:
                    prep_q.i(self.q[i*self.amount_colour_qubits + j])
        
        
        # Finish circuit with q prepped in query
        self.circuit = prep_q + self.circuit
        
        # Run the circuit
        result = self.experiment.run(self.circuit, 1)
        counts = result.get_counts(self.circuit)
        meas_ab = list(counts.keys())[0]
        meas_ab = meas_ab.split()
        
        # add prep_q again to erase its effect XX = I
        self.circuit = prep_q + self.circuit
        
        a = int(meas_ab[1], 2)
        b = int(meas_ab[0], 2)
        
        correct = a
        semi_correct = b - a
        
        return correct, semi_correct


    def random_sequence(self):
        # Choose numbers between 0 and pin_amount (do this num_slots times)
        
        # arr = np.array([0, 0, 1, 1])
        # print("\n\nWATCH OUT: RUNNING WITH HARDCODED STRING %s !!!\n\n" % (arr))
        # return arr
        return np.random.randint(0, self.pin_amount, size=self.num_slots)