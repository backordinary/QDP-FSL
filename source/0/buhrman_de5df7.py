# https://github.com/TimVroomans/Quantum-Mastermind/blob/1c3ae129f2d6d7272320593bf67b233971397a7c/src/mastermind/game/algorithms/Buhrman.py
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 17:38:52 2021

@author: timvr
"""
import numpy as np
from itertools import compress
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from abc import ABC
from mastermind.game.algorithms.Find_Colours import build_find_colours_circuit
from mastermind.game.algorithms.Find_Colour_Positions import build_find_colour_positions_circuit, build_find_colour_positions_alt_circuit
from mastermind.game.game import Game

class Buhrman(Game, ABC):
    def __init__(self, num_slots=4, num_colours=4):
        
        # Initialise Quantum Game
        super(Buhrman, self).__init__(10, num_slots, num_colours, False)
        
        # constants
        self.num_slots = num_slots
        self.num_colours = num_colours
        self.n = num_slots
        self.k = num_colours
        self.logn = int(np.ceil(np.log2(self.n)))
        self.logk = int(np.ceil(np.log2(self.k)))
        
        # Guess init
        self.secret_string_guess = [-1]*self.n  # -1 to clearly indicate if something has fucked up
        
        # find which colours are used, using first part of Buhrman
        # self.used_colours = [any([b==i for b in self.sequence]) for i in range(self.k)] # beun implementation
        self.used_colours = self.find_colours()
        print("\n\nUsed colours:\n\n     %s" % (str(list(compress(list(range(self.k)),self.used_colours)))))
        
        # See if all colours are used
        self.all_colours_used = all(b == 1 for b in self.used_colours)
        
        print("\n\nColour positions:\n")
        if not self.all_colours_used:
            # If not all colours are used: simple continued algorithm
            d = self.used_colours.index(0)  # smallest unused colour
            for (c, colour_used) in enumerate(self.used_colours):
                if colour_used:
                    pos = self.find_colour_positions(c, d)
                    print("     %d: %s" % (c, str(pos)))
                    # change the guess according to the output
                    self.secret_string_guess = [c if j==1 else self.secret_string_guess[i] for (i,j) in enumerate(pos)]
        else:
            # otherwise: more complex continued alg
            pos0 = []
            for c in range(self.k):
                if c == 0:
                    pos = self.find_colour_positions_alt(c)
                    pos0 = pos
                else:
                    pos = self.find_colour_positions(c, 0, pos0)
                print("     Colour %d: %s" % (c, str(pos)))
                # change the guess according to the output
                self.secret_string_guess = [c if j==1 else self.secret_string_guess[i] for (i,j) in enumerate(pos)]
        
        print("\n\nSecret string:\n\n     %s" % (str(self.secret_string_guess)))
        
        if self.secret_string_guess == list(self.sequence):
            print("\n\nThis guess is correct!\n")
        else:
            print("\n\nThis guess is incorrect! It should be %s!\n" % (str(list(self.sequence))))
                    
            
        
    def find_colours(self):
        
        # Quantum Registers
        self.b0 = QuantumRegister(self.logk+1,'b0')
        self.x = QuantumRegister(self.k, 'x')
        self.q = QuantumRegister(self.n*self.logk, 'q')
        self.b = QuantumRegister(self.logk+1,'b')
        self.c = QuantumRegister(self.logn+1,'c')
        self.d = QuantumRegister(1,'d')
        self.e = QuantumRegister(1,'e')
        self.f = QuantumRegister(1,'f')
        # Classical register
        self.classical_x = ClassicalRegister(self.num_colours,'cx')
        
        # Circuit
        self.circuit = QuantumCircuit(self.b0,self.x,self.q,self.b,self.c,self.d,self.e,self.f,self.classical_x)
        
        # If there is no check circuit:
        if self.circuit.size() == 0:
            # Build check circuit
            build_find_colours_circuit(self.circuit, self.b0, self.x, self.q, self.b, self.c, self.d, self.e, self.f, self.sequence)
            # Measure register x
            self.circuit.measure(self.x, self.classical_x)
            
        # Run the circuit
        result = self.experiment.run(self.circuit, 1)
        counts = result.get_counts(self.circuit)
        res_x_string = list(counts.keys())[0]
        res_x = [int(bit) for bit in res_x_string]
        res_x.reverse()
        return res_x
    
    
    def find_colour_positions(self, c, d, d_positions=None):
        
        # Quantum Registers
        self.x = QuantumRegister(self.k, 'x')
        self.q = QuantumRegister(self.n*self.logk, 'q')
        self.a = QuantumRegister(self.logk+1,'a')
        # Classical register
        self.classical_x = ClassicalRegister(self.num_colours,'cx')
        
        # Circuit
        self.circuit = QuantumCircuit(self.x,self.q,self.a,self.classical_x)
        
        # If there is no check circuit:
        if self.circuit.size() == 0:
            # Build check circuit
            build_find_colour_positions_circuit(self.circuit, self.x, self.q, self.a, c, d, self.sequence, d_positions)
            # Measure register x
            self.circuit.measure(self.x, self.classical_x)
            
        # Run the circuit
        result = self.experiment.run(self.circuit, 1)
        counts = result.get_counts(self.circuit)
        res_x_string = list(counts.keys())[0]
        res_x = [int(bit) for bit in res_x_string]
        res_x.reverse()
        return res_x
    
    
    def find_colour_positions_alt(self, c):
        
        # Quantum Registers
        self.x = QuantumRegister(self.k, 'x')
        self.q = QuantumRegister(self.n*self.logk, 'q')
        self.a = QuantumRegister(self.logk+1,'a')
        self.b = QuantumRegister(self.logn+self.logk+1,'b')
        # Classical register
        self.classical_x = ClassicalRegister(self.num_colours,'cx')
        
        # Circuit
        self.circuit = QuantumCircuit(self.x,self.q,self.a,self.b,self.classical_x)
        
        # If there is no check circuit:
        if self.circuit.size() == 0:
            # Build check circuit
            build_find_colour_positions_alt_circuit(self.circuit, self.x, self.q, self.a, self.b, c, self.k, self.sequence)
            # Measure register x
            self.circuit.measure(self.x, self.classical_x)
            
        # Run the circuit
        result = self.experiment.run(self.circuit, 1)
        counts = result.get_counts(self.circuit)
        res_x_string = list(counts.keys())[0]
        res_x = [int(bit) for bit in res_x_string]
        res_x.reverse()
        return res_x
        
        
    # def Algorithm(self):
    #     # If ther is no check circuit:
    #     if self.circuit.size() == 0:
    #         # Build check circuit
    #         build_find_colours_circuit(self.circuit, self.b0, self.x, self.q, self.b, self.c, self.d, self.e, self.f, self.sequence)
    #         # Measure register x
    #         self.circuit.measure(self.x, self.classical_x)
        
    #     # Run the circuit
    #     result = self.experiment.run(self.circuit, 1)
    #     counts = result.get_counts(self.circuit)
    #     x = list(counts.keys())[0] 
    #     print(x)
    #     return x
    
    def give_feedback(self, correct, semi_correct):
        pass
    
    def random_sequence(self):
        # Choose numbers between 0 and pin_amount (do this num_slots times)
        
        # arr = np.array([2, 3, 0, 0])
        # print("\n\nWATCH OUT: RUNNING WITH HARDCODED STRING %s !!!\n\n" % (arr))
        # return arr
        return np.random.randint(0, self.pin_amount, size=self.num_slots)