# https://github.com/Therrk/QPuzzler/blob/7c8489f8fcfb2c8672e3880a36d50437ee236882/Circuit_Logic.py
import itertools
import qiskit as qs
import pygame as pg
from math import isclose, pi
from copy import copy
from numpy import array, angle, allclose
from more_itertools import interleave_longest

class Quantum_Gate():
        
    def __init__(self, cost, conditional, current_Track, current_Position, rectangle = pg.Rect(0, 0, 100, 100)):
        self.cost = cost
        self.conditional = conditional
        self.current_Track = current_Track
        self.current_Position = current_Position
        self.rectangle = rectangle

    def add_conditional(self, new_conditional):
        self.conditional = new_conditional 
        
    def unlink(self):
        if self.conditional:
            if isinstance(self, Conditional_Gate):
                self.conditional.conditional = None
                self.conditional = None
                self.aux_rectangle = pg.Rect(self.rectangle.midbottom[0], self.rectangle.midbottom[1] - 10, 30, 30)
            else:
                self.conditional.aux_rectangle = pg.Rect(self.rectangle.midbottom[0], self.rectangle.midbottom[1] - 10, 30, 30)
                self.conditional.conditional = None
                self.conditional = None
    
    def qiskit_equivalent_dispatcher(self, Quantum_Circuit):
        if hasattr(self, "conditional") and not self.conditional== None:
            self.conditional_qiskit_equivalent(Quantum_Circuit)
        else:
            self.qiskit_equivalent(Quantum_Circuit)

class Quantum_Bit:
    def __init__(self, state = array([complex(1,0), complex(0,0)])):
        self.state = state

    def set_state(self, *args):
        if len(args) == 1:
            self.state = array([complex(args[0][0],args[0][1]), complex(args[0][2],args[0][3])])
        elif len(args) == 4:
            self.state = array([complex(args[0],args[1]), complex(args[2],args[3])])
    
    def __str__(self):
        return "0:" + abs(self.state[0]) + "%, angle: " + angle(self.state[0]) + "; 1: " + abs(self.state[1]) + "%, angle: " + angle(self.state[1])

class Track(): # Classe des track sur lesquelles sont placées les gates
    def __init__(self, level, position, input = [Quantum_Bit()], deletable = False):
        self.input = input
        self.gates = []
        self.level = level
        self.position = position
        self.deletable = deletable
        self.rectangle = pg.Rect(430, 0, 1480, 150)
    
    def move_gate(self, pos, new_gate):# Sert à changer la position des gates
        has_aux_gate = hasattr(new_gate, "aux_gate")
        
        level_Tracks = self.level.tracks
        
        method_Bypass = False
        method_Bypass = isinstance(new_gate, I_Gate)
        if not method_Bypass:
            method_Bypass = (isinstance(new_gate, AUX_Gate) and self.position == 0) or (isinstance(new_gate, SWAP_Gate) and self.position + 1 == len(self.level.tracks))
        
        if not method_Bypass:
            
            if new_gate.current_Track:
                new_gate.current_Track.gates[new_gate.current_Position] = I_Gate(0,None,self,new_gate.current_Position)
        
            if has_aux_gate:
                if new_gate.aux_gate.current_Track:
                    new_gate.aux_gate.current_Track.gates[new_gate.aux_gate.current_Position] = I_Gate(0,None,self,new_gate.current_Position)
                    new_gate.aux_gate.current_Position = None
            
            new_gate.current_Position = None
            
            if not pos < len(self.gates):
                for x in range(pos - len(self.gates) +1):
                    self.gates.append(I_Gate(0,None,self, len(self.gates) + x))
        
            if has_aux_gate:
                if isinstance(new_gate, SWAP_Gate):
                    if not pos < len(level_Tracks[self.position + new_gate.aux_gate.distance].gates):
                        for x in range(pos - len(level_Tracks[self.position + new_gate.aux_gate.distance].gates) + 1):
                            level_Tracks[self.position + new_gate.aux_gate.distance].gates.append(I_Gate(0,None,self, len(self.gates) + x))
                else:
                    if not pos < len(level_Tracks[self.position - new_gate.distance].gates):
                        for x in range(pos - len(level_Tracks[self.position - new_gate.distance].gates) + 1):
                            level_Tracks[self.position - new_gate.distance].gates.append(I_Gate(0,None,self, len(self.gates) + x))
            if not new_gate.current_Track:
                updated_gate = copy(new_gate)
                if has_aux_gate:
                    updated_gate.aux_gate = copy(new_gate.aux_gate)
                    updated_gate.aux_gate.aux_gate = updated_gate
            else:
                updated_gate = new_gate
            updated_gate.current_Position = pos
            updated_gate.current_Track = self
            if isinstance(self.gates[pos], I_Gate):
                self.gates[pos] = updated_gate
            else:
                self.gates.insert(pos, updated_gate)
                after_gate = False
                for gate in self.gates:
                    if gate is updated_gate:
                        after_gate = True
                    if after_gate and not gate is updated_gate:
                        gate.current_Position +=1
                        gate.unlink()
            if has_aux_gate:
                if isinstance(updated_gate.aux_gate.current_Track.gates[pos], I_Gate):
                    updated_gate.aux_gate.current_Track.gates[pos] = updated_gate.aux_gate
                else:
                    updated_gate.aux_gate.current_Track.gates.insert(pos, updated_gate.aux_gate)
                    after_gate = False
                    for gate in updated_gate.aux_gate.current_Track.gates:
                        if gate is updated_gate.aux_gate:
                            after_gate = True
                        if after_gate and not gate is updated_gate.aux_gate:
                            gate.current_Position +=1
                            gate.unlink()
            
            self.i_gate_cleaner()
            if has_aux_gate:
                updated_gate.aux_gate.current_Track.i_gate_cleaner()
            
            return updated_gate
        else: return new_gate
        
    
    def delete_gate(self, del_gate):
        self.gates.insert(del_gate.current_Position, I_Gate(0,None,self,del_gate.current_Position))
        if hasattr(del_gate, 'aux_gate'):
            self.gates.insert(del_gate.aux_gate.current_Position, I_Gate(0,None,self,del_gate.current_Position))
            del_gate.aux_gate.current_Track.gates.remove(del_gate.aux_gate)
        self.gates.remove(del_gate)
        del_gate.unlink()
    
    def delete_track(self):
        for gate in self.gates:
            self.delete_gate(gate)
        self.level.tracks.remove(self)
    
    def i_gate_cleaner(self):
        I_gates_length = 0
        for gate_index in range(len(self.gates)):
            if isinstance(self.gates[gate_index], I_Gate):
                I_gates_length += 1
            else:
                I_gates_length = 0
        self.gates = self.gates[0:len(self.gates) - I_gates_length]
        
class Level():
    def __init__(self, output, available_gates, goal_text, name):
        self.output = output
        self.total_Cost = 0
        self.tracks = []
        self.available_gates = available_gates
        self.goal_text = goal_text
        self.name = name
        self.results = []
        self.snapshots = []

    def add_track(self, track):
        self.tracks.append(track)

    def clear(self):
        self.tracks.clear()
        
    def assign_Conditional(self, gate, conditional):
        if (gate.current_Position == conditional.current_Position and (self.tracks.index(gate.current_Track) == self.tracks.index(conditional.current_Track) + 1 or self.tracks.index(gate.current_Track) == self.tracks.index(conditional.current_Track) - 1)) and not isinstance(gate, I_Gate):
            gate.unlink()
            conditional.unlink()
            gate.conditional = conditional
            conditional.conditional = gate
            conditional.aux_rectangle = gate.rectangle.inflate(30,30)
    
    def compile(self):
        # Construction de l'objet QuantumCircuit
        self.results = []
        self.snapshots = []
        for x in range(len(self.tracks[0].input)):
            gate_series = []
            simulator = qs.Aer.get_backend("statevector_simulator")
            simulator.set_options(device='GPU')
            qr = qs.QuantumRegister(len(self.tracks))
            cr = qs.ClassicalRegister(len(self.tracks))
            qc = qs.QuantumCircuit(qr,cr)
            for qubit_index in range(len(self.tracks)):
                gate_series.append(self.tracks[qubit_index].gates)
                qc.initialize(self.tracks[qubit_index].input[x].state, qr[qubit_index])
            for gate in list(interleave_longest(*gate_series)):
                gate.qiskit_equivalent_dispatcher(qc)
            qc.snapshot("final state")
            self.results.append(qs.execute(qc, backend = simulator).result())
            self.snapshots.append(self.results[x].data()["snapshots"]["statevector"]["final state"][0])
    
    def check_if_successful(self):
        # Vérification de la solution du joueur
        victory = True
        for test_number in range(len(self.output)):
            victory = (allclose(self.output[test_number], self.snapshots[test_number]))
            if not victory:
                inputs = []
                for track in self.tracks:
                    inputs.append(track.input[test_number])
                return [victory, inputs, self.output[test_number], self.snapshots[test_number]]
        else:
            return [victory]


class Conditional_Gate(Quantum_Gate):
    def __init__(self, cost, conditional, current_Track, current_Position, rectangle = None):
        self.cost = cost
        self.conditional = conditional
        self.current_Track = current_Track
        self.current_Position = current_Position
        if rectangle is None:
            self.rectangle = pg.Rect(0, 0, 100, 100)
        else:
            self.rectangle = rectangle
        self.aux_rectangle = pg.Rect(self.rectangle.midbottom[0], self.rectangle.midbottom[1] - 10, 30, 30)
    
    def qiskit_equivalent_dispatcher(self, Quantum_Circuit):
        pass
    
    def __str__(self):
        return "if"
    
    def __repr__(self):
        return self.__str__()
    
    def __copy__(self):
        return Conditional_Gate(self.cost,self.conditional,self.current_Track, self.current_Position, self.rectangle.copy())
    


# Ici, on définit toutes les gates qui seront dans le jeu
# Elles sont toutes pareilles, sauf pour le Qiskit_Equivalent
class SWAP_Gate(Quantum_Gate):
    
    def __init__(self, cost, conditional, current_Track, current_Position, rectangle = None, aux_gate = None):
        self.cost = cost
        self.SWAPconditional = conditional
        self.SWAPcurrent_Track = current_Track
        self.SWAPcurrent_Position = current_Position
        if aux_gate is None:
            self.aux_gate = AUX_Gate(cost, self)
        else:
            self.aux_gate = aux_gate
        if rectangle is None:
            self.rectangle = pg.Rect(430, 0, 100, 100)
        else:
            self.rectangle = rectangle
    
    def __setattr__(self, name, value):
        if name == "conditional":
            self.SWAPconditional = value
        elif name == "current_Track":
            self.SWAPcurrent_Track = value
            if not self.SWAPcurrent_Track.level.tracks[self.SWAPcurrent_Track.position + self.aux_gate.distance] == value:
                self.aux_gate.SWAPcurrent_Track = self.SWAPcurrent_Track.level.tracks[self.SWAPcurrent_Track.position + self.aux_gate.distance]
        elif name == "current_Position":
            self.SWAPcurrent_Position = value
            if not self.aux_gate.current_Position == value:
                self.aux_gate.current_Position = value
        elif name == "distance":
            self.aux_gate.distance = value
        else:
            super(SWAP_Gate, self).__setattr__(name, value)
    
    def __getattr__(self, name):
        if name == "conditional":
            return self.SWAPconditional
        elif name == "current_Track":
            return self.SWAPcurrent_Track
        elif name == "current_Position":
            return self.SWAPcurrent_Position
        elif name == "distance":
            return self.aux_gate.distance
        else: raise AttributeError
    
    def __str__(self):
        return "SWAP"
    
    def __repr__(self):
        return self.__str__()
    
    def qiskit_equivalent(self, Quantum_Circuit):
        Quantum_Circuit.barrier()
        Quantum_Circuit.swap(Quantum_Circuit.qubits[self.current_Track.Position], self.target_Track)  
        return Quantum_Circuit
    
    def conditional_qiskit_equivalent(self, Quantum_Circuit):
        Quantum_Circuit.barrier()
        Quantum_Circuit.cswap(Quantum_Circuit.qubits[self.conditional.current_Track.position] , Quantum_Circuit.qubits[self.current_Track.Position], Quantum_Circuit.qubits[self.aux_gate.current_Track.Position])
        return Quantum_Circuit
    
    def __copy__(self):
        return SWAP_Gate(self.cost,self.conditional,self.current_Track, self.current_Position, self.rectangle.copy())
class AUX_Gate(Quantum_Gate):
    
    def __init__(self, cost, aux_gate):
        
        self.cost = cost
        self.SWAPconditional = None
        self.SWAPcurrent_Track = None
        self.SWAPcurrent_Position = 0
        self.distance = 1
        self.rectangle = pg.Rect(430, 0, 100, 100)
        self.aux_gate = aux_gate
    
    def __setattr__(self, name, value):
        if name == "conditional":
            self.aux_gate.SWAPconditional = value
        elif name == "current_Track":
            self.SWAPcurrent_Track = value
            if not self.SWAPcurrent_Track.level.tracks[self.SWAPcurrent_Track.position - self.distance] == value:
                self.aux_gate.SWAPcurrent_Track = self.SWAPcurrent_Track.level.tracks[self.SWAPcurrent_Track.position - self.distance]
        elif name == "current_Position":
            self.SWAPcurrent_Position = value
            try:
                if not self.aux_gate.current_Position == value:
                    self.aux_gate.current_Position = value
            except AttributeError:
                self.aux_gate.current_Position = value
        else:
            super(AUX_Gate, self).__setattr__(name, value)
    
    def __getattr__(self, name):
        if name == "conditional":
            return self.SWAPconditional
        elif name == "current_Track":
            return self.SWAPcurrent_Track
        elif name == "current_Position":
            return self.SWAPcurrent_Position
        else: raise AttributeError
    
    def __str__(self):
        return "SWAP"
    
    def __repr__(self):
        return "aux"
    
    def qiskit_equivalent_dispatcher(self, Quantum_Circuit):
        pass
    
    def __copy__(self):
        return AUX_Gate(self.cost, self.aux_gate)

class H_Gate(Quantum_Gate):

    def __str__(self):
        return "H"

    def __repr__(self):
        return self.__str__()
    
    def qiskit_equivalent(self, Quantum_Circuit):
        Quantum_Circuit.barrier()
        Quantum_Circuit.h(Quantum_Circuit.qubits[self.current_Track.position])
        return Quantum_Circuit
    
    def conditional_qiskit_equivalent(self, Quantum_Circuit):
        Quantum_Circuit.barrier()
        Quantum_Circuit.ch(Quantum_Circuit.qubits[self.conditional.current_Track.position], Quantum_Circuit.qubits[self.current_Track.position])
        return Quantum_Circuit
    
    def __copy__(self):
        return H_Gate(self.cost,self.conditional,self.current_Track, self.current_Position, self.rectangle.copy())

class X_Gate(Quantum_Gate):
    
    def __str__(self):
        return "X"

    def __repr__(self):
        return self.__str__()
    
    def qiskit_equivalent(self, Quantum_Circuit):
        Quantum_Circuit.barrier()
        Quantum_Circuit.x(Quantum_Circuit.qubits[self.current_Track.position])
        return Quantum_Circuit
    
    def conditional_qiskit_equivalent(self, Quantum_Circuit):
        Quantum_Circuit.barrier()
        Quantum_Circuit.cnot(Quantum_Circuit.qubits[self.conditional.current_Track.position], Quantum_Circuit.qubits[self.current_Track.position])
        return Quantum_Circuit
    
    def __copy__(self):
        return X_Gate(self.cost,self.conditional,self.current_Track, self.current_Position, self.rectangle.copy())

class T_Gate(Quantum_Gate):

    def __str__(self):
        return "T"

    def __repr__(self):
        return self.__str__()
    
    def qiskit_equivalent(self, Quantum_Circuit):
        Quantum_Circuit.barrier()
        Quantum_Circuit.p(pi/4, Quantum_Circuit.qubits[self.current_Track.position])
        return Quantum_Circuit
    
    def conditional_qiskit_equivalent(self, Quantum_Circuit):
        Quantum_Circuit.barrier()
        Quantum_Circuit.cp(pi/4, Quantum_Circuit.qubits[self.conditional.current_Track.position], Quantum_Circuit.qubits[self.current_Track.position])
        return Quantum_Circuit
    
    def __copy__(self):
        return T_Gate(self.cost,self.conditional,self.current_Track, self.current_Position, self.rectangle.copy())

class Z_Gate(Quantum_Gate):

    def __str__(self):
        return "Z"

    def __repr__(self):
        return self.__str__()
    
    def qiskit_equivalent(self, Quantum_Circuit):
        Quantum_Circuit.barrier()
        Quantum_Circuit.p(pi, Quantum_Circuit.qubits[self.current_Track.position])
        return Quantum_Circuit
    
    def conditional_qiskit_equivalent(self, Quantum_Circuit):
        Quantum_Circuit.barrier()
        Quantum_Circuit.cp(pi, Quantum_Circuit.qubits[self.conditional.current_Track.position], Quantum_Circuit.qubits[self.current_Track.position])
        return Quantum_Circuit
    
    def __copy__(self):
        return Z_Gate(self.cost,self.conditional,self.current_Track, self.current_Position, self.rectangle.copy())

class S_Gate(Quantum_Gate):

    def __str__(self):
        return "S"

    def __repr__(self):
        return self.__str__()
    
    def qiskit_equivalent(self, Quantum_Circuit):
        Quantum_Circuit.barrier()
        Quantum_Circuit.p(pi/2, Quantum_Circuit.qubits[self.current_Track.position])
        return Quantum_Circuit
    
    def conditional_qiskit_equivalent(self, Quantum_Circuit):
        Quantum_Circuit.barrier()
        Quantum_Circuit.cp(pi/2, Quantum_Circuit.qubits[self.conditional.current_Track.position], Quantum_Circuit.qubits[self.current_Track.position])
        return Quantum_Circuit
    
    def __copy__(self):
        return S_Gate(self.cost,self.conditional,self.current_Track, self.current_Position, self.rectangle.copy())
    
class I_Gate(Quantum_Gate):
    
    def __str__(self):
        return "I"
    
    def __repr__(self):
        return self.__str__()
    
    def qiskit_equivalent_dispatcher(self, Quantum_Circuit):
        Quantum_Circuit.barrier()
        Quantum_Circuit.id(Quantum_Circuit.qubits[self.current_Track.position])
        return Quantum_Circuit
    
    def __copy__(self):
        return I_Gate(self.cost,self.conditional,self.current_Track, self.current_Position, self.rectangle.copy())
