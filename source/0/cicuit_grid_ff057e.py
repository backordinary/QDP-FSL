# https://github.com/nicholas-a-karle/Quantum-Chess/blob/202061185dc57b4841c7dc9eb109510a4f102fe0/assets/cicuit_grid.py
import numpy as np
from qiskit import *
from assets.globals import *
import pygame

image_i = pygame.image.load("assets/images/gate_images/iden_gate.png")
image_h = pygame.image.load("assets/images/gate_images/h_gate.png")
image_x = pygame.image.load("assets/images/gate_images/x_gate.png")
image_y = pygame.image.load("assets/images/gate_images/y_gate.png")
image_z = pygame.image.load("assets/images/gate_images/z_gate.png")
image_t = pygame.image.load("assets/images/gate_images/t_gate.png")
backend = Aer.get_backend("qasm_simulator")

class Gate:
    def __init__(self, type = EMPTY, target = EMPTY_COORDS, control = EMPTY_COORDS):
        self.type = type
        self.target = target
        self.control = control

    def clear(self):
        self.type = EMPTY
        self.target = EMPTY_COORDS
        self.control = EMPTY_COORDS

class CircuitGrid:

    def __init__(self, num_qbits, max_gates):
        self.num_qbits = num_qbits
        self.max_gates = max_gates
        self.gates = []
        for y in range(num_qbits):
            self.gates.append([])
            for x in range(max_gates):
                self.gates[y].append(Gate(EMPTY, (x, y), EMPTY_COORDS))

    def clear(self):
        for qbit in self.gates:
            for gate in qbit:
                gate.clear()

    def set_gate(self, type = EMPTY, x = 0, y = 0, control = EMPTY):
        gate = self.gates[y][x]
        ctrl = self.gates[control][x]
        #cannot set self to self
        if gate.type == type:
            if gate.control[1] == control:
                return False
            else:
                self.clear_gate(x, y)
                return self.set_gate(type, x, y, control)


        #cannot set control to self
        if control is y:
            return False

        #check that (x, y) is not taken up by CONTROL
        if gate.type is CONTROL:
            return False
        
        #check that control is not taken by CONTROL or other gate
        if control is not EMPTY:
            if ctrl.type is not EMPTY:
                return False
            self.gates[control][x].type = CONTROL
            self.gates[y][x].control = (x, control)

        self.gates[y][x].type = type
        return True

    def clear_gate(self, x = 0, y = 0):
        if self.gates[y][x].type == EMPTY and self.gates[y][x].control == EMPTY_COORDS:
            return False
        if self.gates[y][x].control is not EMPTY_COORDS:
            self.gates[self.gates[y][x].control[1]][self.gates[y][x].control[0]] = Gate(EMPTY, (x, y), EMPTY_COORDS)
        self.gates[y][x] = Gate(EMPTY, (x, y), EMPTY_COORDS)

        return True

    def run_circuit(self, shots, simulate = True):
        c = QuantumCircuit(self.num_qbits, self.num_qbits)
        for qbit in self.gates:
            for gate in qbit:
                target = gate.target[1]
                control = gate.control[1]
                if gate.type is EMPTY:
                    continue
                elif gate.type is I_GATE:
                    if gate.control is not EMPTY_COORDS:
                        print("ERROR")
                    else:
                        c.i(target)
                elif gate.type is H_GATE:
                    if gate.control is not EMPTY_COORDS:
                        c.ch(control, target)
                    else:
                        c.h(target)
                elif gate.type is X_GATE:
                    if gate.control is not EMPTY_COORDS:
                        c.cx(control, target)
                    else:
                        c.x(target)
                elif gate.type is Y_GATE:
                    if gate.control is not EMPTY_COORDS:
                        c.cy(control, target)
                    else:
                        c.y(target)
                elif gate.type is Z_GATE:
                    if gate.control is not EMPTY_COORDS:
                        c.cz(control, target)
                    else:
                        c.z(target)
                elif gate.type is T_GATE:
                    if gate.control is not EMPTY_COORDS:
                        c.ct(control, target)
                    else:
                        c.t(target)
                else:
                    print("ERROR")
        for i in range(self.num_qbits):
            c.measure(i, i)
        # now circuit can be run on simulation or real deal
        if (simulate):
            print("Simulating QC")
            job = backend.run(c, shots=shots)
            result = job.result()
            return result.get_counts(c)
        else:
            print("Sending to QC")

    def draw(self, surface):
        for x in range(self.max_gates):
            for y in range(self.num_qbits):

                type = self.gates[y][x].type

                if type is EMPTY or type is CONTROL:
                    continue
                
                tn = self.gates[y][x].target[0]
                cn = self.gates[y][x].control[0]

                qbht = QBIT_HEIGHT[self.gates[y][x].target[1]]
                qbhc = QBIT_HEIGHT[self.gates[y][x].control[1]]

                tc = (
                    GATE_DIST + (GATE_DIST + GATE_WIDTH_FULL) * tn,
                    qbht - GATE_HEIGHT_HALF
                    )
                ctrlo = (
                    tc[0] + GATE_WIDTH_HALF,
                    qbht
                    )
                ctrle = (
                    GATE_DIST + (GATE_DIST + GATE_WIDTH_FULL) * cn + GATE_WIDTH_HALF,
                    qbhc
                    )

                if self.gates[y][x].control is not EMPTY_COORDS:
                    pygame.draw.line(surface, BLACK, ctrlo, ctrle, 1)
                    pygame.draw.circle(surface, BLACK, ctrle, 5)

                if type is I_GATE:
                    surface.blit(image_i, tc)
                elif type is H_GATE:
                    surface.blit(image_h, tc)
                elif type is X_GATE:
                    surface.blit(image_x, tc)
                elif type is Y_GATE:
                    surface.blit(image_y, tc)
                elif type is Z_GATE:
                    surface.blit(image_z, tc)
                elif type is T_GATE:
                    surface.blit(image_t, tc)
                

            
            
            
    