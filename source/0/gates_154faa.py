# https://github.com/HaleyLin2006/Qu.Race/blob/1a8f840b209de130b70e4f75c17d0b07008f9916/Gates.py
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
import pygame
from Button import Button 
from numpy import pi


pygame.init()

class Q_Circuit():
    def start(self, qubits):
        size = qubits
        qubit = QuantumRegister(size,'q')
        for i in range(qubits):
            bits = ClassicalRegister((i+1), 'c')
            circuit = QuantumCircuit(qubit, bits)
        return circuit

    #Gates
    #1 Qubit
    def H(self, circuit, q_circuit, qubit):
        circuit.h(q_circuit[qubit])

    def NOT(self, circuit, q_circuit, qubit):
        circuit.x(q_circuit[qubit])

    def Y(self, circuit, q_circuit, qubit):
        circuit.y(q_circuit[qubit])

    def Z(self, circuit, q_circuit, qubit):
        circuit.z(q_circuit[qubit])

    #2 Qubit
    def CNOT(self, circuit, q_circuit, q_1, q_2):
        circuit.cx(q_circuit[q_1], q_circuit[q_2])
    
    def Swap(self, circuit, q_circuit, q_1, q_2):
        circuit.swap(q_circuit[q_1], q_circuit[q_2])
    #Phase
    #pi/4
    def S(self, circuit, q_circuit, qubit):
        circuit.s(q_circuit[qubit])
    #pi/2
    def T(self, circuit, q_circuit, qubit):
        circuit.t(q_circuit[qubit])

class Gates(pygame.sprite.Sprite):
    def __init__(self, screen, x, y, gate_type="", Mouse_Press=False, color=(235, 79, 52)):
        pygame.sprite.Sprite.__init__(self)
        self.screen = screen
        self.color = color
        self.x = x
        self.y = y
        self.Mouse_Press = Mouse_Press
        text = pygame.font.Font('COOPBL.ttf',40)
        self.msg = text.render(gate_type, True, (0,0,0), color)
        self.msg_rect = self.msg.get_rect()
        self.msg_rect.center = x, y   

    def draw(self):
        self.rect = pygame.Rect(self.x-15,self.y+15,30,30)
        self.rect.center = (self.x, self.y)
        self.msg_rect.center = (self.x, self.y)
        self.screen.blit(self.msg, self.msg_rect)

    def mouse_is_over(self):
        Mouse = pygame.mouse.get_pos()
        if self.msg_rect.collidepoint(Mouse):
            return True
        else:
            return False

    #change self.x and self.y but doesn't change the value in .draw()
    def update(self):
        if self.mouse_is_over() and self.Mouse_Press == True:
            pos = pygame.mouse.get_pos()
            self.x = pos[0]
            self.y = pos[1]
            self.draw()
            print(self.x, self.y)
                






