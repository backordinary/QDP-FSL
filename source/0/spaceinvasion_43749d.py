# https://github.com/jeromepatel/Space-Invasion-Research/blob/bf76c55d6598d6e521207ac9187f8e29b9a919ae/spaceInvasion.py
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 19:34:41 2020

@author: jyotm
"""
import math
import qiskit
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit import IBMQ
import numpy as np
import operator
import time
import ast
from configparser import RawConfigParser


device = 'sim'   #running job on simulator rather than on real computer
 
def run(program, type, shots = 100):
  
    # Execute the program in the simulator.
    print("Running on the simulator.")
    start = time.time()
    job = qiskit.execute(program, qiskit.Aer.get_backend('qasm_simulator'), shots=shots)
    result = job.result().get_counts()
    stop = time.time()
    print("Request completed in " + str(round((stop - start) / 60, 2)) + "m " + str(round((stop - start) % 60, 2)) + "s")
    return result


def teleportCapsule(program,circuitDict,source,target):
    
    auxQbits = circuitDict['auxQbits']
    capsulesOnPlanets = circuitDict['capsulesQbits']
    capsulesClassicPlanets = circuitDict['capsulesClassical']
    auxCbit = circuitDict['auxCbit']
    
    create_bell_pair(program, auxQbits[0], capsulesOnPlanets[target])
    program.barrier()
    
    
    program.cx(capsulesOnPlanets[source], auxQbits[0])
    program.h(capsulesOnPlanets[source])
    
    program.measure(auxQbits[0],auxCbit[1])
    program.measure(capsulesOnPlanets[source],auxCbit[0])
    
    program.barrier()
    
    #print(auxCbits_arr, capsulesClassicPlanets[0])
    program.z(capsulesOnPlanets[target]).c_if(auxCbit[0], 1)
    program.x(capsulesOnPlanets[target]).c_if(auxCbit[1], 1)
    
    program.barrier()
    
gameRound = 1

def ChangePlanetForCapsule(program,circuitDict):
    global gameRound
    #roundwise teleport the capsules to planets
    if gameRound == 1:
        #teleport to planet 2 from planet 1
        teleportCapsule(program,circuitDict,0,1)
        
    elif gameRound == 2:
        #teleport to planet 1 from 2 in round 2
        teleportCapsule(program,circuitDict,1,2)
        
    elif gameRound == 3:
        #teleport to planet 2 from planet 1 in round 3
        teleportCapsule(program,circuitDict,2,3)
    
    

def playGame():
    #initialize 2 players with player A having 3 weapons and player B having 3 qbits an 2 operations
    global device
    global gameRound
    
    
    #intialize and print messages 
    
    
    print("===================================================")
    print("Hello There! Space Invasion is a two Playe game, play with a partner for more fun. \n")
    print('''The game takes place in a distant space. There are two planets. Inside one of them, there exist two capsules with infinite energy. It's your choice to save it or take it''')
    
    print("===================================================")
    print("Choose your role to continue: \n1.Defender \n2.Attacker")
    rolePlayer1 = int(input())
    print(rolePlayer1)
    if rolePlayer1 != 1 and rolePlayer1 != 2:
        print("Please enter valid choice: \n1.Defender \n2.Attacker")
        rolePlayer1 = int(input())
    
    
    
    gameMoves = 0
    #no of qbits or capsules
    
    circuitDict = {}
    
    capsulesOnPlanets = QuantumRegister(4,name  = 'q')
    circuitDict['capsulesQbits'] = capsulesOnPlanets
    
    capsulesClassicPlanets = ClassicalRegister(2, name = 'c')
    circuitDict['capsulesClassical'] = capsulesClassicPlanets
    
    auxQbits = QuantumRegister(1, name = 'aux_q')
    circuitDict['auxQbits'] = auxQbits
    #auxCbits_arr = ClassicalRegister(n, name = 'aux_c')
    
    auxCbit = []
    auxCbit.append(ClassicalRegister(1,name='interm_c1'))
    auxCbit.append(ClassicalRegister(1,name='interm_c2'))
    circuitDict['auxCbit'] = auxCbit

    
    program = QuantumCircuit(auxQbits,capsulesOnPlanets, auxCbit[0],auxCbit[1],capsulesClassicPlanets)
    
    #program.u3(0.5,0.5,0,capsulesOnPlanets[0])
    
    #initialize 1st qbit with h and after teleportation again apply h, 
    #as it is reversible we will get 0 as measurement of last Q, (in little endian we have 1st value 0)
    #program.x(capsulesOnPlanets[0])
    
    timesFliped = 0
    while gameRound < 4:
        movePlayer2 = int(input("Player 2 enter your move:\n0.For same planet\n1.For teleportation\n"))
        if movePlayer2:
             #teleport the capsule   
            ChangePlanetForCapsule(program, circuitDict)
            timesFliped += 1
        #increment round 
        gameRound += 1
    
    
    
    
    #again apply h to teleported qbit
    #program.h(capsulesOnPlanets[0+n])
    
    program.measure(capsulesOnPlanets[timesFliped],capsulesClassicPlanets[0])
    #program.measure(capsulesOnPlanets[1],capsulesClassicPlanets[0])
    print(program.draw())
    
    shots = 1024
    results = run(program,device,shots)
    print(results)
    #all capsules are on planet 1 initially 
    IsCapsulesOnPlanet1 = [True, True ,True]
    
    
    
  
    


def create_bell_pair(qc, a, b):
    """Creates a bell pair that is entangle 2 qbits in qc using qubits a & b"""
    qc.h(a) # Put qubit a into state |+>
    qc.cx(a,b) # CNOT with a as control and b as target
    
    

def calculateDamageToCapsules():
    #check whether attack has damaged capsule or not
    
    pass

def main():
    #set quantum computer and simulator data
    #set the device for Quantum computation 
    playGame()
    print("program finished ")
    pass


if __name__ == "__main__":
    main()