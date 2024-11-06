# https://github.com/J2304789/Quantum-Superposition-Pixel-Generator/blob/69bcbdf8f644c0d187546a32e4c34068a09991af/Python_Quantum_Superposition_Pixel_Generator/Python_Multi_QASM/Multi_QASM_Fractional/Quantum_Pixel_Generator_Multi_QASM_Fractional_Complete.py
#Import Qiskit and Qiskit.Visualization
import qiskit
from qiskit import QuantumCircuit, assemble, Aer,execute
from qiskit.visualization import plot_bloch_multivector
from math import sqrt, pi
#Create List for Superposition Pixel Generator
from itertools import islice
#Graph Superposition Pixel
import turtle

def Quantum_Pixel_Multi_QASM_Fractional_Complete(Zero,One):
    #Set Current Qiskit Backend to QASM Simulator 
    #Switch if using IBM Quantum Computers
    sim=Aer.get_backend('qasm_simulator')

    #Intializes Quantum Circuit with 2 Qubits and 1 Classical Bit
    qc=QuantumCircuit(2,1)

    #Amount of times simulation is run
    sim_run=4900

    #Sets 1st Qubit into superposition(|+> basis) using controlled x gate and phase shift s gate
    qc.rx(pi/2,1)#Set to |-i>
    qc.s(1)#Set to |+>

    #Collapses superposition of 1st Qubit and assigns value to corrosponding Classical bit
    qc.measure(1,0)

    #sets 2nd Qubit into superposition(|+> or |-> basis) based on if Qubits 3-6 were measured as |0> or |1>
    qc.ry(pi/2,0)#Set to |+>
    qc.cz(1,0)#Set to |-> if control qubit is |1>,else stays at |+>
    qc.ry(-pi,0)#Set to |+> if qubit was at |->,else shifts to |->

    #Creates barrier between gates and measurements for qc.draw() and optimization level
    qc.barrier()

    #Collapses superposition of 2nd Qubit and assigns value to corrosponding Classical bit
    qc.measure(0,0)

    #Draws Quantum Circuit
    qc.draw(output="latex")

    #memory=True to access indivual simulation qubit measurement values
    job=execute(qc,sim,shots=sim_run,memory=True)
    result=job.result()
    counts=result.get_counts()
    memory=result.get_memory()

    #print(memory)
    #print(len(memory))

    #creates lists for iterations
    list_length=[]
    for i in range (0,70):
        list_length.append(70)

    #print(list_length)
    #print(len(list_length))

    #Create List for Superposition Pixel Generator
    Input = iter(memory)
    Quantum_Pixels = [list(islice(Input, x))
            for x in list_length]

    #print(Quantum_Pixels)

    #Start Draw and set Draw to immediate print
    Draw = turtle.Turtle()
    wn=turtle.Screen()
    wn.bgcolor(Zero)
    wn.tracer(0)
    square_int = 30

    #Set Draw to top left corner of specified Print_Vertical and Print_Horizontal
    Draw.penup()
    Draw.forward(-960)
    Draw.setheading(90)
    Draw.forward(475)
    Draw.setheading(0)

    for x in range (0,len(Quantum_Pixels)):
        for i in range (0,len(Quantum_Pixels[x])):
            if Quantum_Pixels[x][i]=="1":

                Draw.color(One)
                Draw.begin_fill()

                Draw.forward(square_int)

                Draw.left(90)
                Draw.forward(square_int)

                Draw.left(90)
                Draw.forward(square_int)

                Draw.left(90)
                Draw.forward(square_int)

                Draw.end_fill()
                Draw.setheading(0)

            Draw.penup()
            Draw.forward(square_int)
            Draw.pendown()
                
        Draw.setheading(270) 
        Draw.penup()
        Draw.forward(square_int)
        Draw.setheading(180) 
        Draw.forward(square_int*len(Quantum_Pixels[x]))
        Draw.setheading(0)
        Draw.pendown()
        
    Draw.getscreen().update()	
    turtle.done()