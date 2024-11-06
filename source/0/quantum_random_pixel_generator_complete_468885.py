# https://github.com/J2304789/Quantum-Random-Pixel-Generator/blob/7f4ad190d29ccd4865fe5b891a488a5f1abd427c/Python_Quantum_Random_Pixel_Generator/Quantum_Superposition/Quantum_Random_Pixel_Generator_Complete.py
#Import Qiskit and Qiskit.Visualization
import qiskit
from qiskit import QuantumCircuit, assemble, Aer,execute
from qiskit.visualization import plot_bloch_multivector
from math import sqrt, pi
#Create List for Superposition Pixel Generator
from itertools import islice
#Graph Superposition Pixel
import turtle

def Quantum_Random_Pixel_Generator_Complete(Zero,One,Two,Three,Four,Five,Six,Seven):
    #Set Current Qiskit Backend to QASM Simulator 
    #Switch if using IBM Quantum Computers
    sim=Aer.get_backend('qasm_simulator')

    #Intializes Quantum Circuit with 3 Qubits and 3 Classical Bits
    qc=QuantumCircuit(3,3)

    #Amount of times simulation is run
    sim_run=1
    qc.h(0)
    qc.h(1)
    qc.h(2)

    #Creates barrier between gates and measurements for qc.draw() and optimization level
    qc.barrier()
    qc.measure(0,0)
    qc.measure(1,1)
    qc.measure(2,2)

    #Draws Quantum Circuit
    qc.draw(output="latex")

    #Function to convert Qubits to Base 10 and returns randomly generated number
    def Generate():
        #memory=True to access indivual simulation qubit measurement values
        job=execute(qc,sim,shots=sim_run,memory=True)
        result=job.result()
        counts=result.get_counts()
        memory=result.get_memory()
        
        #Converts Qubits to Base 10
        int_value=int(memory[0],2)
        
        #Check int_value throughout iterations
        #print(int_value)
        
        return str(int_value)

    #Creates lists for iterations
    list_length=[]
    Generate_memory=[]
    for i in range (0,70):
        list_length.append(70)
        #Creates list of Randomly Generated Numbers
        for j in range(0,70):
            Generate_memory.append(Generate())

    #print(list_length)
    #print(len(list_length))
    #print(Generate_memory)

    #Create List for Superposition Pixel Generator
    Input = iter(Generate_memory)
    Quantum_Pixels = [list(islice(Input, x))
            for x in list_length]

    #print(Quantum_Pixels)

    #Start Draw and set Draw to immediate print
    Draw = turtle.Turtle()
    wn=turtle.Screen()
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
            if Quantum_Pixels[x][i]=="0":

                Draw.color(Zero)
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

            elif Quantum_Pixels[x][i]=="1":
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

            elif Quantum_Pixels[x][i]=="2":
                Draw.color(Two)
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

            elif Quantum_Pixels[x][i]=="3":
                Draw.color(Three)
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

            elif Quantum_Pixels[x][i]=="4":
                Draw.color(Four)
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

            elif Quantum_Pixels[x][i]=="5":
                Draw.color(Five)
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

            elif Quantum_Pixels[x][i]=="6":
                Draw.color(Six)
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
                
            else:
                Draw.color(Seven)
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