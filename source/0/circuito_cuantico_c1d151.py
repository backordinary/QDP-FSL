# https://github.com/Oscar-CR/Simulador-cuantico-python/blob/f096b3ffa585f0132fa007a9d6515c3dd764fd86/circuito-cuantico.py
""" 
    Representación de circuito cuantico de 3 qubits con compuertas Hadamard y CNOT
    Dependecias:
    - numpy  (pip install numpy) 
    - qiskit  (pip install qiskit) 
    

    Consideraciones
    0.7071067811865475 = 1/√2

 """
from qiskit import QuantumCircuit, execute, Aer, IBMQ, BasicAer, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector, Operator, partial_trace

import tkinter as tk
from tkinter import ttk


""" -------------------- Simulador de circuito --------------------"""

""" Creacion de circuito de 3 qubits """

def resultado():
    total_qbits = int(text_qbits.get()) 
   

    if isinstance(total_qbits, int):
        # Asignacion de qubits
        qubits=total_qbits
        # Convercion del circuito cuantico
        circuito = QuantumCircuit(qubits)
        # Agregado de compuerta de Hadamard  en qubit 0
        circuito.h(0)

        for i in range(1, total_qbits):
            print(i)

            # Agregado de compuerta CNOT  en qubit 0 t target i
            circuito.cx(0, i)

        print("Simulador de circuito cuantico de 3 qubits")
        print(circuito)
        label_circuit.config(text=circuito)
    else: 
        label_circuit.config("Solo se aceptan numeros enteros")
   
    total_qbits=int(text_qbits.get())
    
  
    """ -------------------- Cálculo del vector de estado --------------------"""

    # Asignamos el estado inicial del vector
    estadoQ = Statevector.from_int(0, 2**qubits)

    # Convertimos el estado actual (bits) a estado cuantico (qubits)
    estadoQ = estadoQ.evolve(circuito)
    estadoLista= []

    for i in estadoQ:
        estadoLista.append(i)

    print("Vector de estado: ", estadoLista)



    label_vector.config(text=estadoLista)

    """ -------------------- Simulador de probabilidad --------------------"""

    # Para obtener la probabilidad de que sea 000 o 111, se debe generar un  circuito clasico con valores cuanticos y clasicos 
    q = QuantumRegister(qubits)
    c= ClassicalRegister(qubits)

    # Posteriomente se asignan al generadode de circuito
    circuitoClasico = QuantumCircuit(q, c)

    
    # Puerta logica de Hadamard asignada en el primer qubit 
    circuitoClasico.h(q[0])

    # Puerta CNOT asignada en el primer y segundo qubit
    for i in range(1, total_qbits):
        print(i)

        # Agregado de compuerta CNOT  en qubit 0 t target i
        circuitoClasico.cx(q[0], q[i])
        # Medición de qubits

    circuitoClasico.measure(q, c)

    # Se hace la simulación mediante el algoritmo de qasm_simulator
    simulador = BasicAer.get_backend('qasm_simulator')
    job = execute(circuitoClasico, simulador)
    resultado = job.result()

    # Se cuentan los resultados
    total = resultado.get_counts(circuitoClasico)

    print("Simulador de probabilidad:") 

    probabilidad = []
    for key, value in total.items():
        probabilidad.append(value)
        print("La probabilidad que se genere ", key, " como cadena de bits es de ", value)

    text_total1.delete(0,"")
    text_total1.insert(0,probabilidad[0]) 

    text_total2.delete(0,"")
    text_total2.insert(0,probabilidad[1]) 
    """ -------------------- Representacion Unitaria  --------------------"""

    # Para crear una representacion, se debe generar mediante la funcion Operator, agregando el circuito
    Unitaria = Operator(circuito)

    # La función se encarga de todo, creando un array, para mostrar los datos es necesario acceder a la propiedad data.
    Unitaria.data

    print("Representación unitaria del circuito")

    for x in Unitaria.data:
        print("-----------------------------------------------------------------")
        print(x)
        print("-----------------------------------------------------------------")

    label_unitaria.config(text=Unitaria.data)

   

ventana = tk.Tk()
ventana.title("Simulador de circuito cuantico")
ventana.config(width=500, height=700)


label_title = ttk.Label(text="Simulador de circuito cuántico",)
label_title.place(x=20, y=20)
label_title.config(anchor="center",font=("Roboto", 22))


label_qbits = ttk.Label(text="Cantidad de Qbits (1-7): ",)
label_qbits.place(x=20, y=80,)


label_compuerta = ttk.Label(text="Compuertas ",)
label_compuerta.place(x=360, y=120)

label_hadamard = ttk.Label(text="[H] = Hadamard ",)
label_hadamard.place(x=360, y=140)

label_cnot = ttk.Label(text="[X] = CNOT ",)
label_cnot.place(x=360, y=160)


text_qbits = ttk.Entry()
text_qbits.place(x=160, y=80, width=60)



label_circuit = ttk.Label(text="")
label_circuit.place(x=20, y=110)
label_circuit.config(font=("Courier New", 10),background='#fff')

label_probabilidad = ttk.Label(text="Probabilidad",)
label_probabilidad.place(x=20, y=360)
label_probabilidad.config(font=("Roboto", 12))

label_total1 = ttk.Label(text="La probabilidad que se genere 000 como cadena de bits es de:",)
label_total1.place(x=20, y=390)
text_total1 = ttk.Entry()
text_total1.place(x=350, y=390, width=60)


label_total2 = ttk.Label(text="La probabilidad que se genere 011 como cadena de bits es de:",)
label_total2.place(x=20, y=410)
text_total2 = ttk.Entry()
text_total2.place(x=350, y=410, width=60)


label_vector_title = ttk.Label(text="Vector de estado",)
label_vector_title.place(x=20, y=440)
label_vector_title.config(font=("Roboto", 12))

label_vector = ttk.Label(text="",)
label_vector.place(x=20, y=460)
label_vector.config(background='#fff',font=("Courier New", 8))

label_unitaria_title = ttk.Label(text="Representación Unitaria",)
label_unitaria_title.place(x=20, y=500)
label_unitaria_title.config(font=("Roboto", 12))

label_unitaria = ttk.Label(text="",)
label_unitaria.place(x=20, y=520)
label_unitaria.config(background='#fff',font=("Courier New", 8))



botton_resultado = ttk.Button(text="CALCULAR", command=resultado)
botton_resultado.place(x=240, y=75)


ventana.mainloop()


   