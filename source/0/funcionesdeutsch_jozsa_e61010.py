# https://github.com/Nicolascastro25/DEUTSCH-Y-DEUTSCH-JOZSA/blob/7c9d99af1368ca1632bff4d41d75cd36c25ba27b/FuncionesDeutsch-jozsa.py
import numpy as np
# Import Qiskit
from qiskit import QuantumCircuit, transpile
from qiskit import Aer
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as mpl

def traducirbits(n):
    aux = 0
    r = 0
    while aux < len(n):
        r += int(n[-1-aux])*(2**aux)
        aux += 1
    return r
    
def imprimir(m):
    for i in m:
        print(" ".join(list(map(str,i))))
        
#El Aerproveedor contiene una variedad de backends de simuladores de alto rendimiento para una variedad de métodos de simulación.
#Se crea un nuevo backend de simulador
slt = Aer.get_backend('qasm_simulator')

print("Funcion 1")
a1 = [[0 for k in range(2**(5))] for l in range(2**(5))]
aux = 0
print("Resultado de: ",str(0)+str(0)+str(0)+str(0)+str(0))
#Crear un nuevo circuito.
c = QuantumCircuit(5, 5)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(0)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(1)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(2)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(3)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(4)
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
# Medimos bits cuánticos en bits clásicos (tuplas).
c.measure([0,1,2,3,4],[4,3,2,1,0])
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
#Transpila uno o más circuitos, de acuerdo con algunos objetivos dados
comC = transpile(c, slt)
#Devuelve el trabajo de la simulación
trabajo = slt.run(comC, shots=1000)
#Obtener resultado del trabajo.
r = trabajo.result()
#Obtener los datos del histograma de un experimento.
recorrido = r.get_counts(c)
#Imprime los datos obtenidos del histograma
print("\nEl recuento total para 00 y 11 es:",recorrido)
#imprimimos el circuito
print(c)
#Trazar un histograma de datos de recuentos de entrada.
plot_histogram(recorrido)
#Mostrar todas las figuras abiertas.
mpl.show()
print("Resultado de: ",str(0)+str(0)+str(0)+str(0)+str(1))
#Crear un nuevo circuito.
c = QuantumCircuit(5, 5)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(0)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(1)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(2)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(3)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(4)
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
# Medimos bits cuánticos en bits clásicos (tuplas).
c.measure([0,1,2,3,4],[4,3,2,1,0])
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
#Transpila uno o más circuitos, de acuerdo con algunos objetivos dados
comC = transpile(c, slt)
#Devuelve el trabajo de la simulación
trabajo = slt.run(comC, shots=1000)
#Obtener resultado del trabajo.
r = trabajo.result()
#Obtener los datos del histograma de un experimento.
recorrido = r.get_counts(c)
#Imprime los datos obtenidos del histograma
print("\nEl recuento total para 00 y 11 es:",recorrido)
#imprimimos el circuito
print(c)
#Trazar un histograma de datos de recuentos de entrada.
plot_histogram(recorrido)
#Mostrar todas las figuras abiertas.
mpl.show()
print("Resultado de: ",str(0)+str(0)+str(0)+str(1)+str(0))
#Crear un nuevo circuito.
c = QuantumCircuit(5, 5)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(0)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(1)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(2)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(3)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(4)
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
# Medimos bits cuánticos en bits clásicos (tuplas).
c.measure([0,1,2,3,4],[4,3,2,1,0])
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
#Transpila uno o más circuitos, de acuerdo con algunos objetivos dados
comC = transpile(c, slt)
#Devuelve el trabajo de la simulación
trabajo = slt.run(comC, shots=1000)
#Obtener resultado del trabajo.
r = trabajo.result()
#Obtener los datos del histograma de un experimento.
recorrido = r.get_counts(c)
#Imprime los datos obtenidos del histograma
print("\nEl recuento total para 00 y 11 es:",recorrido)
#imprimimos el circuito
print(c)
#Trazar un histograma de datos de recuentos de entrada.
plot_histogram(recorrido)
#Mostrar todas las figuras abiertas.
mpl.show()
print("Resultado de: ",str(0)+str(0)+str(0)+str(1)+str(1))
#Crear un nuevo circuito.
c = QuantumCircuit(5, 5)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(0)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(1)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(2)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(3)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(4)
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
# Medimos bits cuánticos en bits clásicos (tuplas).
c.measure([0,1,2,3,4],[4,3,2,1,0])
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
#Transpila uno o más circuitos, de acuerdo con algunos objetivos dados
comC = transpile(c, slt)
#Devuelve el trabajo de la simulación
trabajo = slt.run(comC, shots=1000)
#Obtener resultado del trabajo.
r = trabajo.result()
#Obtener los datos del histograma de un experimento.
recorrido = r.get_counts(c)
#Imprime los datos obtenidos del histograma
print("\nEl recuento total para 00 y 11 es:",recorrido)
#imprimimos el circuito
print(c)
#Trazar un histograma de datos de recuentos de entrada.
plot_histogram(recorrido)
#Mostrar todas las figuras abiertas.
mpl.show()
print("Resultado de: ",str(0)+str(0)+str(1)+str(0)+str(0))
#Crear un nuevo circuito.
c = QuantumCircuit(5, 5)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(0)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(1)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(2)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(3)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(4)
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
# Medimos bits cuánticos en bits clásicos (tuplas).
c.measure([0,1,2,3,4],[4,3,2,1,0])
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
#Transpila uno o más circuitos, de acuerdo con algunos objetivos dados
comC = transpile(c, slt)
#Devuelve el trabajo de la simulación
trabajo = slt.run(comC, shots=1000)
#Obtener resultado del trabajo.
r = trabajo.result()
#Obtener los datos del histograma de un experimento.
recorrido = r.get_counts(c)
#Imprime los datos obtenidos del histograma
print("\nEl recuento total para 00 y 11 es:",recorrido)
#imprimimos el circuito
print(c)
#Trazar un histograma de datos de recuentos de entrada.
plot_histogram(recorrido)
#Mostrar todas las figuras abiertas.
mpl.show()
print("Resultado de: ",str(0)+str(0)+str(1)+str(0)+str(1))
#Crear un nuevo circuito.
c = QuantumCircuit(5, 5)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(0)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(1)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(2)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(3)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(4)
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
# Medimos bits cuánticos en bits clásicos (tuplas).
c.measure([0,1,2,3,4],[4,3,2,1,0])
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
#Transpila uno o más circuitos, de acuerdo con algunos objetivos dados
comC = transpile(c, slt)
#Devuelve el trabajo de la simulación
trabajo = slt.run(comC, shots=1000)
#Obtener resultado del trabajo.
r = trabajo.result()
#Obtener los datos del histograma de un experimento.
recorrido = r.get_counts(c)
#Imprime los datos obtenidos del histograma
print("\nEl recuento total para 00 y 11 es:",recorrido)
#imprimimos el circuito
print(c)
#Trazar un histograma de datos de recuentos de entrada.
plot_histogram(recorrido)
#Mostrar todas las figuras abiertas.
mpl.show()
print("Resultado de: ",str(0)+str(0)+str(1)+str(1)+str(0))
#Crear un nuevo circuito.
c = QuantumCircuit(5, 5)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(0)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(1)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(2)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(3)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(4)
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
# Medimos bits cuánticos en bits clásicos (tuplas).
c.measure([0,1,2,3,4],[4,3,2,1,0])
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
#Transpila uno o más circuitos, de acuerdo con algunos objetivos dados
comC = transpile(c, slt)
#Devuelve el trabajo de la simulación
trabajo = slt.run(comC, shots=1000)
#Obtener resultado del trabajo.
r = trabajo.result()
#Obtener los datos del histograma de un experimento.
recorrido = r.get_counts(c)
#Imprime los datos obtenidos del histograma
print("\nEl recuento total para 00 y 11 es:",recorrido)
#imprimimos el circuito
print(c)
#Trazar un histograma de datos de recuentos de entrada.
plot_histogram(recorrido)
#Mostrar todas las figuras abiertas.
mpl.show()
print("Resultado de: ",str(0)+str(0)+str(1)+str(1)+str(1))
#Crear un nuevo circuito.
c = QuantumCircuit(5, 5)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(0)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(1)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(2)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(3)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(4)
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
# Medimos bits cuánticos en bits clásicos (tuplas).
c.measure([0,1,2,3,4],[4,3,2,1,0])
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
#Transpila uno o más circuitos, de acuerdo con algunos objetivos dados
comC = transpile(c, slt)
#Devuelve el trabajo de la simulación
trabajo = slt.run(comC, shots=1000)
#Obtener resultado del trabajo.
r = trabajo.result()
#Obtener los datos del histograma de un experimento.
recorrido = r.get_counts(c)
#Imprime los datos obtenidos del histograma
print("\nEl recuento total para 00 y 11 es:",recorrido)
#imprimimos el circuito
print(c)
#Trazar un histograma de datos de recuentos de entrada.
plot_histogram(recorrido)
#Mostrar todas las figuras abiertas.
mpl.show()
print("Resultado de: ",str(0)+str(1)+str(0)+str(0)+str(0))
#Crear un nuevo circuito.
c = QuantumCircuit(5, 5)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(0)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(1)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(2)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(3)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(4)
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
# Medimos bits cuánticos en bits clásicos (tuplas).
c.measure([0,1,2,3,4],[4,3,2,1,0])
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
#Transpila uno o más circuitos, de acuerdo con algunos objetivos dados
comC = transpile(c, slt)
#Devuelve el trabajo de la simulación
trabajo = slt.run(comC, shots=1000)
#Obtener resultado del trabajo.
r = trabajo.result()
#Obtener los datos del histograma de un experimento.
recorrido = r.get_counts(c)
#Imprime los datos obtenidos del histograma
print("\nEl recuento total para 00 y 11 es:",recorrido)
#imprimimos el circuito
print(c)
#Trazar un histograma de datos de recuentos de entrada.
plot_histogram(recorrido)
#Mostrar todas las figuras abiertas.
mpl.show()
print("Resultado de: ",str(0)+str(1)+str(0)+str(0)+str(1))
#Crear un nuevo circuito.
c = QuantumCircuit(5, 5)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(0)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(1)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(2)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(3)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(4)
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
# Medimos bits cuánticos en bits clásicos (tuplas).
c.measure([0,1,2,3,4],[4,3,2,1,0])
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
#Transpila uno o más circuitos, de acuerdo con algunos objetivos dados
comC = transpile(c, slt)
#Devuelve el trabajo de la simulación
trabajo = slt.run(comC, shots=1000)
#Obtener resultado del trabajo.
r = trabajo.result()
#Obtener los datos del histograma de un experimento.
recorrido = r.get_counts(c)
#Imprime los datos obtenidos del histograma
print("\nEl recuento total para 00 y 11 es:",recorrido)
#imprimimos el circuito
print(c)
#Trazar un histograma de datos de recuentos de entrada.
plot_histogram(recorrido)
#Mostrar todas las figuras abiertas.
mpl.show()
print("Resultado de: ",str(0)+str(1)+str(0)+str(1)+str(0))
#Crear un nuevo circuito.
c = QuantumCircuit(5, 5)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(0)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(1)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(2)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(3)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(4)
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
# Medimos bits cuánticos en bits clásicos (tuplas).
c.measure([0,1,2,3,4],[4,3,2,1,0])
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
#Transpila uno o más circuitos, de acuerdo con algunos objetivos dados
comC = transpile(c, slt)
#Devuelve el trabajo de la simulación
trabajo = slt.run(comC, shots=1000)
#Obtener resultado del trabajo.
r = trabajo.result()
#Obtener los datos del histograma de un experimento.
recorrido = r.get_counts(c)
#Imprime los datos obtenidos del histograma
print("\nEl recuento total para 00 y 11 es:",recorrido)
#imprimimos el circuito
print(c)
#Trazar un histograma de datos de recuentos de entrada.
plot_histogram(recorrido)
#Mostrar todas las figuras abiertas.
mpl.show()
print("Resultado de: ",str(0)+str(1)+str(0)+str(1)+str(1))
#Crear un nuevo circuito.
c = QuantumCircuit(5, 5)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(0)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(1)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(2)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(3)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(4)
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
# Medimos bits cuánticos en bits clásicos (tuplas).
c.measure([0,1,2,3,4],[4,3,2,1,0])
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
#Transpila uno o más circuitos, de acuerdo con algunos objetivos dados
comC = transpile(c, slt)
#Devuelve el trabajo de la simulación
trabajo = slt.run(comC, shots=1000)
#Obtener resultado del trabajo.
r = trabajo.result()
#Obtener los datos del histograma de un experimento.
recorrido = r.get_counts(c)
#Imprime los datos obtenidos del histograma
print("\nEl recuento total para 00 y 11 es:",recorrido)
#imprimimos el circuito
print(c)
#Trazar un histograma de datos de recuentos de entrada.
plot_histogram(recorrido)
#Mostrar todas las figuras abiertas.
mpl.show()
print("Resultado de: ",str(0)+str(1)+str(1)+str(0)+str(0))
#Crear un nuevo circuito.
c = QuantumCircuit(5, 5)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(0)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(1)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(2)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(3)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(4)
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
# Medimos bits cuánticos en bits clásicos (tuplas).
c.measure([0,1,2,3,4],[4,3,2,1,0])
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
#Transpila uno o más circuitos, de acuerdo con algunos objetivos dados
comC = transpile(c, slt)
#Devuelve el trabajo de la simulación
trabajo = slt.run(comC, shots=1000)
#Obtener resultado del trabajo.
r = trabajo.result()
#Obtener los datos del histograma de un experimento.
recorrido = r.get_counts(c)
#Imprime los datos obtenidos del histograma
print("\nEl recuento total para 00 y 11 es:",recorrido)
#imprimimos el circuito
print(c)
#Trazar un histograma de datos de recuentos de entrada.
plot_histogram(recorrido)
#Mostrar todas las figuras abiertas.
mpl.show()
print("Resultado de: ",str(0)+str(1)+str(1)+str(0)+str(1))
#Crear un nuevo circuito.
c = QuantumCircuit(5, 5)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(0)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(1)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(2)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(3)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(4)
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
# Medimos bits cuánticos en bits clásicos (tuplas).
c.measure([0,1,2,3,4],[4,3,2,1,0])
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
#Transpila uno o más circuitos, de acuerdo con algunos objetivos dados
comC = transpile(c, slt)
#Devuelve el trabajo de la simulación
trabajo = slt.run(comC, shots=1000)
#Obtener resultado del trabajo.
r = trabajo.result()
#Obtener los datos del histograma de un experimento.
recorrido = r.get_counts(c)
#Imprime los datos obtenidos del histograma
print("\nEl recuento total para 00 y 11 es:",recorrido)
#imprimimos el circuito
print(c)
#Trazar un histograma de datos de recuentos de entrada.
plot_histogram(recorrido)
#Mostrar todas las figuras abiertas.
mpl.show()
print("Resultado de: ",str(0)+str(1)+str(1)+str(1)+str(0))
#Crear un nuevo circuito.
c = QuantumCircuit(5, 5)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(0)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(1)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(2)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(3)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(4)
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
# Medimos bits cuánticos en bits clásicos (tuplas).
c.measure([0,1,2,3,4],[4,3,2,1,0])
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
#Transpila uno o más circuitos, de acuerdo con algunos objetivos dados
comC = transpile(c, slt)
#Devuelve el trabajo de la simulación
trabajo = slt.run(comC, shots=1000)
#Obtener resultado del trabajo.
r = trabajo.result()
#Obtener los datos del histograma de un experimento.
recorrido = r.get_counts(c)
#Imprime los datos obtenidos del histograma
print("\nEl recuento total para 00 y 11 es:",recorrido)
#imprimimos el circuito
print(c)
#Trazar un histograma de datos de recuentos de entrada.
plot_histogram(recorrido)
#Mostrar todas las figuras abiertas.
mpl.show()
print("Resultado de: ",str(0)+str(1)+str(1)+str(1)+str(1))
#Crear un nuevo circuito.
c = QuantumCircuit(5, 5)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(0)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(1)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(2)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(3)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(4)
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
# Medimos bits cuánticos en bits clásicos (tuplas).
c.measure([0,1,2,3,4],[4,3,2,1,0])
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
#Transpila uno o más circuitos, de acuerdo con algunos objetivos dados
comC = transpile(c, slt)
#Devuelve el trabajo de la simulación
trabajo = slt.run(comC, shots=1000)
#Obtener resultado del trabajo.
r = trabajo.result()
#Obtener los datos del histograma de un experimento.
recorrido = r.get_counts(c)
#Imprime los datos obtenidos del histograma
print("\nEl recuento total para 00 y 11 es:",recorrido)
#imprimimos el circuito
print(c)
#Trazar un histograma de datos de recuentos de entrada.
plot_histogram(recorrido)
#Mostrar todas las figuras abiertas.
mpl.show()
print("Resultado de: ",str(1)+str(0)+str(0)+str(0)+str(0))
#Crear un nuevo circuito.
c = QuantumCircuit(5, 5)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(0)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(1)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(2)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(3)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(4)
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
# Medimos bits cuánticos en bits clásicos (tuplas).
c.measure([0,1,2,3,4],[4,3,2,1,0])
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
#Transpila uno o más circuitos, de acuerdo con algunos objetivos dados
comC = transpile(c, slt)
#Devuelve el trabajo de la simulación
trabajo = slt.run(comC, shots=1000)
#Obtener resultado del trabajo.
r = trabajo.result()
#Obtener los datos del histograma de un experimento.
recorrido = r.get_counts(c)
#Imprime los datos obtenidos del histograma
print("\nEl recuento total para 00 y 11 es:",recorrido)
#imprimimos el circuito
print(c)
#Trazar un histograma de datos de recuentos de entrada.
plot_histogram(recorrido)
#Mostrar todas las figuras abiertas.
mpl.show()
print("Resultado de: ",str(1)+str(0)+str(0)+str(0)+str(1))
#Crear un nuevo circuito.
c = QuantumCircuit(5, 5)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(0)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(1)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(2)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(3)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(4)
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
# Medimos bits cuánticos en bits clásicos (tuplas).
c.measure([0,1,2,3,4],[4,3,2,1,0])
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
#Transpila uno o más circuitos, de acuerdo con algunos objetivos dados
comC = transpile(c, slt)
#Devuelve el trabajo de la simulación
trabajo = slt.run(comC, shots=1000)
#Obtener resultado del trabajo.
r = trabajo.result()
#Obtener los datos del histograma de un experimento.
recorrido = r.get_counts(c)
#Imprime los datos obtenidos del histograma
print("\nEl recuento total para 00 y 11 es:",recorrido)
#imprimimos el circuito
print(c)
#Trazar un histograma de datos de recuentos de entrada.
plot_histogram(recorrido)
#Mostrar todas las figuras abiertas.
mpl.show()
print("Resultado de: ",str(1)+str(0)+str(0)+str(1)+str(0))
#Crear un nuevo circuito.
c = QuantumCircuit(5, 5)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(0)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(1)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(2)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(3)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(4)
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
# Medimos bits cuánticos en bits clásicos (tuplas).
c.measure([0,1,2,3,4],[4,3,2,1,0])
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
#Transpila uno o más circuitos, de acuerdo con algunos objetivos dados
comC = transpile(c, slt)
#Devuelve el trabajo de la simulación
trabajo = slt.run(comC, shots=1000)
#Obtener resultado del trabajo.
r = trabajo.result()
#Obtener los datos del histograma de un experimento.
recorrido = r.get_counts(c)
#Imprime los datos obtenidos del histograma
print("\nEl recuento total para 00 y 11 es:",recorrido)
#imprimimos el circuito
print(c)
#Trazar un histograma de datos de recuentos de entrada.
plot_histogram(recorrido)
#Mostrar todas las figuras abiertas.
mpl.show()
print("Resultado de: ",str(1)+str(0)+str(0)+str(1)+str(1))
#Crear un nuevo circuito.
c = QuantumCircuit(5, 5)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(0)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(1)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(2)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(3)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(4)
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
# Medimos bits cuánticos en bits clásicos (tuplas).
c.measure([0,1,2,3,4],[4,3,2,1,0])
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
#Transpila uno o más circuitos, de acuerdo con algunos objetivos dados
comC = transpile(c, slt)
#Devuelve el trabajo de la simulación
trabajo = slt.run(comC, shots=1000)
#Obtener resultado del trabajo.
r = trabajo.result()
#Obtener los datos del histograma de un experimento.
recorrido = r.get_counts(c)
#Imprime los datos obtenidos del histograma
print("\nEl recuento total para 00 y 11 es:",recorrido)
#imprimimos el circuito
print(c)
#Trazar un histograma de datos de recuentos de entrada.
plot_histogram(recorrido)
#Mostrar todas las figuras abiertas.
mpl.show()
print("Resultado de: ",str(1)+str(0)+str(1)+str(0)+str(0))
#Crear un nuevo circuito.
c = QuantumCircuit(5, 5)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(0)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(1)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(2)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(3)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(4)
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
# Medimos bits cuánticos en bits clásicos (tuplas).
c.measure([0,1,2,3,4],[4,3,2,1,0])
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
#Transpila uno o más circuitos, de acuerdo con algunos objetivos dados
comC = transpile(c, slt)
#Devuelve el trabajo de la simulación
trabajo = slt.run(comC, shots=1000)
#Obtener resultado del trabajo.
r = trabajo.result()
#Obtener los datos del histograma de un experimento.
recorrido = r.get_counts(c)
#Imprime los datos obtenidos del histograma
print("\nEl recuento total para 00 y 11 es:",recorrido)
#imprimimos el circuito
print(c)
#Trazar un histograma de datos de recuentos de entrada.
plot_histogram(recorrido)
#Mostrar todas las figuras abiertas.
mpl.show()
print("Resultado de: ",str(1)+str(0)+str(1)+str(0)+str(1))
#Crear un nuevo circuito.
c = QuantumCircuit(5, 5)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(0)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(1)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(2)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(3)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(4)
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
# Medimos bits cuánticos en bits clásicos (tuplas).
c.measure([0,1,2,3,4],[4,3,2,1,0])
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
#Transpila uno o más circuitos, de acuerdo con algunos objetivos dados
comC = transpile(c, slt)
#Devuelve el trabajo de la simulación
trabajo = slt.run(comC, shots=1000)
#Obtener resultado del trabajo.
r = trabajo.result()
#Obtener los datos del histograma de un experimento.
recorrido = r.get_counts(c)
#Imprime los datos obtenidos del histograma
print("\nEl recuento total para 00 y 11 es:",recorrido)
#imprimimos el circuito
print(c)
#Trazar un histograma de datos de recuentos de entrada.
plot_histogram(recorrido)
#Mostrar todas las figuras abiertas.
mpl.show()
print("Resultado de: ",str(1)+str(0)+str(1)+str(1)+str(0))
#Crear un nuevo circuito.
c = QuantumCircuit(5, 5)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(0)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(1)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(2)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(3)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(4)
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
# Medimos bits cuánticos en bits clásicos (tuplas).
c.measure([0,1,2,3,4],[4,3,2,1,0])
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
#Transpila uno o más circuitos, de acuerdo con algunos objetivos dados
comC = transpile(c, slt)
#Devuelve el trabajo de la simulación
trabajo = slt.run(comC, shots=1000)
#Obtener resultado del trabajo.
r = trabajo.result()
#Obtener los datos del histograma de un experimento.
recorrido = r.get_counts(c)
#Imprime los datos obtenidos del histograma
print("\nEl recuento total para 00 y 11 es:",recorrido)
#imprimimos el circuito
print(c)
#Trazar un histograma de datos de recuentos de entrada.
plot_histogram(recorrido)
#Mostrar todas las figuras abiertas.
mpl.show()
print("Resultado de: ",str(1)+str(0)+str(1)+str(1)+str(1))
#Crear un nuevo circuito.
c = QuantumCircuit(5, 5)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(0)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(1)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(2)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(3)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(4)
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
# Medimos bits cuánticos en bits clásicos (tuplas).
c.measure([0,1,2,3,4],[4,3,2,1,0])
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
#Transpila uno o más circuitos, de acuerdo con algunos objetivos dados
comC = transpile(c, slt)
#Devuelve el trabajo de la simulación
trabajo = slt.run(comC, shots=1000)
#Obtener resultado del trabajo.
r = trabajo.result()
#Obtener los datos del histograma de un experimento.
recorrido = r.get_counts(c)
#Imprime los datos obtenidos del histograma
print("\nEl recuento total para 00 y 11 es:",recorrido)
#imprimimos el circuito
print(c)
#Trazar un histograma de datos de recuentos de entrada.
plot_histogram(recorrido)
#Mostrar todas las figuras abiertas.
mpl.show()
print("Resultado de: ",str(1)+str(1)+str(0)+str(0)+str(0))
#Crear un nuevo circuito.
c = QuantumCircuit(5, 5)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(0)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(1)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(2)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(3)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(4)
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
# Medimos bits cuánticos en bits clásicos (tuplas).
c.measure([0,1,2,3,4],[4,3,2,1,0])
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
#Transpila uno o más circuitos, de acuerdo con algunos objetivos dados
comC = transpile(c, slt)
#Devuelve el trabajo de la simulación
trabajo = slt.run(comC, shots=1000)
#Obtener resultado del trabajo.
r = trabajo.result()
#Obtener los datos del histograma de un experimento.
recorrido = r.get_counts(c)
#Imprime los datos obtenidos del histograma
print("\nEl recuento total para 00 y 11 es:",recorrido)
#imprimimos el circuito
print(c)
#Trazar un histograma de datos de recuentos de entrada.
plot_histogram(recorrido)
#Mostrar todas las figuras abiertas.
mpl.show()
print("Resultado de: ",str(1)+str(1)+str(0)+str(0)+str(1))
#Crear un nuevo circuito.
c = QuantumCircuit(5, 5)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(0)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(1)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(2)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(3)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(4)
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
# Medimos bits cuánticos en bits clásicos (tuplas).
c.measure([0,1,2,3,4],[4,3,2,1,0])
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
#Transpila uno o más circuitos, de acuerdo con algunos objetivos dados
comC = transpile(c, slt)
#Devuelve el trabajo de la simulación
trabajo = slt.run(comC, shots=1000)
#Obtener resultado del trabajo.
r = trabajo.result()
#Obtener los datos del histograma de un experimento.
recorrido = r.get_counts(c)
#Imprime los datos obtenidos del histograma
print("\nEl recuento total para 00 y 11 es:",recorrido)
#imprimimos el circuito
print(c)
#Trazar un histograma de datos de recuentos de entrada.
plot_histogram(recorrido)
#Mostrar todas las figuras abiertas.
mpl.show()
print("Resultado de: ",str(1)+str(1)+str(0)+str(1)+str(0))
#Crear un nuevo circuito.
c = QuantumCircuit(5, 5)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(0)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(1)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(2)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(3)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(4)
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
# Medimos bits cuánticos en bits clásicos (tuplas).
c.measure([0,1,2,3,4],[4,3,2,1,0])
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
#Transpila uno o más circuitos, de acuerdo con algunos objetivos dados
comC = transpile(c, slt)
#Devuelve el trabajo de la simulación
trabajo = slt.run(comC, shots=1000)
#Obtener resultado del trabajo.
r = trabajo.result()
#Obtener los datos del histograma de un experimento.
recorrido = r.get_counts(c)
#Imprime los datos obtenidos del histograma
print("\nEl recuento total para 00 y 11 es:",recorrido)
#imprimimos el circuito
print(c)
#Trazar un histograma de datos de recuentos de entrada.
plot_histogram(recorrido)
#Mostrar todas las figuras abiertas.
mpl.show()
print("Resultado de: ",str(1)+str(1)+str(0)+str(1)+str(1))
#Crear un nuevo circuito.
c = QuantumCircuit(5, 5)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(0)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(1)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(2)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(3)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(4)
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
# Medimos bits cuánticos en bits clásicos (tuplas).
c.measure([0,1,2,3,4],[4,3,2,1,0])
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
#Transpila uno o más circuitos, de acuerdo con algunos objetivos dados
comC = transpile(c, slt)
#Devuelve el trabajo de la simulación
trabajo = slt.run(comC, shots=1000)
#Obtener resultado del trabajo.
r = trabajo.result()
#Obtener los datos del histograma de un experimento.
recorrido = r.get_counts(c)
#Imprime los datos obtenidos del histograma
print("\nEl recuento total para 00 y 11 es:",recorrido)
#imprimimos el circuito
print(c)
#Trazar un histograma de datos de recuentos de entrada.
plot_histogram(recorrido)
#Mostrar todas las figuras abiertas.
mpl.show()
print("Resultado de: ",str(1)+str(1)+str(1)+str(0)+str(0))
#Crear un nuevo circuito.
c = QuantumCircuit(5, 5)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(0)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(1)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(2)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(3)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(4)
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
# Medimos bits cuánticos en bits clásicos (tuplas).
c.measure([0,1,2,3,4],[4,3,2,1,0])
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
#Transpila uno o más circuitos, de acuerdo con algunos objetivos dados
comC = transpile(c, slt)
#Devuelve el trabajo de la simulación
trabajo = slt.run(comC, shots=1000)
#Obtener resultado del trabajo.
r = trabajo.result()
#Obtener los datos del histograma de un experimento.
recorrido = r.get_counts(c)
#Imprime los datos obtenidos del histograma
print("\nEl recuento total para 00 y 11 es:",recorrido)
#imprimimos el circuito
print(c)
#Trazar un histograma de datos de recuentos de entrada.
plot_histogram(recorrido)
#Mostrar todas las figuras abiertas.
mpl.show()
print("Resultado de: ",str(1)+str(1)+str(1)+str(0)+str(1))
#Crear un nuevo circuito.
c = QuantumCircuit(5, 5)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(0)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(1)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(2)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(3)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(4)
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
# Medimos bits cuánticos en bits clásicos (tuplas).
c.measure([0,1,2,3,4],[4,3,2,1,0])
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
#Transpila uno o más circuitos, de acuerdo con algunos objetivos dados
comC = transpile(c, slt)
#Devuelve el trabajo de la simulación
trabajo = slt.run(comC, shots=1000)
#Obtener resultado del trabajo.
r = trabajo.result()
#Obtener los datos del histograma de un experimento.
recorrido = r.get_counts(c)
#Imprime los datos obtenidos del histograma
print("\nEl recuento total para 00 y 11 es:",recorrido)
#imprimimos el circuito
print(c)
#Trazar un histograma de datos de recuentos de entrada.
plot_histogram(recorrido)
#Mostrar todas las figuras abiertas.
mpl.show()
print("Resultado de: ",str(1)+str(1)+str(1)+str(1)+str(0))
#Crear un nuevo circuito.
c = QuantumCircuit(5, 5)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(0)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(1)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(2)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(3)
if 0 == 1:
    #Aplicar la puerta base X.
    c.x(4)
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
# Medimos bits cuánticos en bits clásicos (tuplas).
c.measure([0,1,2,3,4],[4,3,2,1,0])
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
#Transpila uno o más circuitos, de acuerdo con algunos objetivos dados
comC = transpile(c, slt)
#Devuelve el trabajo de la simulación
trabajo = slt.run(comC, shots=1000)
#Obtener resultado del trabajo.
r = trabajo.result()
#Obtener los datos del histograma de un experimento.
recorrido = r.get_counts(c)
#Imprime los datos obtenidos del histograma
print("\nEl recuento total para 00 y 11 es:",recorrido)
#imprimimos el circuito
print(c)
#Trazar un histograma de datos de recuentos de entrada.
plot_histogram(recorrido)
#Mostrar todas las figuras abiertas.
mpl.show()
print("Resultado de: ",str(1)+str(1)+str(1)+str(1)+str(1))
#Crear un nuevo circuito.
c = QuantumCircuit(5, 5)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(0)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(1)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(2)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(3)
if 1 == 1:
    #Aplicar la puerta base X.
    c.x(4)
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
# Medimos bits cuánticos en bits clásicos (tuplas).
c.measure([0,1,2,3,4],[4,3,2,1,0])
#La barrera actúa como una directiva para la compilación de circuitos para separar partes de un circuito
c.barrier()
#Transpila uno o más circuitos, de acuerdo con algunos objetivos dados
comC = transpile(c, slt)
#Devuelve el trabajo de la simulación
trabajo = slt.run(comC, shots=1000)
#Obtener resultado del trabajo.
r = trabajo.result()
#Obtener los datos del histograma de un experimento.
recorrido = r.get_counts(c)
#Imprime los datos obtenidos del histograma
print("\nEl recuento total para 00 y 11 es:",recorrido)
#imprimimos el circuito
print(c)
#Trazar un histograma de datos de recuentos de entrada.
plot_histogram(recorrido)
#Mostrar todas las figuras abiertas.
mpl.show()

#Para cada una de las siguientes lineas no se documentará puesto que la definición de las funciones son las usadas en el apartado anterior.
print("Funcion 2")
print("Resultado de: ",str(0)+str(0)+str(0)+str(0)+str(0))
c = QuantumCircuit(5, 5)
if 0 == 1:
    c.x(0)
if 0 == 1:
    c.x(1)
if 0 == 1:
    c.x(2)
if 0 == 1:
    c.x(3)
if 0 == 1:
    c.x(4)
c.barrier()
c.cnot(0,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(0)+str(0)+str(0)+str(0)+str(1))
c = QuantumCircuit(5, 5)
if 0 == 1:
    c.x(0)
if 0 == 1:
    c.x(1)
if 0 == 1:
    c.x(2)
if 0 == 1:
    c.x(3)
if 1 == 1:
    c.x(4)
c.barrier()
c.cnot(0,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(0)+str(0)+str(0)+str(1)+str(0))
c = QuantumCircuit(5, 5)
if 0 == 1:
    c.x(0)
if 0 == 1:
    c.x(1)
if 0 == 1:
    c.x(2)
if 1 == 1:
    c.x(3)
if 0 == 1:
    c.x(4)
c.barrier()
c.cnot(0,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(0)+str(0)+str(0)+str(1)+str(1))
c = QuantumCircuit(5, 5)
if 0 == 1:
    c.x(0)
if 0 == 1:
    c.x(1)
if 0 == 1:
    c.x(2)
if 1 == 1:
    c.x(3)
if 1 == 1:
    c.x(4)
c.barrier()
c.cnot(0,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(0)+str(0)+str(1)+str(0)+str(0))
c = QuantumCircuit(5, 5)
if 0 == 1:
    c.x(0)
if 0 == 1:
    c.x(1)
if 1 == 1:
    c.x(2)
if 0 == 1:
    c.x(3)
if 0 == 1:
    c.x(4)
c.barrier()
c.cnot(0,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(0)+str(0)+str(1)+str(0)+str(1))
c = QuantumCircuit(5, 5)
if 0 == 1:
    c.x(0)
if 0 == 1:
    c.x(1)
if 1 == 1:
    c.x(2)
if 0 == 1:
    c.x(3)
if 1 == 1:
    c.x(4)
c.barrier()
c.cnot(0,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(0)+str(0)+str(1)+str(1)+str(0))
c = QuantumCircuit(5, 5)
if 0 == 1:
    c.x(0)
if 0 == 1:
    c.x(1)
if 1 == 1:
    c.x(2)
if 1 == 1:
    c.x(3)
if 0 == 1:
    c.x(4)
c.barrier()
c.cnot(0,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(0)+str(0)+str(1)+str(1)+str(1))
c = QuantumCircuit(5, 5)
if 0 == 1:
    c.x(0)
if 0 == 1:
    c.x(1)
if 1 == 1:
    c.x(2)
if 1 == 1:
    c.x(3)
if 1 == 1:
    c.x(4)
c.barrier()
c.cnot(0,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(0)+str(1)+str(0)+str(0)+str(0))
c = QuantumCircuit(5, 5)
if 0 == 1:
    c.x(0)
if 1 == 1:
    c.x(1)
if 0 == 1:
    c.x(2)
if 0 == 1:
    c.x(3)
if 0 == 1:
    c.x(4)
c.barrier()
c.cnot(0,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(0)+str(1)+str(0)+str(0)+str(1))
c = QuantumCircuit(5, 5)
if 0 == 1:
    c.x(0)
if 1 == 1:
    c.x(1)
if 0 == 1:
    c.x(2)
if 0 == 1:
    c.x(3)
if 1 == 1:
    c.x(4)
c.barrier()
c.cnot(0,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(0)+str(1)+str(0)+str(1)+str(0))
c = QuantumCircuit(5, 5)
if 0 == 1:
    c.x(0)
if 1 == 1:
    c.x(1)
if 0 == 1:
    c.x(2)
if 1 == 1:
    c.x(3)
if 0 == 1:
    c.x(4)
c.barrier()
c.cnot(0,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(0)+str(1)+str(0)+str(1)+str(1))
c = QuantumCircuit(5, 5)
if 0 == 1:
    c.x(0)
if 1 == 1:
    c.x(1)
if 0 == 1:
    c.x(2)
if 1 == 1:
    c.x(3)
if 1 == 1:
    c.x(4)
c.barrier()
c.cnot(0,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(0)+str(1)+str(1)+str(0)+str(0))
c = QuantumCircuit(5, 5)
if 0 == 1:
    c.x(0)
if 1 == 1:
    c.x(1)
if 1 == 1:
    c.x(2)
if 0 == 1:
    c.x(3)
if 0 == 1:
    c.x(4)
c.barrier()
c.cnot(0,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(0)+str(1)+str(1)+str(0)+str(1))
c = QuantumCircuit(5, 5)
if 0 == 1:
    c.x(0)
if 1 == 1:
    c.x(1)
if 1 == 1:
    c.x(2)
if 0 == 1:
    c.x(3)
if 1 == 1:
    c.x(4)
c.barrier()
c.cnot(0,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(0)+str(1)+str(1)+str(1)+str(0))
c = QuantumCircuit(5, 5)
if 0 == 1:
    c.x(0)
if 1 == 1:
    c.x(1)
if 1 == 1:
    c.x(2)
if 1 == 1:
    c.x(3)
if 0 == 1:
    c.x(4)
c.barrier()
c.cnot(0,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(0)+str(1)+str(1)+str(1)+str(1))
c = QuantumCircuit(5, 5)
if 0 == 1:
    c.x(0)
if 1 == 1:
    c.x(1)
if 1 == 1:
    c.x(2)
if 1 == 1:
    c.x(3)
if 1 == 1:
    c.x(4)
c.barrier()
c.cnot(0,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(1)+str(0)+str(0)+str(0)+str(0))
c = QuantumCircuit(5, 5)
if 1 == 1:
    c.x(0)
if 0 == 1:
    c.x(1)
if 0 == 1:
    c.x(2)
if 0 == 1:
    c.x(3)
if 0 == 1:
    c.x(4)
c.barrier()
c.cnot(0,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(1)+str(0)+str(0)+str(0)+str(1))
c = QuantumCircuit(5, 5)
if 1 == 1:
    c.x(0)
if 0 == 1:
    c.x(1)
if 0 == 1:
    c.x(2)
if 0 == 1:
    c.x(3)
if 1 == 1:
    c.x(4)
c.barrier()
c.cnot(0,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(1)+str(0)+str(0)+str(1)+str(0))
c = QuantumCircuit(5, 5)
if 1 == 1:
    c.x(0)
if 0 == 1:
    c.x(1)
if 0 == 1:
    c.x(2)
if 1 == 1:
    c.x(3)
if 0 == 1:
    c.x(4)
c.barrier()
c.cnot(0,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(1)+str(0)+str(0)+str(1)+str(1))
c = QuantumCircuit(5, 5)
if 1 == 1:
    c.x(0)
if 0 == 1:
    c.x(1)
if 0 == 1:
    c.x(2)
if 1 == 1:
    c.x(3)
if 1 == 1:
    c.x(4)
c.barrier()
c.cnot(0,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(1)+str(0)+str(1)+str(0)+str(0))
c = QuantumCircuit(5, 5)
if 1 == 1:
    c.x(0)
if 0 == 1:
    c.x(1)
if 1 == 1:
    c.x(2)
if 0 == 1:
    c.x(3)
if 0 == 1:
    c.x(4)
c.barrier()
c.cnot(0,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(1)+str(0)+str(1)+str(0)+str(1))
c = QuantumCircuit(5, 5)
if 1 == 1:
    c.x(0)
if 0 == 1:
    c.x(1)
if 1 == 1:
    c.x(2)
if 0 == 1:
    c.x(3)
if 1 == 1:
    c.x(4)
c.barrier()
c.cnot(0,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(1)+str(0)+str(1)+str(1)+str(0))
c = QuantumCircuit(5, 5)
if 1 == 1:
    c.x(0)
if 0 == 1:
    c.x(1)
if 1 == 1:
    c.x(2)
if 1 == 1:
    c.x(3)
if 0 == 1:
    c.x(4)
c.barrier()
c.cnot(0,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(1)+str(0)+str(1)+str(1)+str(1))
c = QuantumCircuit(5, 5)
if 1 == 1:
    c.x(0)
if 0 == 1:
    c.x(1)
if 1 == 1:
    c.x(2)
if 1 == 1:
    c.x(3)
if 1 == 1:
    c.x(4)
c.barrier()
c.cnot(0,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(1)+str(1)+str(0)+str(0)+str(0))
c = QuantumCircuit(5, 5)
if 1 == 1:
    c.x(0)
if 1 == 1:
    c.x(1)
if 0 == 1:
    c.x(2)
if 0 == 1:
    c.x(3)
if 0 == 1:
    c.x(4)
c.barrier()
c.cnot(0,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(1)+str(1)+str(0)+str(0)+str(1))
c = QuantumCircuit(5, 5)
if 1 == 1:
    c.x(0)
if 1 == 1:
    c.x(1)
if 0 == 1:
    c.x(2)
if 0 == 1:
    c.x(3)
if 1 == 1:
    c.x(4)
c.barrier()
c.cnot(0,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(1)+str(1)+str(0)+str(1)+str(0))
c = QuantumCircuit(5, 5)
if 1 == 1:
    c.x(0)
if 1 == 1:
    c.x(1)
if 0 == 1:
    c.x(2)
if 1 == 1:
    c.x(3)
if 0 == 1:
    c.x(4)
c.barrier()
c.cnot(0,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(1)+str(1)+str(0)+str(1)+str(1))
c = QuantumCircuit(5, 5)
if 1 == 1:
    c.x(0)
if 1 == 1:
    c.x(1)
if 0 == 1:
    c.x(2)
if 1 == 1:
    c.x(3)
if 1 == 1:
    c.x(4)
c.barrier()
c.cnot(0,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(1)+str(1)+str(1)+str(0)+str(0))
c = QuantumCircuit(5, 5)
if 1 == 1:
    c.x(0)
if 1 == 1:
    c.x(1)
if 1 == 1:
    c.x(2)
if 0 == 1:
    c.x(3)
if 0 == 1:
    c.x(4)
c.barrier()
c.cnot(0,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(1)+str(1)+str(1)+str(0)+str(1))
c = QuantumCircuit(5, 5)
if 1 == 1:
    c.x(0)
if 1 == 1:
    c.x(1)
if 1 == 1:
    c.x(2)
if 0 == 1:
    c.x(3)
if 1 == 1:
    c.x(4)
c.barrier()
c.cnot(0,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(1)+str(1)+str(1)+str(1)+str(0))
c = QuantumCircuit(5, 5)
if 1 == 1:
    c.x(0)
if 1 == 1:
    c.x(1)
if 1 == 1:
    c.x(2)
if 1 == 1:
    c.x(3)
if 0 == 1:
    c.x(4)
c.barrier()
c.cnot(0,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(1)+str(1)+str(1)+str(1)+str(1))
c = QuantumCircuit(5, 5)
if 1 == 1:
    c.x(0)
if 1 == 1:
    c.x(1)
if 1 == 1:
    c.x(2)
if 1 == 1:
    c.x(3)
if 1 == 1:
    c.x(4)
c.barrier()
c.cnot(0,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()

print("Funcion 3")
print("Resultado de: ",str(0)+str(0)+str(0)+str(0)+str(0))
c = QuantumCircuit(5, 5)
if 0 == 1:
    c.x(0)
if 0 == 1:
    c.x(1)
if 0 == 1:
    c.x(2)
if 0 == 1:
    c.x(3)
if 0 == 1:
    c.x(4)
c.barrier()
c.cnot(1,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(0)+str(0)+str(0)+str(0)+str(1))
c = QuantumCircuit(5, 5)
if 0 == 1:
    c.x(0)
if 0 == 1:
    c.x(1)
if 0 == 1:
    c.x(2)
if 0 == 1:
    c.x(3)
if 1 == 1:
    c.x(4)
c.barrier()
c.cnot(1,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(0)+str(0)+str(0)+str(1)+str(0))
c = QuantumCircuit(5, 5)
if 0 == 1:
    c.x(0)
if 0 == 1:
    c.x(1)
if 0 == 1:
    c.x(2)
if 1 == 1:
    c.x(3)
if 0 == 1:
    c.x(4)
c.barrier()
c.cnot(1,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(0)+str(0)+str(0)+str(1)+str(1))
c = QuantumCircuit(5, 5)
if 0 == 1:
    c.x(0)
if 0 == 1:
    c.x(1)
if 0 == 1:
    c.x(2)
if 1 == 1:
    c.x(3)
if 1 == 1:
    c.x(4)
c.barrier()
c.cnot(1,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(0)+str(0)+str(1)+str(0)+str(0))
c = QuantumCircuit(5, 5)
if 0 == 1:
    c.x(0)
if 0 == 1:
    c.x(1)
if 1 == 1:
    c.x(2)
if 0 == 1:
    c.x(3)
if 0 == 1:
    c.x(4)
c.barrier()
c.cnot(1,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(0)+str(0)+str(1)+str(0)+str(1))
c = QuantumCircuit(5, 5)
if 0 == 1:
    c.x(0)
if 0 == 1:
    c.x(1)
if 1 == 1:
    c.x(2)
if 0 == 1:
    c.x(3)
if 1 == 1:
    c.x(4)
c.barrier()
c.cnot(1,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(0)+str(0)+str(1)+str(1)+str(0))
c = QuantumCircuit(5, 5)
if 0 == 1:
    c.x(0)
if 0 == 1:
    c.x(1)
if 1 == 1:
    c.x(2)
if 1 == 1:
    c.x(3)
if 0 == 1:
    c.x(4)
c.barrier()
c.cnot(1,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(0)+str(0)+str(1)+str(1)+str(1))
c = QuantumCircuit(5, 5)
if 0 == 1:
    c.x(0)
if 0 == 1:
    c.x(1)
if 1 == 1:
    c.x(2)
if 1 == 1:
    c.x(3)
if 1 == 1:
    c.x(4)
c.barrier()
c.cnot(1,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(0)+str(1)+str(0)+str(0)+str(0))
c = QuantumCircuit(5, 5)
if 0 == 1:
    c.x(0)
if 1 == 1:
    c.x(1)
if 0 == 1:
    c.x(2)
if 0 == 1:
    c.x(3)
if 0 == 1:
    c.x(4)
c.barrier()
c.cnot(1,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(0)+str(1)+str(0)+str(0)+str(1))
c = QuantumCircuit(5, 5)
if 0 == 1:
    c.x(0)
if 1 == 1:
    c.x(1)
if 0 == 1:
    c.x(2)
if 0 == 1:
    c.x(3)
if 1 == 1:
    c.x(4)
c.barrier()
c.cnot(1,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(0)+str(1)+str(0)+str(1)+str(0))
c = QuantumCircuit(5, 5)
if 0 == 1:
    c.x(0)
if 1 == 1:
    c.x(1)
if 0 == 1:
    c.x(2)
if 1 == 1:
    c.x(3)
if 0 == 1:
    c.x(4)
c.barrier()
c.cnot(1,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(0)+str(1)+str(0)+str(1)+str(1))
c = QuantumCircuit(5, 5)
if 0 == 1:
    c.x(0)
if 1 == 1:
    c.x(1)
if 0 == 1:
    c.x(2)
if 1 == 1:
    c.x(3)
if 1 == 1:
    c.x(4)
c.barrier()
c.cnot(1,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(0)+str(1)+str(1)+str(0)+str(0))
c = QuantumCircuit(5, 5)
if 0 == 1:
    c.x(0)
if 1 == 1:
    c.x(1)
if 1 == 1:
    c.x(2)
if 0 == 1:
    c.x(3)
if 0 == 1:
    c.x(4)
c.barrier()
c.cnot(1,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(0)+str(1)+str(1)+str(0)+str(1))
c = QuantumCircuit(5, 5)
if 0 == 1:
    c.x(0)
if 1 == 1:
    c.x(1)
if 1 == 1:
    c.x(2)
if 0 == 1:
    c.x(3)
if 1 == 1:
    c.x(4)
c.barrier()
c.cnot(1,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(0)+str(1)+str(1)+str(1)+str(0))
c = QuantumCircuit(5, 5)
if 0 == 1:
    c.x(0)
if 1 == 1:
    c.x(1)
if 1 == 1:
    c.x(2)
if 1 == 1:
    c.x(3)
if 0 == 1:
    c.x(4)
c.barrier()
c.cnot(1,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(0)+str(1)+str(1)+str(1)+str(1))
c = QuantumCircuit(5, 5)
if 0 == 1:
    c.x(0)
if 1 == 1:
    c.x(1)
if 1 == 1:
    c.x(2)
if 1 == 1:
    c.x(3)
if 1 == 1:
    c.x(4)
c.barrier()
c.cnot(1,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(1)+str(0)+str(0)+str(0)+str(0))
c = QuantumCircuit(5, 5)
if 1 == 1:
    c.x(0)
if 0 == 1:
    c.x(1)
if 0 == 1:
    c.x(2)
if 0 == 1:
    c.x(3)
if 0 == 1:
    c.x(4)
c.barrier()
c.cnot(1,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(1)+str(0)+str(0)+str(0)+str(1))
c = QuantumCircuit(5, 5)
if 1 == 1:
    c.x(0)
if 0 == 1:
    c.x(1)
if 0 == 1:
    c.x(2)
if 0 == 1:
    c.x(3)
if 1 == 1:
    c.x(4)
c.barrier()
c.cnot(1,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(1)+str(0)+str(0)+str(1)+str(0))
c = QuantumCircuit(5, 5)
if 1 == 1:
    c.x(0)
if 0 == 1:
    c.x(1)
if 0 == 1:
    c.x(2)
if 1 == 1:
    c.x(3)
if 0 == 1:
    c.x(4)
c.barrier()
c.cnot(1,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(1)+str(0)+str(0)+str(1)+str(1))
c = QuantumCircuit(5, 5)
if 1 == 1:
    c.x(0)
if 0 == 1:
    c.x(1)
if 0 == 1:
    c.x(2)
if 1 == 1:
    c.x(3)
if 1 == 1:
    c.x(4)
c.barrier()
c.cnot(1,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(1)+str(0)+str(1)+str(0)+str(0))
c = QuantumCircuit(5, 5)
if 1 == 1:
    c.x(0)
if 0 == 1:
    c.x(1)
if 1 == 1:
    c.x(2)
if 0 == 1:
    c.x(3)
if 0 == 1:
    c.x(4)
c.barrier()
c.cnot(1,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(1)+str(0)+str(1)+str(0)+str(1))
c = QuantumCircuit(5, 5)
if 1 == 1:
    c.x(0)
if 0 == 1:
    c.x(1)
if 1 == 1:
    c.x(2)
if 0 == 1:
    c.x(3)
if 1 == 1:
    c.x(4)
c.barrier()
c.cnot(1,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(1)+str(0)+str(1)+str(1)+str(0))
c = QuantumCircuit(5, 5)
if 1 == 1:
    c.x(0)
if 0 == 1:
    c.x(1)
if 1 == 1:
    c.x(2)
if 1 == 1:
    c.x(3)
if 0 == 1:
    c.x(4)
c.barrier()
c.cnot(1,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(1)+str(0)+str(1)+str(1)+str(1))
c = QuantumCircuit(5, 5)
if 1 == 1:
    c.x(0)
if 0 == 1:
    c.x(1)
if 1 == 1:
    c.x(2)
if 1 == 1:
    c.x(3)
if 1 == 1:
    c.x(4)
c.barrier()
c.cnot(1,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(1)+str(1)+str(0)+str(0)+str(0))
c = QuantumCircuit(5, 5)
if 1 == 1:
    c.x(0)
if 1 == 1:
    c.x(1)
if 0 == 1:
    c.x(2)
if 0 == 1:
    c.x(3)
if 0 == 1:
    c.x(4)
c.barrier()
c.cnot(1,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(1)+str(1)+str(0)+str(0)+str(1))
c = QuantumCircuit(5, 5)
if 1 == 1:
    c.x(0)
if 1 == 1:
    c.x(1)
if 0 == 1:
    c.x(2)
if 0 == 1:
    c.x(3)
if 1 == 1:
    c.x(4)
c.barrier()
c.cnot(1,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(1)+str(1)+str(0)+str(1)+str(0))
c = QuantumCircuit(5, 5)
if 1 == 1:
    c.x(0)
if 1 == 1:
    c.x(1)
if 0 == 1:
    c.x(2)
if 1 == 1:
    c.x(3)
if 0 == 1:
    c.x(4)
c.barrier()
c.cnot(1,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(1)+str(1)+str(0)+str(1)+str(1))
c = QuantumCircuit(5, 5)
if 1 == 1:
    c.x(0)
if 1 == 1:
    c.x(1)
if 0 == 1:
    c.x(2)
if 1 == 1:
    c.x(3)
if 1 == 1:
    c.x(4)
c.barrier()
c.cnot(1,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(1)+str(1)+str(1)+str(0)+str(0))
c = QuantumCircuit(5, 5)
if 1 == 1:
    c.x(0)
if 1 == 1:
    c.x(1)
if 1 == 1:
    c.x(2)
if 0 == 1:
    c.x(3)
if 0 == 1:
    c.x(4)
c.barrier()
c.cnot(1,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(1)+str(1)+str(1)+str(0)+str(1))
c = QuantumCircuit(5, 5)
if 1 == 1:
    c.x(0)
if 1 == 1:
    c.x(1)
if 1 == 1:
    c.x(2)
if 0 == 1:
    c.x(3)
if 1 == 1:
    c.x(4)
c.barrier()
c.cnot(1,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(1)+str(1)+str(1)+str(1)+str(0))
c = QuantumCircuit(5, 5)
if 1 == 1:
    c.x(0)
if 1 == 1:
    c.x(1)
if 1 == 1:
    c.x(2)
if 1 == 1:
    c.x(3)
if 0 == 1:
    c.x(4)
c.barrier()
c.cnot(1,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(1)+str(1)+str(1)+str(1)+str(1))
c = QuantumCircuit(5, 5)
if 1 == 1:
    c.x(0)
if 1 == 1:
    c.x(1)
if 1 == 1:
    c.x(2)
if 1 == 1:
    c.x(3)
if 1 == 1:
    c.x(4)
c.barrier()
c.cnot(1,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Funcion 4")
print("Resultado de: ",str(0)+str(0)+str(0)+str(0)+str(0))
c = QuantumCircuit(5, 5)
if 0 == 1:
    c.x(0)
if 0 == 1:
    c.x(1)
if 0 == 1:
    c.x(2)
if 0 == 1:
    c.x(3)
if 0 == 1:
    c.x(4)
c.barrier()
c.cnot(2,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(0)+str(0)+str(0)+str(0)+str(1))
c = QuantumCircuit(5, 5)
if 0 == 1:
    c.x(0)
if 0 == 1:
    c.x(1)
if 0 == 1:
    c.x(2)
if 0 == 1:
    c.x(3)
if 1 == 1:
    c.x(4)
c.barrier()
c.cnot(2,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(0)+str(0)+str(0)+str(1)+str(0))
c = QuantumCircuit(5, 5)
if 0 == 1:
    c.x(0)
if 0 == 1:
    c.x(1)
if 0 == 1:
    c.x(2)
if 1 == 1:
    c.x(3)
if 0 == 1:
    c.x(4)
c.barrier()
c.cnot(2,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(0)+str(0)+str(0)+str(1)+str(1))
c = QuantumCircuit(5, 5)
if 0 == 1:
    c.x(0)
if 0 == 1:
    c.x(1)
if 0 == 1:
    c.x(2)
if 1 == 1:
    c.x(3)
if 1 == 1:
    c.x(4)
c.barrier()
c.cnot(2,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(0)+str(0)+str(1)+str(0)+str(0))
c = QuantumCircuit(5, 5)
if 0 == 1:
    c.x(0)
if 0 == 1:
    c.x(1)
if 1 == 1:
    c.x(2)
if 0 == 1:
    c.x(3)
if 0 == 1:
    c.x(4)
c.barrier()
c.cnot(2,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(0)+str(0)+str(1)+str(0)+str(1))
c = QuantumCircuit(5, 5)
if 0 == 1:
    c.x(0)
if 0 == 1:
    c.x(1)
if 1 == 1:
    c.x(2)
if 0 == 1:
    c.x(3)
if 1 == 1:
    c.x(4)
c.barrier()
c.cnot(2,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(0)+str(0)+str(1)+str(1)+str(0))
c = QuantumCircuit(5, 5)
if 0 == 1:
    c.x(0)
if 0 == 1:
    c.x(1)
if 1 == 1:
    c.x(2)
if 1 == 1:
    c.x(3)
if 0 == 1:
    c.x(4)
c.barrier()
c.cnot(2,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(0)+str(0)+str(1)+str(1)+str(1))
c = QuantumCircuit(5, 5)
if 0 == 1:
    c.x(0)
if 0 == 1:
    c.x(1)
if 1 == 1:
    c.x(2)
if 1 == 1:
    c.x(3)
if 1 == 1:
    c.x(4)
c.barrier()
c.cnot(2,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(0)+str(1)+str(0)+str(0)+str(0))
c = QuantumCircuit(5, 5)
if 0 == 1:
    c.x(0)
if 1 == 1:
    c.x(1)
if 0 == 1:
    c.x(2)
if 0 == 1:
    c.x(3)
if 0 == 1:
    c.x(4)
c.barrier()
c.cnot(2,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(0)+str(1)+str(0)+str(0)+str(1))
c = QuantumCircuit(5, 5)
if 0 == 1:
    c.x(0)
if 1 == 1:
    c.x(1)
if 0 == 1:
    c.x(2)
if 0 == 1:
    c.x(3)
if 1 == 1:
    c.x(4)
c.barrier()
c.cnot(2,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(0)+str(1)+str(0)+str(1)+str(0))
c = QuantumCircuit(5, 5)
if 0 == 1:
    c.x(0)
if 1 == 1:
    c.x(1)
if 0 == 1:
    c.x(2)
if 1 == 1:
    c.x(3)
if 0 == 1:
    c.x(4)
c.barrier()
c.cnot(2,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(0)+str(1)+str(0)+str(1)+str(1))
c = QuantumCircuit(5, 5)
if 0 == 1:
    c.x(0)
if 1 == 1:
    c.x(1)
if 0 == 1:
    c.x(2)
if 1 == 1:
    c.x(3)
if 1 == 1:
    c.x(4)
c.barrier()
c.cnot(2,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(0)+str(1)+str(1)+str(0)+str(0))
c = QuantumCircuit(5, 5)
if 0 == 1:
    c.x(0)
if 1 == 1:
    c.x(1)
if 1 == 1:
    c.x(2)
if 0 == 1:
    c.x(3)
if 0 == 1:
    c.x(4)
c.barrier()
c.cnot(2,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(0)+str(1)+str(1)+str(0)+str(1))
c = QuantumCircuit(5, 5)
if 0 == 1:
    c.x(0)
if 1 == 1:
    c.x(1)
if 1 == 1:
    c.x(2)
if 0 == 1:
    c.x(3)
if 1 == 1:
    c.x(4)
c.barrier()
c.cnot(2,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(0)+str(1)+str(1)+str(1)+str(0))
c = QuantumCircuit(5, 5)
if 0 == 1:
    c.x(0)
if 1 == 1:
    c.x(1)
if 1 == 1:
    c.x(2)
if 1 == 1:
    c.x(3)
if 0 == 1:
    c.x(4)
c.barrier()
c.cnot(2,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(0)+str(1)+str(1)+str(1)+str(1))
c = QuantumCircuit(5, 5)
if 0 == 1:
    c.x(0)
if 1 == 1:
    c.x(1)
if 1 == 1:
    c.x(2)
if 1 == 1:
    c.x(3)
if 1 == 1:
    c.x(4)
c.barrier()
c.cnot(2,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(1)+str(0)+str(0)+str(0)+str(0))
c = QuantumCircuit(5, 5)
if 1 == 1:
    c.x(0)
if 0 == 1:
    c.x(1)
if 0 == 1:
    c.x(2)
if 0 == 1:
    c.x(3)
if 0 == 1:
    c.x(4)
c.barrier()
c.cnot(2,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(1)+str(0)+str(0)+str(0)+str(1))
c = QuantumCircuit(5, 5)
if 1 == 1:
    c.x(0)
if 0 == 1:
    c.x(1)
if 0 == 1:
    c.x(2)
if 0 == 1:
    c.x(3)
if 1 == 1:
    c.x(4)
c.barrier()
c.cnot(2,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(1)+str(0)+str(0)+str(1)+str(0))
c = QuantumCircuit(5, 5)
if 1 == 1:
    c.x(0)
if 0 == 1:
    c.x(1)
if 0 == 1:
    c.x(2)
if 1 == 1:
    c.x(3)
if 0 == 1:
    c.x(4)
c.barrier()
c.cnot(2,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(1)+str(0)+str(0)+str(1)+str(1))
c = QuantumCircuit(5, 5)
if 1 == 1:
    c.x(0)
if 0 == 1:
    c.x(1)
if 0 == 1:
    c.x(2)
if 1 == 1:
    c.x(3)
if 1 == 1:
    c.x(4)
c.barrier()
c.cnot(2,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(1)+str(0)+str(1)+str(0)+str(0))
c = QuantumCircuit(5, 5)
if 1 == 1:
    c.x(0)
if 0 == 1:
    c.x(1)
if 1 == 1:
    c.x(2)
if 0 == 1:
    c.x(3)
if 0 == 1:
    c.x(4)
c.barrier()
c.cnot(2,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(1)+str(0)+str(1)+str(0)+str(1))
c = QuantumCircuit(5, 5)
if 1 == 1:
    c.x(0)
if 0 == 1:
    c.x(1)
if 1 == 1:
    c.x(2)
if 0 == 1:
    c.x(3)
if 1 == 1:
    c.x(4)
c.barrier()
c.cnot(2,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(1)+str(0)+str(1)+str(1)+str(0))
c = QuantumCircuit(5, 5)
if 1 == 1:
    c.x(0)
if 0 == 1:
    c.x(1)
if 1 == 1:
    c.x(2)
if 1 == 1:
    c.x(3)
if 0 == 1:
    c.x(4)
c.barrier()
c.cnot(2,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(1)+str(0)+str(1)+str(1)+str(1))
c = QuantumCircuit(5, 5)
if 1 == 1:
    c.x(0)
if 0 == 1:
    c.x(1)
if 1 == 1:
    c.x(2)
if 1 == 1:
    c.x(3)
if 1 == 1:
    c.x(4)
c.barrier()
c.cnot(2,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(1)+str(1)+str(0)+str(0)+str(0))
c = QuantumCircuit(5, 5)
if 1 == 1:
    c.x(0)
if 1 == 1:
    c.x(1)
if 0 == 1:
    c.x(2)
if 0 == 1:
    c.x(3)
if 0 == 1:
    c.x(4)
c.barrier()
c.cnot(2,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(1)+str(1)+str(0)+str(0)+str(1))
c = QuantumCircuit(5, 5)
if 1 == 1:
    c.x(0)
if 1 == 1:
    c.x(1)
if 0 == 1:
    c.x(2)
if 0 == 1:
    c.x(3)
if 1 == 1:
    c.x(4)
c.barrier()
c.cnot(2,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(1)+str(1)+str(0)+str(1)+str(0))
c = QuantumCircuit(5, 5)
if 1 == 1:
    c.x(0)
if 1 == 1:
    c.x(1)
if 0 == 1:
    c.x(2)
if 1 == 1:
    c.x(3)
if 0 == 1:
    c.x(4)
c.barrier()
c.cnot(2,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(1)+str(1)+str(0)+str(1)+str(1))
c = QuantumCircuit(5, 5)
if 1 == 1:
    c.x(0)
if 1 == 1:
    c.x(1)
if 0 == 1:
    c.x(2)
if 1 == 1:
    c.x(3)
if 1 == 1:
    c.x(4)
c.barrier()
c.cnot(2,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(1)+str(1)+str(1)+str(0)+str(0))
c = QuantumCircuit(5, 5)
if 1 == 1:
    c.x(0)
if 1 == 1:
    c.x(1)
if 1 == 1:
    c.x(2)
if 0 == 1:
    c.x(3)
if 0 == 1:
    c.x(4)
c.barrier()
c.cnot(2,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(1)+str(1)+str(1)+str(0)+str(1))
c = QuantumCircuit(5, 5)
if 1 == 1:
    c.x(0)
if 1 == 1:
    c.x(1)
if 1 == 1:
    c.x(2)
if 0 == 1:
    c.x(3)
if 1 == 1:
    c.x(4)
c.barrier()
c.cnot(2,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(1)+str(1)+str(1)+str(1)+str(0))
c = QuantumCircuit(5, 5)
if 1 == 1:
    c.x(0)
if 1 == 1:
    c.x(1)
if 1 == 1:
    c.x(2)
if 1 == 1:
    c.x(3)
if 0 == 1:
    c.x(4)
c.barrier()
c.cnot(2,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()
print("Resultado de: ",str(1)+str(1)+str(1)+str(1)+str(1))
c = QuantumCircuit(5, 5)
if 1 == 1:
    c.x(0)
if 1 == 1:
    c.x(1)
if 1 == 1:
    c.x(2)
if 1 == 1:
    c.x(3)
if 1 == 1:
    c.x(4)
c.barrier()
c.cnot(2,4)
c.barrier()
c.measure([0,1,2,3,4],[4,3,2,1,0])
c.barrier()
comC = transpile(c, slt)
trabajo = slt.run(comC, shots=1000)
r = trabajo.result()
recorrido = r.get_counts(c)
print("\nEl recuento total para 00 y 11 es:",recorrido)
print(c)
plot_histogram(recorrido)
mpl.show()