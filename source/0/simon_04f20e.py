# https://github.com/tamames/Computacion_Cuantica/blob/a1383253c7a43baf65eec93fe3c20ffd72f5b7dc/simon.py
# importing Qiskit
from qiskit import IBMQ, BasicAer
from qiskit.providers.ibmq import least_busy
from qiskit import QuantumCircuit, execute

# import basic plot tools
from qiskit.visualization import plot_histogram
from qiskit_textbook.tools import simon_oracle

# import other things
from numpy.random import randint
import matplotlib.pyplot as plt

def oraculo(b,n):

    # qiskit ya tiene un oraculo de simon implementado pero queria construir
    # el mio propio

    # primero definimos el circuito cuantico
    oracle=QuantumCircuit(2*n)

    # definimos un nuevo str que es igual a b pero al reves, nos servira para
    # luego
    b_invertido=b[::-1]

    # primero copiamos el estado de los qubits input a los qubits output
    for i in range(n):
        oracle.cx(i,n+i)

    # si la cadena es '00...0' no hace falta hacer nada mas, la funcion es 1-to-1
    if b=='0'*n:
        return oracle

    # chequear cual es la ultima cifra de b distinta de cero y tomar ese qubit
    # como qubit de partida de las siguientes CNOT y el target son las posiciones
    # i /b[i]=='1' pero dadas la vuelta (b_invertido)

    # asi designamos cual es el ultimo bit de b distinto de cero y tomamos ese
    # numero como la posicion del qubit control
    for i in range(len(b)):
        if b_invertido[i]=='1':
            pos_control=i
            break

    # para los qubit target necesitamos los valores de b_invertido que son '1'
    pos_target=[]
    for i in range(len(b_invertido)):
        if b_invertido[i]=='1':
            pos_target.append(i)
        else:
            continue

    # construimos las CNOT finales
    for i in pos_target:
        oracle.cx(pos_control,n+i)


    return oracle


n=3# int(input('Introduce el numero de caracteres de tu str: '))

# se genera una cadena de longitud n aleatoria de 0 y 1
t=randint(2,size=n)
b=''.join([str(i) for i in t])


simon=QuantumCircuit(2*n,n)

#aplicamos Hadamard al primer registro
simon.h(range(n))

simon.barrier()

# añadimos al circuito nuestro oraculo
simon+=oraculo(b,n)
simon.barrier()

# y volvemos a aplicar Hadamard al segundo registro
simon.h(range(n))

# medimos el segundo registro
simon.measure(range(n),range(n))


# print(simon.draw())

# simulamos el circuito
backend = BasicAer.get_backend('qasm_simulator')
shots = 1024
results = execute(simon, backend=backend, shots=shots).result()
counts = results.get_counts()
# plot_histogram(counts)
# plt.show()
#ahora falta resolver el sistema para encontrar b

# tenemos una forma de encontrar el string secreto

# esto nos genera una lista con los resultados que se han obtenido de la simulacion
z=list(counts)

def dotproduct(z,b):
    # aqui definimos una funcion para calcular el producto escalar en modulo 2
    valor=0
    for i in range(len(b)):
        valor+=int(b[i])*int(z[i])
    return (valor%2)



def comprobar(z):
    # ahora lo que vamos a hacer es ver que str de los posibles es ortogonal a
    # los cuatro que nos han salido, ese str sera el str secreto que estamos buscando
    todoslosstring=[format(i,'0'+str(n)+'b') for i in range(2**n)] # esto crea una lista
    # con todos los numeros entre 0 y 2^n en binario
    # nos sirve para comprobar que valores son ortogonales a los que salen de la medida
    result=''
    for numero in todoslosstring:
        contador=0
        for i in z:
            if (dotproduct(i,numero)==0):
                contador+=1
        if (contador==len(z)):
            result=numero

    return result

secreto=comprobar(z)
print(secreto)
print(b)
