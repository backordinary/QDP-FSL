# https://github.com/alvaro18101/PECC_Hackathon_2022_QuantumTarget/blob/44a56f413c0ca04a084e97e7f1f5958360c30201/Challenge-1/Q_BattleShip_1.py
#!/usr/bin/env python
# coding: utf-8

# In[1]:


from qiskit import *
from qiskit.visualization import plot_histogram

from qiskit.quantum_info import Statevector
from qiskit.visualization import array_to_latex, plot_bloch_multivector
from qiskit.extensions import UnitaryGate

from qiskit.providers.aer import AerSimulator
from qiskit.providers.fake_provider import FakeYorktown

from qiskit import execute, Aer

from qiskit import transpile


# In[2]:


import numpy as np
import math 
import matplotlib.pyplot as plt

import getpass

import random


# In[3]:


n = 4
#global tablero1
tablero1 = np.empty([n, n], dtype="<U1000")
#global tablero2
tablero2 = np.empty([n, n], dtype="<U1000")
k=1
for i in range(n):
    for j in range(n):
        tablero1[i][j] = str(k)
        tablero2[i][j] = str(k)
        k+=1
#tablero1=np.asmatrix(tablero1)

barcos1 = np.empty([n], dtype="<U1000")
barcos2 = np.empty([n], dtype="<U1000")

hits1 = np.empty([100], dtype="<U1000")
hits2 = np.empty([100], dtype="<U1000")

j1=0 #JUGADOR 1
j2=0 #JUGADOR 2

turn=0


# In[4]:


def home():
    print("ESCOGER BARCOS\n")
    escoger(1,barcos1)
    #print(barcos1)
    print("//////////////////////////////////////\n")
    escoger(2,barcos2)
    #barcos2=Qtarget_16()[0:4]
    #print(barcos2)
    print("//////////////////////////////////////\n")
    bol = True
    
    while (bol):
        jugada(2,1,barcos1,tablero1)
        #Q_jugadaPC()
        #jugada(1,"PC",barcos2,tablero2)
        jugada(1,2,barcos2,tablero2)
        print("***JUGADOR "+str(1)+" SCORE: "+str(j1)+"***")
        print("***JUGADOR "+str(2)+" SCORE: "+str(j2)+"***")
        print("//////////////////////////////////////\n")
        if (j1==4):
            bol=False
        if (j2==4):
            bol=False
        global turn
        turn+=1

def escoger(s, barcos):
    print("Jugador "+str(s)+" escoja sus 4 barcos")
    #player1 = getpass.getpass(prompt = "Player 1's turn:")
    #barcos[i]=input("Ingrese barco "+str(i+1)+"\n")
    barcos[0]=getpass.getpass(prompt ="Ingrese barco "+str(0+1)+"\n")
    #player1 = getpass.getpass(prompt = "Player 1's turn:")
    #barcos[i]=input("Ingrese barco "+str(i+1)+"\n")
    barcos[1]=getpass.getpass(prompt ="Ingrese barco "+str(1+1)+"\n")
    #player1 = getpass.getpass(prompt = "Player 1's turn:")
    #barcos[i]=input("Ingrese barco "+str(i+1)+"\n")
    barcos[2]=getpass.getpass(prompt ="Ingrese barco "+str(2+1)+"\n")
    #player1 = getpass.getpass(prompt = "Player 1's turn:")
    #barcos[i]=input("Ingrese barco "+str(i+1)+"\n")
    barcos[3]=getpass.getpass(prompt ="Ingrese barco "+str(3+1)+"\n")
    return barcos

def jugada(jugadorA, jugadorB, barcos,tablero):
    #print("TABLERO ENTRADA")
    #imprimir(tablero)
    print("-Jugador "+str(jugadorA)+" Ingrese una coordenada para atacar al jugador "+str(jugadorB))
    t = input("--Ingrese la casilla a la que quiere atacar\n")
    #for i in range(4):
      #print(t==barcos[i])
    
    if "Q" in t:
        t=t.replace("Q","")
        #JUGADA CUÁNTICA
        print("---JUGADA CUANTICA:---")
        QQ=Qtarget_()
        print("👉"+str(QQ))
        if (QQ):
            ataque(t,barcos,tablero,jugadorA)
            
            print("🟣---ATAQUE CUÁNTICO:---")
            
            hits = np.empty([100], dtype="<U1000")
            if jugadorA==1:
                hits=hits1
            if jugadorA==2:
                hits=hits2
            
            hitsnew=np.setdiff1d(hits,[''])
            #print(hitsnew)
            hits=[int(numeric_string) for numeric_string in hitsnew]
            print(hits)
            
            Qtarget = Qtarget_n(4)
            for i in range(len(hits)):
                if hits[i] in Qtarget:
                    Qtarget.remove(hits[i])
            
            t_=random.choice(Qtarget)
            global turn
            turn+=1
            if jugadorA==1:
                hits1[turn] = t_
                #print(hits1)
            if jugadorA==2:
                hits2[turn] = t_
                #print(hits2)                
            
            ataque(t_,barcos,tablero,jugadorA)
            #comparar(t_, tablero, "🟣", "Q_ATTACK!")
            
    else:
        #JUGADA NORMAL
        ataque(t,barcos,tablero,jugadorA)
            
    imprimir(tablero)
    print("")


def ataque(t,barcos,tablero,jugadorA):
    if jugadorA==1:
        hits1[turn] = t
        #print(hits1)     
    if jugadorA==2:
        hits2[turn] = t
        #print(hits2)
        
    if(t==barcos[0] or t==barcos[1] or t==barcos[2] or t==barcos[3]):
        #comparar(t, tablero, "\U0001F7E2", "SCORE")
        comparar(t, tablero, "🟢", "HIT!")
        if jugadorA==1:
            global j1
            j1+=1
        if jugadorA==2:
            global j2
            j2+=1
    else:
        #comparar(t, tablero, "\1U0001F534", "MISS")
        #tablero=comparar(t, tablero, "🔴", "FAIL!")
        comparar(t, tablero, "🔴", "FAIL!")
    
def Q_jugadaPC(jugadorA, jugadorB, barcos,tablero):
    #print("TABLERO ENTRADA")
    #imprimir(tablero)
    print("Turno de la PC")
    
def comparar(t, tablero, accion, output):
    x=-1
    y=-1
    if (t==tablero[0][0]):# or t=="\U0001F7E2" or t=="\U0001F534"):
        y=0
        x=0
        break
    else:
        continue
    if (t==tablero[0][1]):# or t=="\U0001F7E2" or t=="\U0001F534"):
        y=1
        x=0
        break
    else:
        continue
    if (t==tablero[0][2]):# or t=="\U0001F7E2" or t=="\U0001F534"):
        y=2
        x=0
        break
    else:
        continue
    if (t==tablero[0][3]):# or t=="\U0001F7E2" or t=="\U0001F534"):
        y=3
        x=0
        break
    else:
        continue
    if (t==tablero[1][0]):# or t=="\U0001F7E2" or t=="\U0001F534"):
        y=0
        x=1
        break
    else:
        continue
    if (t==tablero[1][1]):# or t=="\U0001F7E2" or t=="\U0001F534"):
        y=1
        x=1
        break
    else:
        continue
    if (t==tablero[1][2]):# or t=="\U0001F7E2" or t=="\U0001F534"):
        y=2
        x=1
        break
    else:
        continue
    if (t==tablero[1][3]):# or t=="\U0001F7E2" or t=="\U0001F534"):
        y=3
        x=1
        break
    else:
        continue
    if (t==tablero[2][0]):# or t=="\U0001F7E2" or t=="\U0001F534"):
        y=0
        x=2
        break
    else:
        continue
    if (t==tablero[2][1]):# or t=="\U0001F7E2" or t=="\U0001F534"):
        y=1
        x=2
        break
    else:
        continue
    if (t==tablero[2][2]):# or t=="\U0001F7E2" or t=="\U0001F534"):
        y=2
        x=2
        break
    else:
        continue
    if (t==tablero[2][3]):# or t=="\U0001F7E2" or t=="\U0001F534"):
        y=3
        x=2
        break
    else:
        continue
    if (t==tablero[3][0]):# or t=="\U0001F7E2" or t=="\U0001F534"):
        y=0
        x=3
        break
    else:
        continue
    if (t==tablero[3][1]):# or t=="\U0001F7E2" or t=="\U0001F534"):
        y=1
        x=3
        break
    else:
        continue
    if (t==tablero[3][2]):# or t=="\U0001F7E2" or t=="\U0001F534"):
        y=2
        x=3
        break
    else:
        continue
    if (t==tablero[3][3]):# or t=="\U0001F7E2" or t=="\U0001F534"):
        y=3
        x=3
        break
    else:
        continue
    #print("type_tablero",type(tablero[i][j]))
    #print("type t",type(t),"t: ",t,"x: ",x,"y: ",y)
    #print(bol_)
    if (x!=-1 and y!=-1):
        tablero[x][y] = accion
    print(output)
    #return tablero
    
def imprimir(tab):
    print(" "+tab[0][0],end=" ")
    print(" "+tab[0][1],end=" ")
    print(" "+tab[0][2],end=" ")
    print(" "+tab[0][3],end=" ")
    print(" ")
    print(" "+tab[1][0],end=" ")
    print(" "+tab[1][1],end=" ")
    print(" "+tab[1][2],end=" ")
    print(" "+tab[1][3],end=" ")
    print(" ")
    print(" "+tab[2][0],end=" ")
    print(" "+tab[2][1],end=" ")
    print(" "+tab[2][2],end=" ")
    print(" "+tab[2][3],end=" ")
    print(" ")
    print(" "+tab[3][0],end=" ")
    print(" "+tab[3][1],end=" ")
    print(" "+tab[3][2],end=" ")
    print(" "+tab[3][3],end=" ")
    print(" ")


# Generador de números aleatorios (1-16)

# In[5]:


def bin2dec(bin_):
    decimal = 0
    for digit in bin_:
        decimal = decimal*2 + int(digit)
    return decimal


# In[6]:


def Qtarget_n(n):
    device_backend = FakeYorktown()
    sim_yorktown = AerSimulator.from_backend(device_backend)
    N_QUBITS = n
    circ = QuantumCircuit(N_QUBITS, N_QUBITS)
    circ.h(0)
    for idx in range(N_QUBITS - 1):
        circ.cx(idx, idx + 1)
    circ.measure_all()
    #circ.draw('mpl')

    tcirc = transpile(circ, sim_yorktown)
    result_noise = sim_yorktown.run(tcirc, shots=1024).result()
    counts_noise = result_noise.get_counts(0)
    #print(counts_noise)    
    lcounts=list(counts_noise.keys())
    #print(lcounts)
    for i in range(len(lcounts)):
        lcounts[i]=str(bin2dec(lcounts[i].replace(" 0000",""))+1) # se suma 1 para que los valores vayan desde 0+1 hasta 15+1
    #print(lcounts)
    #return(lcounts[0:4])
    return(lcounts)


# Generador números aleatorios (0-1)

# In[7]:


def Qtarget_():
    circ_ghz = QuantumCircuit(3, 3)     # circuito con 3 qubits y 3 bits clásicos
    circ_ghz.h(0)                       # aplicar compuerta H al qubit 0
    circ_ghz.cx(0, 1)                   # aplicar compuerta CNOT a los qubits 0 y 1
    circ_ghz.cx(0, 2)                   # aplicar compuerta CNOT a los qubits 0 y 2
    circ_ghz.barrier()
    circ_ghz.measure([0,1,2], [0,1,2])  # medir los 3 qubits en los 3 bits clásicos
    
    circ_ghz.draw('mpl')                # mostrar el circuito
    
    simulator = Aer.get_backend('aer_simulator')

    job = execute(circ_ghz, simulator, shots=1)               # ejecutar el circuito una sola vez
    counts = job.result().get_counts(circ_ghz)                # obtener los resultados de la ejecución
    #print(counts)
    r=23
    if '1' in list(counts.keys())[0]:
        r=1
    else:
        r=0
    #print("r=",r)
    return r


# # INICIO

# In[ ]:


home()

