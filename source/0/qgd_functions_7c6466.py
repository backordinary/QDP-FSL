# https://github.com/rlosadagarcia/QFT-Gate-Circuit-Decomposer/blob/86f527a8a1eb504536c71bd6269a1bb6d3e69502/Project/Run_example/QGD_functions.py
#!/usr/bin/env python
# coding: utf-8

# # Quantum Gate Decomposition
from qiskit import *
from qiskit.visualization import *
from qiskit.quantum_info import *
from qiskit.extensions import *
from qiskit.tools.monitor import *
from qiskit.providers.ibmq import *
from qiskit.circuit import *
from qiskit.circuit.library import*

#import Our_Qiskit_Functions as oq
import sympy as sy
import numpy as np
import matplotlib as mpl
import math as m
import scipy as sc
import copy

from itertools import permutations,product

from ipywidgets import interactive
from IPython.core.display import display 

S_simulator = Aer.backends(name='statevector_simulator')[0]
M_simulator = Aer.backends(name='qasm_simulator')[0]
U_simulator = Aer.get_backend('unitary_simulator')
 
#IBMQ.enable_account('690f441b74197a4d72968214bf5d7c446e999c114dcf7247f821a9604c4ac2d1136815105e210fc44485bd885148d815e441135c43f7bc7668195a5911f6cb01')
# provider = IBMQ.get_provider(hub='ibm-q')
# backend = provider.get_backend('ibmq_qasm_simulator')
#mpl.rcParams['figure.dpi'] = 75
# large_width = 4000
# np.set_printoptions(linewidth=large_width)

# ## Definiciones preliminares
# ### Productos recursivos:
# OJO: Los inputs son arrays $\left[w_0,...,w_{(n-1)}\right]$ y devuelven $ \displaystyle \bigwedge \left( w_{n-1}, \ldots \bigwedge \left( w_1, w_{0} \right) \right)$ donde $\bigwedge \left( \cdot , \cdot \right)$ es la operación a realizar
# Producto de Kroenecker recursivo
def RKron(w:np.ndarray)->np.ndarray:
    w_aux=w[0]
    for s in range(1,len(w)):
        w_aux = np.kron(w[s],w_aux)
    return w_aux

# Producto tensorial recursivo
def RProd(w:np.ndarray)->np.ndarray:
    w_aux=w[0]
    for s in range(1,len(w)):
        w_aux = np.tensordot(w[s],w_aux,axes=1)
    return w_aux

# Producto matricial recursivo
def RMult(w:np.ndarray)->np.ndarray:
    w_aux=w[0]
    for s in range(1,len(w)):
        w_aux = np.matmul(w[s],w_aux)
    return w_aux


# ### Indexaciones de parámetros:

# In[5]:


#Parámetros iniciales:

# #nº qbits
n=4


#Para n qbits hay un mínimo teórico de "layers" para desentrelazarlo de los (n-1) restantes:
#Para 2 qbits,3; para 3, 14; pueden consultarse en la referencia.
limites_puertas=[3,14,63,267]

#Nº de ciclos a implementar 
cycles=limites_puertas[0:n-1]
cycles.reverse()

# #Orden de desentrelazamiento para los qbits
# #Por defecto empezamos desentrelazando q_{n-1}-->q_{1} (puede cambiarse)
disent_order=list(range(n-1,-1,-1))


#Lista con la prioridad u orden para montar las layers del circuito:
#Realmente para cada qbit puede haber una prioridad distinta, pero por
#defecto ponemos que la prioridad general es q_{n-1}->q_{0}

#Definimos un diccionario que recoge el orden de desentrelazado y
#la prioridad de desentrelazado de cada qbit: (disentanglement_dictionary)
def disent_dict(disent_order:list):
    disent_dict={}
    disent_qbits=[]
    for k in disent_order:
        disent_qbits.append(k)
        disent_dict[k]=[q for q in disent_order if not q in disent_qbits]
    return disent_dict
dis = disent_dict(disent_order)

# #Rotaciones en las capas unitarias; en la capa final
# #[layer_mode(cx,U3cx,cz,U3cz); final_rotation_mode(rx,rz,U3)]
# mode=['cx','U3']

# #Inicializacion ('qft','invqft','random';'array_unitario_a_inplementar')
# initial='qft'

# #Juntamos todas las condiciones:
# conditions = [n,dis,cycles,mode,initial]
# # conditions


# In[6]:


# def PD(conditions:list):
#     n,dis,cycles,mode,initial = conditions
    
#     p={}
#     names=['α','β','ɣ','θ','μ','δ']
#     names2 = ['ε','σ','ρ']
    
#     if mode[0]=='cx' or mode[0]=='cz':
#         names_disp=names[0:4]
#     elif mode[0]== 'U3cx' or mode[0]== 'U3cz':
#         names_disp=names
#     else:
#         print(" Incorrect mode selected: Use 'cx' or 'cz'; 'U3cx' or 'U3cz' ")
    
#     disentangle_sequence=list(dis.keys())[0:n-1] 
#     for qbit in  disentangle_sequence:
#         ind_qbit= list(disentangle_sequence).index(qbit)
#         for i in range(cycles[ind_qbit]):
#             for j in dis[qbit]:
#                 p['Q%dC%dTL%d'%(qbit,i,j)]= [Parameter(str(i)+name+'_'+str(qbit)+'-'+str(j)) for name in names_disp]    

#     if mode[1]=='rx' or mode[1]=='rz':
#         p["RF"] = [Parameter(names2[0]+'_'+str(s)) for s in range(n-1,-1,-1)]
#     elif mode[1]=='U3':
#         p["RF"] = [Parameter(name+'_'+str(s)) for s in range(n-1,-1,-1) for name in names2]
#     else:
#         print(" Incorrect mode[1] selected: Use 'rx' or 'rz'; 'U3' ")
#     return p    


# In[7]:


#La estructura general es: {Qbit_to_disentangle:{Cycle:{Target_Qbits_ordered}}}
#Multi_Parameter_Dictionary
def MPD(conditions:list):
    n,dis,cycles,mode,initial = conditions
    
    p={}
    names=['α','β','ɣ','θ','μ','δ']
    names2 = ['ε','σ','ρ']
    if mode[0]=='cx' or mode[0]=='cz':
        names_disp=names[0:4]
    elif mode[0]== 'U3cx' or mode[0]== 'U3cz':
        names_disp=names
    else:
        print(" Incorrect mode selected: Use 'cx' or 'cz'; 'U3cx' or 'U3cz' ")
        
    disentangle_sequence=list(dis.keys())[0:n-1] 
    for qbit in  disentangle_sequence:
        ind_qbit= list(disentangle_sequence).index(qbit)
        p['Q%d'% qbit]={'C%d'%i: {'L%d'%j: [Parameter(str(i)+name+'_'+str(qbit)+'-'+str(j)) for name in names_disp] for j in dis[qbit] } for i in range(cycles[ind_qbit]) }
        
    if mode[1]=='rx' or mode[1]=='rz':
#         p["RF"] = [Parameter(names2[0]+'_'+str(s)) for s in list(range(n-1,-1,-1))]
        p["RF"] = [[Parameter(names2[0]+'_'+str(s))] for s in range(n)]
    elif mode[1]=='U3':
#         p["RF"] = [Parameter(i+'_'+str(s)) for s in list(range(n-1,-1,-1)) for i in names2]
        p["RF"] = [[Parameter(i+'_'+str(s)) for i in names2 ] for s in range(n)]
    else:
        print(" Incorrect mode[1] selected: Use 'rx' or 'rz'; 'U3' ")
    return p   

#Multi_Parameter_Dictionary inicializado aleatoriamente
def MPDIni(conditions:list):
    n,dis,cycles,mode,initial = conditions
    V = {}
    
    #Nº de parámetros por layer
    if mode[0]=='cx' or mode[0]=='cz':
        nparam_layer = 4
    elif mode[0]== 'U3cx' or mode[0]== 'U3cz':
        nparam_layer = 6
    else:
        print(" Incorrect mode[0] selected: Use 'cx' or 'cz'; 'U3cx' or 'U3cz' ")
         
    #Nº de parámetros en la fr_layer:
    if mode[1]=='rx' or mode[1]=='rz':
        nparam_frlayer = 1
    elif mode[1]=='U3':
        nparam_frlayer = 3
    else:
        print(" Incorrect mode[1] selected: Use 'rx' or 'rz'; 'U3' ")

    disentangle_sequence=list(dis.keys())[0:n-1] 
    for qbit in  disentangle_sequence:
        ind_qbit= list(disentangle_sequence).index(qbit)
        V['Q%d'% qbit]={'C%d'%i: {'L%d'%j: np.random.rand(nparam_layer) for j in dis[qbit] } for i in range(cycles[ind_qbit]) }
    V["RF"] = np.random.rand(n,nparam_frlayer)*np.pi           
#     V["RF"] = np.random.rand(nparam_frlayer)
    return V    


# In[8]:


# MPD([2,disent_dict(list(range(n-1,-1,-1))),[3,2],['cx','U3'],initial])
# MPDIni([2,disent_dict(list(range(n-1,-1,-1))),[3,2],['cx','U3'],initial])


# # Implementación del circuito cuántico

# ## Operador inicial:

# In[9]:


#Crea circuito y operador inicializadores
def Ini(n:int,initial:list):
    qr = QuantumRegister(n,name='q')
    qc = QuantumCircuit(qr,name='qc')
    
    if type(initial)==str:
        if initial=='qft':
            qc.compose(QFT(num_qubits=n, approximation_degree=0,
                           do_swaps=True, inverse=False, insert_barriers=True,
                           name='qft'),qubits=range(n),inplace=True)
            op=Operator(qc).data
        elif initial=='invqft':
            qc.compose(QFT(num_qubits=n, approximation_degree=0,
                           do_swaps=True, inverse=True, insert_barriers=True,
                           name='invqft'),qubits=range(n),inplace=True)
            op=Operator(qc).data
        elif initial=='random':
            op = random_unitary(2**n)
            RG = UnitaryGate(op)
            qc.append(RG,range(n))
        else:
            print("Error en initial[0]")
    else:
        op=initial[1]
        RG = UnitaryGate(op)
        qc.append(RG,range(n))
    qc.barrier()
    return [qc,op]


# ## QGD Circuit

# In[10]:


#Si no introducimos un circuito qc en las rutinas anteriores,
#nos devuelve el circuito mínimo qc2 sobre el que monta el ciclo/capa
def qcundefined(qc,qc2:QuantumCircuit,n:int):    
    if qc!=None:
        qc.compose(qc2,qubits= range(n),inplace=True)
        qc_return = qc
    else:
        qc_return = qc2
    return qc_return
    
#Layer unitaria 
def QLayer(n:int,ind:list,mode:list,pdict:dict,qc=None):
    qr = QuantumRegister(2,name='q')
    circ_layer = QuantumCircuit(qr)   
    
    p=getPD(pdict,ind)
    
    if mode[0]=='cx' or mode[0]=='cz':
        circ_layer.rz(p[0],0)
        circ_layer.ry(p[1],0)
        if mode[0]=='cx':
            circ_layer.rx(p[2],1)
            circ_layer.ry(p[3],1)
            circ_layer.cx(0,1)
        if mode[0]=='cz':
            circ_layer.rz(p[2],1)
            circ_layer.ry(p[3],1)
            circ_layer.cz(0,1)
    elif mode[0]=='U3cx' or mode[0]=='U3cz':
        circ_layer.u(p[0],p[1],p[2],0)
        circ_layer.u(p[3],p[4],p[5],1)
        if mode[0]=='U3cx':
            circ_layer.cx(0,1)
        if mode[0]=='U3cz':
            circ_layer.cz(0,1)
    else:
        print(" Incorrect mode selected: Use 'cx'; 'cz'; 'U3cx' or 'U3cz' ")
        
    if qc!=None:
        #display(circ_layer.draw('mpl'))
        qc.compose(circ_layer,qubits=[ind[0],ind[2]],inplace=True)
#         qc.barrier()
        #display(qc.circuit('mpl'))
        qc_return = qc
    else:
        qc_return = circ_layer
    return qc_return  
    
#Ciclo de layers 
def QCycle(n:int,dq:int,tq:list,i:int,mode:list,p:dict,qc=None):
    qr = QuantumRegister(n,name='q')
    circ_cycle = QuantumCircuit(qr)    
    
    for j in tq:
        QLayer(n,[dq,i,j],mode,p,circ_cycle)
        circ_cycle.barrier()
        
    return qcundefined(qc,circ_cycle,n)


    if qc!=None:
        qc.compose(circ_cycle,qubits= range(n),inplace=True)
        qc_return = qc
    else:
        qc_return = circ_cycle
    return qc_return
    
#Estructura desentrelazadora para el qbit dq
def QDisentangle(n:int,dq:int,tq:list,M:int,mode:list,p:dict,qc=None):
    qr = QuantumRegister(n,name='q')
    circ_qdis = QuantumCircuit(qr)
    
    for i in range(M):
        QCycle(n,dq,tq,i,mode,p,circ_qdis)
        circ_qdis.barrier()
    return qcundefined(qc,circ_qdis,n)
    
    
#Estructura desentrelazadora total
def QGDStructure(p:dict,conditions:list,qc=None):
    n,dis,cycles,mode,initial = conditions
    
    qr = QuantumRegister(n,name='q')
    circ_qdis = QuantumCircuit(qr)
     
    for dq in list(dis.keys())[0:n-1]:
        ind_dq = list(dis).index(dq)
        QDisentangle(n,dq,dis[dq],cycles[ind_dq],mode,p,circ_qdis)
        circ_qdis.barrier()
    return qcundefined(qc,circ_qdis,n)

#Capa de rotaciones finales:
def QFRLayer(p:dict,conditions:list,qc=None):
    n,dis,cycles,mode,initial = conditions
    
    qr = QuantumRegister(n,name='q')
    circ_frlayer = QuantumCircuit(qr)
    if mode[1]=='rz':
        for s in range(n):
            circ_frlayer.rz(p['RF'][s],s)
    elif mode[1]=='rx':
        ind = list(dis.keys())[0] 
        circ_frlayer.rz(p['RF'][ind][0],ind)
        for s in [s for s in range(n) if s!=ind]:
            circ_frlayer.rx(p['RF'][s][0],s)
    elif mode[1]=='U3':
        for s in range(n):
            circ_frlayer.u(p['RF'][s][0],p['RF'][s][1],p['RF'][s][2],s)
    else:
        print(" Incorrect frmode selected: Use 'rx'; 'rz'; 'U3' ")    
    return qcundefined(qc,circ_frlayer,n)

def QFR1Layer(p:dict,layer:int,conditions:list,qc=None):
    n,dis,cycles,mode,initial = conditions
    qr = QuantumRegister(n,name='q')
    circ_frlayer = QuantumCircuit(qr)
    if mode[1]=='rz':
        circ_frlayer.rz(p['RF'][layer][0],layer)
    elif mode[1]=='rx':
        ind = list(dis.keys())[0]
        if layer==ind:
            circ_frlayer.rz(p['RF'][layer][0],layer)
        else:
            circ_frlayer.rx(p['RF'][layer][0],layer)
    elif mode[1]=='U3':
        circ_frlayer.u(p['RF'][layer][0],p['RF'][layer][1],p['RF'][layer][2],layer)
    else:
        print(" Incorrect frmode selected: Use 'rx'; 'rz'; 'U3' ")    
    return qcundefined(qc,circ_frlayer,n)


# ## Montar todo el circuito:

# In[11]:


#Circuito y operador total
# Posibilidad de incluir el operador inicializador, estructura QFD y capa final de rotaciones con parts
# parts = [Inicializador,QGD,FinalRotationLayers] (boolean)

#Puede ser de utilidad sacar el operador únicamente de la estructura que desentrelaza un único qbit,
#De los qbits que se van a desentrelazar: disent_order[0:n-1],
#podemos poner los índices de los circuito/operadores asociados
#  [,[n-1,..,1],] sería igual que [,True,];
#También admite [,[j],] con disent_order[n]<j<disent_order[0]

#Idem con las capas de rotacion

#Por defecto devuelve circuito total
def getQC(p:dict,conditions:list,parts=[True,True,True]):
    n,dis,cycles,mode,initial = conditions
    if parts[0]==True:
        qc_ini = Ini(n,initial)[0]
    else:
        qc_ini = None
    
    if parts[1]==True:
        qc_qgf = QGDStructure(p,conditions,qc_ini)
    elif type(parts[1])==list:
        if qc_ini == None:
            qc_ini = QuantumCircuit(n)
        for dq in parts[1]:
            ind_dq = list(dis).index(dq)
            QDisentangle(n,dq,dis[dq],cycles[ind_dq],mode,p,qc_ini)
        qc_qgf = qc_ini
    else:
        qc_qgf = qc_ini
        
    if parts[2]==True:
        qc_to_return = QFRLayer(p,conditions,qc_qgf)
    elif type(parts[2])==list:
        if qc_qgf == None:
            qc_qgf = QuantumCircuit(n)
        for dq in parts[2]:
            QFR1Layer(p,dq,conditions,qc_qgf)
        qc_to_return = qc_qgf
    else:
        qc_to_return = qc_qgf
    
    return qc_to_return

def opQC(p:dict,conditions:list,parts=[True,True,True]):
    return Operator(getQC(p,conditions,parts)).data


# In[12]:


#Introducimos nombre o nº del índice del diccionario y sacamos los parámetros
def getPD(p:dict,indexes:list):
    if type(indexes)==str:
        to_return = p[indexes]
    elif indexes[0]=='RF':
        to_return = p['RF'][indexes[1]]
    elif type(indexes)==list:            
        if len(indexes)==3:
            if type(indexes[0])==str:
                to_return = p[indexes[0]][indexes[1]][indexes[2]]
            elif type(indexes[0])==int:
                to_return = p['Q%d'%indexes[0]]['C%d'%indexes[1]]['L%d'%indexes[2]]
            else:
                print("Error")
        elif len(indexes)==2:
            if type(indexes[0])==str:
                to_return = p[indexes[0]][indexes[1]]
            elif type(indexes[0])==int:
                to_return = p['Q%d'%indexes[0]]['C%d'%indexes[1]]
            else:
                print("Error")
        elif len(indexes)==1:
            if type(indexes[0])==str:
                to_return = p[indexes[0]]
            elif type(indexes[0])==int:
                to_return = p['Q%d'%indexes[0]]
            else:
                print("Error")
        else:
            print("Error length")
    else:
        print("Error type of Index")
    return to_return


# In[13]:


# #Probar a montar el circuito completo
# n=3
# cycles=[2,1]
# disent_order=list(range(n-1,-1,-1))
# dis = disent_dict(disent_order)
# mode=['cx','U3']
# initial='qft'
# conditions = [n,dis,cycles,mode,initial]
# p=MPD(conditions)
# p0=MPDIni(conditions)


# display(getQC(MPD(conditions),conditions,[False,[1],[2,1]]).draw(output='mpl',fold=-1))
# display(QGDStructure(p,conditions).draw(output='mpl',fold=-1))
# display(QFRLayer(p,conditions).draw(output='mpl',fold=-1))
# display(getQC(p,conditions,[True,True,False]).draw(output='mpl',fold=-1))
# opQC(p0,conditions,[True,True,False])


# ## Asignación de parámetros:

# In[14]:


#Update Circuit with Parameters
def UpQC(p0:dict,p:dict,indexes,qc: QuantumCircuit):
    if indexes=='All':
        to_return = qc.assign_parameters({getPD(p,[dq,i,j])[s]: getPD(p0,[dq,i,j])[s] for dq in list(p.keys())[:-1] for i in p[dq].keys() for j in p[dq][i].keys() for s in range(len(p[dq][i][j]))}).assign_parameters({getPD(p,'RF')[j][s]: getPD(p0,'RF')[j][s] for j in range(len(p['RF'])) for s in range(len(p['RF'][j])) })
    elif indexes=='RF':
        to_return = qc.assign_parameters({getPD(p,'RF')[j][s]: getPD(p0,'RF')[j][s] for j in range(len(p['RF'])) for s in range(len(p['RF'][j]))})
    elif indexes[0]=='RF':
#         to_return = qc.assign_parameters({getPD(p,'RF')[indexes[1]][s]: getPD(p0,'RF')[indexes[1]][s] for s in range(len(p['RF'][indexes[1]])) })
        to_return = qc.assign_parameters({getPD(p,indexes)[s]: getPD(p0,indexes)[s] for s in range(len(p['RF'][indexes[1]])) })
    elif indexes=='QGD':
        to_return = qc.assign_parameters({getPD(p,[dq,i,j])[s]: getPD(p0,[dq,i,j])[s] for dq in list(p.keys())[:-1] for i in p[dq].keys() for j in p[dq][i].keys() for s in range(len(p[dq][i][j]))})
    elif type(indexes)==list:
        if len(indexes)==1:
            to_return = qc.assign_parameters({getPD(p,['Q%d'%indexes[0],i,j])[s]: getPD(p0,['Q%d'%indexes[0],i,j])[s] for i in getPD(p,indexes).keys() for j in getPD(p,['Q%d'%indexes[0],i]).keys() for s in range(len(getPD(p,['Q%d'%indexes[0],i,j])))})
        elif len(indexes)==2:
            to_return = qc.assign_parameters({getPD(p,['Q%d'%indexes[0],'C%d'%indexes[1],j])[s]: getPD(p0,['Q%d'%indexes[0],'C%d'%indexes[1],j])[s] for j in getPD(p,['Q%d'%indexes[0],'C%d'%indexes[1]]).keys() for s in range(len(getPD(p,['Q%d'%indexes[0],'C%d'%indexes[1],j])))})
        elif len(indexes)==3:
            to_return = qc.assign_parameters({getPD(p,['Q%d'%indexes[0],'C%d'%indexes[1],'L%d'%indexes[2]])[s]: getPD(p0,['Q%d'%indexes[0],'C%d'%indexes[1],'L%d'%indexes[2]])[s] for s in range(len(getPD(p,['Q%d'%indexes[0],'C%d'%indexes[1],'L%d'%indexes[2]])))})        
        else:
            print("Error")
    else:
        print("Error")
    return to_return


# In[15]:


# n=3
# cycles=[2,1]
# disent_order=list(range(n-1,-1,-1))
# dis = disent_dict(disent_order)
# mode=['cx','U3']
# initial='qft'
# conditions = [n,dis,cycles,mode,initial]
# p=MPD(conditions)
# p0=MPDIni(conditions)

# qc = getQC(p,conditions)
# qc.parameters
# #Updateamos todos los parámetros del circuito: UpQC(p0,p,'All',qc)
# display(UpQC(p0,p,'All',qc).draw(output='mpl',fold=-1))

# #Updateamos solamente [qbit_to_disentangle,ciclo,layer=target_qbit]: UpQC(p0,p,[qbit,ciclo,layer],qc)
# display(UpQC(p0,p,[2],qc).draw(output='mpl',fold=-1))
# display(UpQC(p0,p,[2,0],qc).draw(output='mpl',fold=-1))
# display(UpQC(p0,p,[2,0,1],qc).draw(output='mpl',fold=-1))

# #Updateamos solamente ['RF',layer=target_qbit]: UpQC(p0,p,['RF',layer=target_qbit],qc)
# display(UpQC(p0,p,['RF',2],qc).draw(output='mpl',fold=-1))


# # Funcion de coste

# ## Distancia entre operadores unitarios y fidelidad:

# In[16]:


#Hilbert-Schmidt test:
def HSTest(U,V):
    d=U.shape[0]
    return 1-(d)**(-2)*np.abs(np.matmul(V,U).trace())**2
def HSTest2(UV):
    d=UV.shape[0]
    return 1-(d)**(-4)*np.abs(UV.trace())**2

#Gate fidelity:
def GF_HS(U,V):
    d=U.shape[0]
    return 1-d/(d+1)*HSTest(U,V)
def GF_HS2(UV):
    d=UV.shape[0]
    return 1-d/(d+1)*HSTest2(UV)



#Frobenius norm:
def FTest(U,V):
    d=U.shape[0]
    return d-np.real(np.matmul(V,U).trace())
def FTest2(UV):
    d=UV.shape[0]
    return d-np.real(UV.trace())
#Gate fidelity of the test:
def GF_F(U,V):
    d=U.shape[0]
    return 1-d/(d+1)+1/(d*(d+1))*(d-FTest(U,V))**2
def GF_F2(UV):
    d=UV.shape[0]
    return 1-d/(d+1)+1/(d*(d+1))*(d-FTest2(UV))**2


# ## Definiciones auxiliares:

# In[17]:


#Indexar con (l,m) la submatriz del operador unitario total del circuito
def isa(l: int,m: int, n:int):
    half = 2**(n-1)
    return tuple([slice(l*half,(l+1)*half,1),slice(m*half,(m+1)*half,1)])


# In[18]:


#Update diccionario de parámetros en el sitio a optimizar:
# (Update Parameter Dictionary)
def UpPD(lam,input_dict:dict,ind,overwrite=False,boolean=False):
#     dq,i,j // 'RF' = ind   // ['RF',qbit_rotacion]

    if overwrite==True:
        dict_aux=input_dict
    elif overwrite==False:
        #Deep copy of the dictionary
        dict_aux = copy.deepcopy(input_dict)
    else:
        print("Error")
    
    if boolean==True:
        print("Input hasn't changed")
    else:
        if ind=='RF':
            dict_aux['RF'] = pack(lam)
        elif ind[0]=='RF':
            dict_aux['RF'][ind[1]]=lam
        elif type(ind)==list:
            dict_aux['Q%d'%ind[0]]['C%d'%ind[1]]['L%d'%ind[2]] = lam
        else:
            print("Error")
    return dict_aux


# In[19]:


# UpPD([0]*3,p,['RF',2],False)


# # Función de coste:

# In[20]:


# Tipo de función de coste: 'Disentangle','Hilbert-Schmidt','Frobenius'
def CF(lam,tipo:str,input_dict:dict,ind,conditions:list,parts:list,b=False):

    n,dis,cycles,mode,initial = conditions
#     dq,i,j // 'RF' = ind  // ['RF',qbit_rotacion]
    
    #Diccionario con parámetros lam en sitio a optimizar
    p_lam = UpPD(lam,input_dict,ind,False,b)
    #Operador unitario del circuito
    U = opQC(p_lam,conditions,parts)
    
    if tipo=='Disentangle':
        #Kappa:
        r = range(0,2)
        K = np.array([np.matmul(U[isa(i,j,n)],U[isa(p,q,n)].swapaxes(-2, -1).conj()) for i in r for j in r for p in r for q in r])
        #OJO U_ij · U_pq^(dagg) luego usamos matmul directamente o RProd al revés

        #Array identidad con el elemento [0,0] de cada matriz en K
        I = np.identity(2**(n-1))
        IK = np.array([K[:,0,0][s]*I for s in range(len(K[:,0,0]))])

        H = K-IK
        #1 np.abs(H)**2
        #2 np.multiply(H,H.conj())
        
        to_return = np.multiply(H,H.conj()).sum()
    elif tipo=='Hilbert-Schmidt' or tipo=='HS':
        to_return = HSTest2(U)
    elif tipo=='Frobenius' or tipo=='F':
        to_return = FTest2(U)
    elif tipo=='F+HS' or tipo=='HS+F':
        to_return = FTest2(U)+HSTest2(U)
    else:
        print("Error en el tipo seleccionado")
    return to_return    
def CFMult(lam,tipos:list,input_dict:dict,ind,conditions:list,parts:list,b=False):
    return sum([CF(lam,tipos[s],input_dict,ind,conditions,parts[s],b=False) for s in range(len(tipos))])

#Función de coste Raw a evaluar array de valores iniciales:
def RCF(evaluar: dict,tipo,conditions:list,parts=[True,True,False]):
    return CF(None,tipo,evaluar,None,conditions,parts,True)

def MRCF(evaluar: dict,conditions:list,tipos:list,parts:list,suma=True):
    if suma==True:
        to_return = sum([RCF(evaluar,tipos[s],conditions,parts[s]) for s in range(len(tipos))])
    elif suma==False:
        to_return = [RCF(evaluar,tipos[s],conditions,parts[s]) for s in range(len(tipos))]
    else:
        print("Error")
    return to_return


# ## Pack/Unpack; save/load 

# In[21]:


from itertools import chain
#Para optimizar toda la layers de rotaciones a la vez tienen
#que estar al mismo nivel como argumentos los parámetros a optimizar
def unpack(A):
    return list(chain.from_iterable(A))

def pack(A):
    d=int(len(A)/3)
    return [np.array(A[3*s:3*(s+1)]) for s in range(d)]


# In[22]:


import json
def pack_to_save(input_dict):
    param_dict = copy.deepcopy(input_dict)
    for key in list(param_dict.keys())[:-1]:
        for key2 in list(param_dict[key].keys()):
            for key3 in list(param_dict[key][key2].keys()):
                param_dict[key][key2][key3]=list(param_dict[key][key2][key3])
    param_dict['RF']=unpack(param_dict['RF'])
    return param_dict

def unpack_to_load(input_dict):
    param_dict = copy.deepcopy(input_dict)
    for key in list(param_dict.keys())[:-1]:
        for key2 in list(param_dict[key].keys()):
            for key3 in list(param_dict[key][key2].keys()):
                param_dict[key][key2][key3]=np.array(param_dict[key][key2][key3])
    param_dict['RF']=pack(param_dict['RF'])
    return param_dict


# In[23]:


def save_to_file(inputs:dict,file:str):
    with open(file, 'w') as convert_file:
        convert_file.write(json.dumps(json.dumps(pack_to_save(inputs))))

import ast
def retrieve(file:str):
    return unpack_to_load(ast.literal_eval(json.loads(open(file,"r").read())))


# ## Medir con función

# In[24]:


def SMeasure(qc: QuantumCircuit,sim:object, nshots: int):
    job = execute(qc,sim,shots = nshots)
    results = job.result()
    counts = results.get_counts()
    return display(counts,plot_histogram(counts))

def MMeasure(qc: QuantumCircuit,sim:object):
    qcc = qc.copy('measure')
    qcc.measure_all()
    job = execute(qcc,sim)
    results = job.result()
    counts = results.get_counts()
    return display(counts,plot_histogram(counts))

def UMeasure(qccc: QuantumCircuit,sim:object):
    job = execute(qccc,sim)
    results = job.result()
    return results.get_unitary()

def SSimulation(qc:QuantumCircuit,sim:object):
    job = execute(qc, sim)
    state = job.result().get_statevector()
    for i in range(2**n):
        s = format(i,"b") # Convert to binary
        s = (n-len(s))*"0"+s # Prepend zeroes if needed
        print("Amplitude of",s,"=",state[i])

    for i in range(2**n):
        s = format(i,"b") # Convert to binary
        s = (n-len(s))*"0"+s # Prepend zeroes if needed
        print("Probability of",s,"=",abs(state[i])**2)
        
def MMeasure2(qcA: QuantumCircuit,qcB:QuantumCircuit,sim:object):
    qccA = qcA.copy(qcA.name+'measure')
    qccA.measure_all()
    jobA = execute(qccA,sim)
    resultsA = jobA.result()
    
    qccB = qcB.copy(qcB.name+'measure')
    qccB.measure_all()
    jobB = execute(qccB,sim)
    resultsB = jobB.result()
    
    return plot_histogram([resultsA.get_counts(qccA),resultsB.get_counts(qccB)],legend=[qccA.name,qccB.name])




#def Optimizacion(conditions:list,cf_type:list,opt_type:list,opt_parameters=[1e-6,1e-6,15,6],p_ini=False,displays=False):
#    n,dis,cycles,mode,initial = conditions
#
#    p=MPD(conditions)
#
#    eps,tol,Nmax,Nopt = opt_parameters
#
#    num_it=0
#    if p_ini==False:
#        print("Iterante inicial aleatorio")
#        print("--------------------------------------------- \n\r")
#        p_ini=MPDIni(conditions)
#    else:
#        print("Iterante inicial forzado")
#        print("--------------------------------------------- \n\r")
#    p_opt=copy.deepcopy(p_ini)
#
##     Opt=[]
##     Coste=[]
#    b=False
#
#
#
#    disentangle_sequence=list(dis.keys())[0:n-1]
##     parts=[[True,disentangle_sequence[0:s+1],False] for s in range(n-1)]
#
#    if opt_type[0]=='It':
#        disent_parts=[[True,disentangle_sequence[0:s+1],False] for s in range(n-1)]
#    elif opt_type[0]=='Sim':
#        disent_parts=[[True,True,False]]*(n-1)
#    else:
#        print("Error in opt type")
#    finalrot_parts=[True,True,True]
#
#    Resultados_Intento=[]
#
#    if cf_type[2]=='separated':
#        Resultados_Intento.append([[20,20],[20,20]])
#        Test=Resultados_Intento[num_it][0][0]+Resultados_Intento[num_it][1][0]
#    elif cf_type[2]=='merged':
#        Resultados_Intento.append([20,20])
#        Test=Resultados_Intento[num_it][0]
#    else:
#        print("Error")
#
#    while Test>eps:
#        if num_it>Nmax:
#            print("Max iterations reached")
#            break
#        print("Vuelta %d \n\r" % num_it)
#        for dq in disentangle_sequence:
#            ind_dq=list(disentangle_sequence).index(dq)
#            for i in range(cycles[ind_dq]):
#                #i indice de ciclo
#                ind_j=0
#                for j in dis[dq]:
#                    #j nº de layer target
#                    #ind_j índice de conteo
#                    ind=[dq,i,j]
#                    print("Qbit %d Ciclo %d Layer %d" % (dq,i,j))
#
#                    if cf_type[2]=='separated':
#                        OLayer=sc.optimize.minimize(CF,getPD(p_opt,ind),
#                                                    args=(cf_type[0],p_opt,ind,conditions,disent_parts[ind_dq],b),
#                                                    bounds=[(0,np.pi)]*(4),tol=1e-14)
#                    elif cf_type[2]=='merged':
#                        OLayer=sc.optimize.minimize(CFMult,getPD(p_opt,ind),
#                                                    args=(cf_type[0:2],p_opt,ind,conditions,[disent_parts[ind_dq],finalrot_parts],b),
#                                                    bounds=[(0,np.pi)]*(4),tol=1e-14)
#                    else:
#                        print("Error")
#                    Oresult=[OLayer.fun,OLayer.x]
#
#
#                    if opt_type[1]==True:
#                        for s in range(Nopt):
#                            print("Minimizar Iteracion, intento %d de optimizar Q%dC%dL%d" % (s,dq,i,j))
#                            if s==0:
#                                if cf_type[2]=='separated':
#                                    Osub=sc.optimize.minimize(CF,getPD(p_opt,ind),
#                                                              args=(cf_type[0],p_opt,ind,conditions,disent_parts[ind_dq],b),
#                                                              bounds=[(0,np.pi)]*(4),tol=1e-14)
#                                elif cf_type[2]=='merged':
#                                    Osub=sc.optimize.minimize(CFMult,getPD(p_opt,ind),
#                                                              args=(cf_type[0:2],p_opt,ind,conditions,[disent_parts[ind_dq],finalrot_parts],b),
#                                                              bounds=[(0,np.pi)]*(4),tol=1e-14)
#                                else:
#                                    print("Error")
#                            else:
#                                if cf_type[2]=='separated':
#                                    Osub=sc.optimize.minimize(CF,p_to_opt,
#                                                          args=(cf_type[0],p_opt,ind,conditions,disent_parts[ind_dq],b),
#                                                          bounds=[(0,np.pi)]*(4),tol=1e-14)
#                                elif cf_type[2]=='merged':
#                                    Osub=sc.optimize.minimize(CFMult,p_to_opt,
#                                                              args=(cf_type[0:2],p_opt,ind,conditions,[disent_parts[ind_dq],finalrot_parts],b),
#                                                              bounds=[(0,np.pi)]*(4),tol=1e-14)
#                                else:
#                                    print("Error")
#
#
#
#                            if s>int(Nopt/2) and abs(Osub.fun-Oresult[0])<tol:
#                                Oresult=[Osub.fun,Osub.x]
#                                print("Exit tolerancia, FCoste: \n\r", Oresult[0])
#                                break
#                            elif abs(Osub.fun-Oresult[0])<tol:
#                                print("Randomizamos iterante inicial layer \n\r")
#                                p_to_opt=np.random.rand(len(getPD(p_opt,ind)))
#                            elif Osub.fun<Oresult[0]:
#                                Oresult=[Osub.fun,Osub.x]
#                                p_to_opt=Osub.x
#                                print("Actualizamos iterante al anterior \n\r", Oresult[0])
#                            else:
#                                print("Randomizamos iterante inicial layer \n\r")
#                                p_to_opt=np.random.rand(len(getPD(p_opt,ind)))
#
#
#
#
#                    UpPD(Oresult[1],p_opt,ind,overwrite=True)
#    #                 Coste.append(Oresult[0])
##                     print("---------------------------------------------")
##                     print("Intento ",num_it)
##                     print("Optimization Q%dC%dL%d result" %(ind_dq,i,j))
#                    print("FCoste Disentangle Q%dC%dL%d \n\r" % (dq,i,j) ,Oresult[0])
#                    print("--------------------------------------------- \n\r")
#                    ind_j+=1
#
#        ind='RF'
#        unziped_parameters=unpack(getPD(p_opt,ind))
#        print("Capa de Rotaciones Finales\n\r")
#        if cf_type[2]=='separated':
#            OFRLayer=sc.optimize.minimize(CF,unziped_parameters,
#                                  args=(cf_type[1],p_opt,ind,conditions,finalrot_parts),
#                                  bounds=[(0,np.pi)]*3*n,tol=1e-14)
#        elif cf_type[2]=='merged':
#            OFRLayer=sc.optimize.minimize(CFMult,unziped_parameters,
#                              args=(cf_type[0:2],p_opt,ind,conditions,[[True,True,False],finalrot_parts]),
#                              bounds=[(0,np.pi)]*3*n,tol=1e-14)
#        else:
#            print("Error")
#
#
#
#        OFRresult=[OFRLayer.fun,OFRLayer.x]
#        if opt_type[1]==True:
#
#            for s in range(Nopt):
#                print("Minimizar Iteracion, intento %d de optimizar RF" % s)
#                if s==0:
#                    if cf_type[2]=='separated':
#                        Osub=sc.optimize.minimize(CF,unziped_parameters,
#                                                  args=(cf_type[1],p_opt,ind,conditions,finalrot_parts),
#                                                  bounds=[(0,np.pi)]*3*n,tol=1e-14)
#                    elif cf_type[2]=='merged':
#                        Osub=sc.optimize.minimize(CFMult,unziped_parameters,
#                                                  args=(cf_type[0:2],p_opt,ind,conditions,[[True,True,False],finalrot_parts]),
#                                                  bounds=[(0,np.pi)]*3*n,tol=1e-14)
#                    else:
#                        print("Error")
#                else:
#                    if cf_type[2]=='separated':
#                        Osub=sc.optimize.minimize(CF,unziped_parameters,
#                                              args=(cf_type[1],p_opt,ind,conditions,finalrot_parts),
#                                              bounds=[(0,np.pi)]*3*n,tol=1e-14)
#                    elif cf_type[2]=='merged':
#                        Osub=sc.optimize.minimize(CFMult,unziped_parameters,
#                                                  args=(cf_type[0:2],p_opt,ind,conditions,[[True,True,False],finalrot_parts]),
#                                                  bounds=[(0,np.pi)]*3*n,tol=1e-14)
#                    else:
#                        print("Error")
##
#
#
#                if s>int(Nopt/2) and abs(Osub.fun-Oresult[0])<tol:
#                    OFRresult=[Osub.fun,Osub.x]
#                    print("Exit tolerancia, FCoste: \n\r", OFRresult[0])
#                    break
#                elif abs(Osub.fun-Oresult[0])<tol:
#                    print("Randomizamos iterante inicial layer \n\r")
#                    p_to_opt=np.random.rand(len(getPD(p_opt,ind)))
#                elif Osub.fun<Oresult[0]:
#                    OFRresult=[Osub.fun,Osub.x]
#                    p_to_opt=Osub.x
#                    print("Actualizamos iterante al anterior \n\r", OFRresult[0])
#                else:
#                    print("Randomizamos iterante inicial layer \n\r")
#                    p_to_opt=np.random.rand(len(unpack(getPD(p_opt,ind))))
#
#        UpPD(OFRresult[1],p_opt,ind,overwrite=True)
##                 Coste.append(Oresult[0])
#        print("Optimization RF result \n\r",OFRresult[0])
#        print("--------------------------------------------- \n\r")
#
#        print("---------------------------------------------")
#        print("---------------------------------------------")
#        print("FCoste Total Vuelta %d \n\r" % num_it,OFRresult[0])
#        print("---------------------------------------------")
#        print("--------------------------------------------- \n\r")
#
#        if cf_type[2]=='separated':
#            Resultados_Intento.append([Oresult,OFRresult])
#            Test = abs(Resultados_Intento[num_it][0][0]+Resultados_Intento[num_it][1][0]-Resultados_Intento[num_it-1][0][0]-Resultados_Intento[num_it-1][1][0])
#        elif cf_type[2]=='merged':
#            Resultados_Intento.append(OFRresult)
#            Test = abs(Resultados_Intento[num_it][0]-Resultados_Intento[num_it-1][0])
#        else:
#            print("Error")
#
#        num_it+=1
#
#        if Test<tol:
#            print("Break_tolerancia")
#            break
#
#    if cf_type[2]=='separated':
#        to_return = [Resultados_Intento[-1][0][0]+Resultados_Intento[-1][1][0],p_opt]
#    elif cf_type[2]=='merged':
#        to_return = [Resultados_Intento[-1][0],p_opt]
#    else:
#        print("Error")
#
#    return to_return


def find_indices(lst, condition):
    return [i for i, elem in enumerate(lst) if condition(elem)]

# [list(e) for e in list(permutations(list(r), n-1))]
def ObtPerm(n,ciclo_max=10):
#     return [list(e) for e in list(permutations(list(range(1,ciclo_max)), n-1))]
    return [list(p) for p in product(list(range(1,ciclo_max)), repeat=n-1)]

def PoolCycles(n,ciclo_max=10):
    cota_inf = (n-1)*n/2
    cota_sup = n**2
    permutaciones = ObtPerm(n,ciclo_max)
    costes_ciclos = list(range(n-1,0,-1))
    costes_permutaciones = [np.dot(permutaciones[i],costes_ciclos) for i in range((ciclo_max-1)**(n-1))]
    indices_admitidos = find_indices(costes_permutaciones, lambda e: cota_inf <= e <= cota_sup)
    return [permutaciones[i] for i in indices_admitidos]


def Optimizacion(conditions:list,cf_type:list,opt_type:list,opt_parameters=[1e-6,1e-6,20,6,20,5],p_ini=False,displays=False):
    n,dis,cycles,mode,initial = conditions
    
    p=MPD(conditions)
    
    eps,tol,Nmax,Nopt,Coste_exit,Coste_extend = opt_parameters
    
    num_it=0
    if p_ini==False:
        print("Iterante inicial aleatorio")
        print("--------------------------------------------- \n\r")
        p_ini=MPDIni(conditions)
    else:
        print("Iterante inicial forzado")
        print("--------------------------------------------- \n\r")
    p_opt=copy.deepcopy(p_ini)

#     Opt=[]
#     Coste=[]
    b=False
        
        
        
    disentangle_sequence=list(dis.keys())[0:n-1]
#     parts=[[True,disentangle_sequence[0:s+1],False] for s in range(n-1)]
    
    if opt_type[0]=='It':
        disent_parts=[[True,disentangle_sequence[0:s+1],False] for s in range(n-1)]
    elif opt_type[0]=='Sim':
        disent_parts=[[True,True,False]]*(n-1)
    else:
        print("Error in opt type")
    finalrot_parts=[True,True,True]

    Resultados_Intento=[]
    
    if cf_type[2]=='separated':
        Resultados_Intento.append([[20,20],[20,20]])
        Test=Resultados_Intento[num_it][0][0]+Resultados_Intento[num_it][1][0]
    elif cf_type[2]=='merged':
        Resultados_Intento.append([20,20])
        Test=Resultados_Intento[num_it][0]
    else:
        print("Error")
    
    while Test>eps:
        if num_it>100 and Test<1e-3:
            print("La convergencia es demasiado lenta, reiniciamos")
            break
        if num_it>Nmax:
            print("Max iterations reached")
            if cf_type[2]=='separated' and Resultados_Intento[num_it][0][0]+Resultados_Intento[num_it][1][0]<Coste_extend:
                print("Extendemos Nmax de vueltas para la optimizacion")
                Nmax=Nmax*2
            elif cf_type[2]=='merged' and Resultados_Intento[num_it][0]<Coste_extend:
                print("Extendemos Nmax de vueltas para la optimizacion")
                Nmax=Nmax*2
            else:
                print("No extendemos Nmax, se alcanza Nmax y no parece que converja")
                break
        print("Vuelta %d \n\r" % num_it)
        for dq in disentangle_sequence:
            ind_dq=list(disentangle_sequence).index(dq)
            for i in range(cycles[ind_dq]):
                #i indice de ciclo
                ind_j=0
                for j in dis[dq]:
                    #j nº de layer target
                    #ind_j índice de conteo
                    ind=[dq,i,j]
                    if displays==True:
                        print("Qbit %d Ciclo %d Layer %d" % (dq,i,j))
                    
                    if cf_type[2]=='separated':
                        OLayer=sc.optimize.minimize(CF,getPD(p_opt,ind),
                                                    args=(cf_type[0],p_opt,ind,conditions,disent_parts[ind_dq],b),
                                                    bounds=[(0,np.pi)]*(4),tol=1e-14)
                    elif cf_type[2]=='merged':
                        OLayer=sc.optimize.minimize(CFMult,getPD(p_opt,ind),
                                                    args=(cf_type[0:2],p_opt,ind,conditions,[disent_parts[ind_dq],finalrot_parts],b),
                                                    bounds=[(0,np.pi)]*(4),tol=1e-14)
                    else:
                        print("Error")
                    Oresult=[OLayer.fun,OLayer.x]
                    
                    
                    if opt_type[1]==True:
                        for s in range(Nopt):
                            print("Minimizar Iteracion, intento %d de optimizar Q%dC%dL%d" % (s,dq,i,j))
                            if s==0:
                                if cf_type[2]=='separated':
                                    Osub=sc.optimize.minimize(CF,getPD(p_opt,ind),
                                                              args=(cf_type[0],p_opt,ind,conditions,disent_parts[ind_dq],b),
                                                              bounds=[(0,np.pi)]*(4),tol=1e-14)
                                elif cf_type[2]=='merged':
                                    Osub=sc.optimize.minimize(CFMult,getPD(p_opt,ind),
                                                              args=(cf_type[0:2],p_opt,ind,conditions,[disent_parts[ind_dq],finalrot_parts],b),
                                                              bounds=[(0,np.pi)]*(4),tol=1e-14)
                                else:
                                    print("Error")
                            else:
                                if cf_type[2]=='separated':
                                    Osub=sc.optimize.minimize(CF,p_to_opt,
                                                          args=(cf_type[0],p_opt,ind,conditions,disent_parts[ind_dq],b),
                                                          bounds=[(0,np.pi)]*(4),tol=1e-14)
                                elif cf_type[2]=='merged':
                                    Osub=sc.optimize.minimize(CFMult,p_to_opt,
                                                              args=(cf_type[0:2],p_opt,ind,conditions,[disent_parts[ind_dq],finalrot_parts],b),
                                                              bounds=[(0,np.pi)]*(4),tol=1e-14)
                                else:
                                    print("Error")
                                    
                                
                            
                            if s>int(Nopt/2) and abs(Osub.fun-Oresult[0])<tol:
                                Oresult=[Osub.fun,Osub.x]
                                print("Exit tolerancia, FCoste: \n\r", Oresult[0])
                                break
                            elif abs(Osub.fun-Oresult[0])<tol:
                                print("Randomizamos iterante inicial layer \n\r")
                                p_to_opt=np.random.rand(len(getPD(p_opt,ind)))
                            elif Osub.fun<Oresult[0]:
                                Oresult=[Osub.fun,Osub.x]
                                p_to_opt=Osub.x
                                print("Actualizamos iterante al anterior \n\r", Oresult[0])
                            else:
                                print("Randomizamos iterante inicial layer \n\r")
                                p_to_opt=np.random.rand(len(getPD(p_opt,ind)))
                            
                                
                                
                            
                    UpPD(Oresult[1],p_opt,ind,overwrite=True)
                    if displays==True:
                        print("FCoste Disentangle Q%dC%dL%d \n\r" % (dq,i,j) ,Oresult[0])
                        print("--------------------------------------------- \n\r")
                    ind_j+=1

        ind='RF'
        unziped_parameters=unpack(getPD(p_opt,ind))
        if displays==True:
            print("Capa de Rotaciones Finales\n\r")
            
        if cf_type[2]=='separated':
            OFRLayer=sc.optimize.minimize(CF,unziped_parameters,
                                  args=(cf_type[1],p_opt,ind,conditions,finalrot_parts),
                                  bounds=[(0,np.pi)]*3*n,tol=1e-14)
        elif cf_type[2]=='merged':
            OFRLayer=sc.optimize.minimize(CFMult,unziped_parameters,
                              args=(cf_type[0:2],p_opt,ind,conditions,[[True,True,False],finalrot_parts]),
                              bounds=[(0,np.pi)]*3*n,tol=1e-14)
        else:
            print("Error")
        


        OFRresult=[OFRLayer.fun,OFRLayer.x]
        if opt_type[1]==True:
            
            for s in range(Nopt):
                print("Minimizar Iteracion, intento %d de optimizar RF" % s)
                if s==0:
                    if cf_type[2]=='separated':
                        Osub=sc.optimize.minimize(CF,unziped_parameters,
                                                  args=(cf_type[1],p_opt,ind,conditions,finalrot_parts),
                                                  bounds=[(0,np.pi)]*3*n,tol=1e-14)
                    elif cf_type[2]=='merged':
                        Osub=sc.optimize.minimize(CFMult,unziped_parameters,
                                                  args=(cf_type[0:2],p_opt,ind,conditions,[[True,True,False],finalrot_parts]),
                                                  bounds=[(0,np.pi)]*3*n,tol=1e-14)
                    else:
                        print("Error")
                else:
                    if cf_type[2]=='separated':
                        Osub=sc.optimize.minimize(CF,unziped_parameters,
                                              args=(cf_type[1],p_opt,ind,conditions,finalrot_parts),
                                              bounds=[(0,np.pi)]*3*n,tol=1e-14)
                    elif cf_type[2]=='merged':
                        Osub=sc.optimize.minimize(CFMult,unziped_parameters,
                                                  args=(cf_type[0:2],p_opt,ind,conditions,[[True,True,False],finalrot_parts]),
                                                  bounds=[(0,np.pi)]*3*n,tol=1e-14)
                    else:
                        print("Error")
#
                    

                if s>int(Nopt/2) and abs(Osub.fun-Oresult[0])<tol:
                    OFRresult=[Osub.fun,Osub.x]
                    print("Exit tolerancia, FCoste: \n\r", OFRresult[0])
                    break
                elif abs(Osub.fun-Oresult[0])<tol:
                    print("Randomizamos iterante inicial layer \n\r")
                    p_to_opt=np.random.rand(len(getPD(p_opt,ind)))
                elif Osub.fun<Oresult[0]:
                    OFRresult=[Osub.fun,Osub.x]
                    p_to_opt=Osub.x
                    print("Actualizamos iterante al anterior \n\r", OFRresult[0])
                else:
                    print("Randomizamos iterante inicial layer \n\r")
                    p_to_opt=np.random.rand(len(unpack(getPD(p_opt,ind))))
            
        UpPD(OFRresult[1],p_opt,ind,overwrite=True)
        if displays==True:
            print("Optimization RF result \n\r",OFRresult[0])
            print("--------------------------------------------- \n\r")
        
        print("---------------------------------------------")
        print("---------------------------------------------")
        print("FCoste Total Vuelta %d \n\r" % num_it,OFRresult[0])
        print("---------------------------------------------")
        print("--------------------------------------------- \n\r")
        
        if cf_type[2]=='separated':
            Resultados_Intento.append([Oresult,OFRresult])
            
            if num_it==int(Nmax/3) and Resultados_Intento[num_it][0][0]+Resultados_Intento[num_it][1][0]>Coste_exit:
                print("Intento dado por perdido, coste demasiado alto")
                return  [Resultados_Intento[-1][0][0]+Resultados_Intento[-1][1][0],p_opt]
            
            Test = abs(Resultados_Intento[num_it][0][0]+Resultados_Intento[num_it][1][0]-Resultados_Intento[num_it-1][0][0]-Resultados_Intento[num_it-1][1][0])
        elif cf_type[2]=='merged':
            Resultados_Intento.append(OFRresult)
            
            if num_it==int(Nmax/3) and Resultados_Intento[num_it][0]>Coste_exit:
                print("Intento dado por perdido, coste demasiado alto")
                return  [Resultados_Intento[-1][0],p_opt]
            
            Test = abs(Resultados_Intento[num_it][0]-Resultados_Intento[num_it-1][0])
        else:
            print("Error")
        
        
        
        num_it+=1
        
        if Test<tol:
            print("Break_tolerancia")
            break

    if cf_type[2]=='separated':
        to_return = [Resultados_Intento[-1][0][0]+Resultados_Intento[-1][1][0],p_opt]
    elif cf_type[2]=='merged':
        to_return = [Resultados_Intento[-1][0],p_opt]
    else:
        print("Error")
    
    return to_return
