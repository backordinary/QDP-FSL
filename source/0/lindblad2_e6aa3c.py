# https://github.com/SanderStammbach/Sanders-Studium/blob/1957c67c64dd38c889754d40f86195edee06391b/Master-Arbeit/Lindblad2.py
# /bin/env/python
from calendar import c
from ctypes import c_char_p
from email import message_from_file
import imp
from tkinter import messagebox
from turtle import color, title
from typing import List
from IPython.display import display
from re import A, U
from sys import displayhook
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, Aer, transpile, assemble
from qiskit.visualization import plot_histogram
from math import gcd
from numpy.random import randint
import pandas as pd
from fractions import Fraction
import qutip
print("conda activate")
from qutip import mesolve as mesolve
from qutip import basis as basis
print("import succesful")
from qutip import tensor as tensor
from qutip import dag as dag
from qutip import steadystate as steadystate
from qutip import *
from qutip import ptrace 
from Loup_for_different_coupling import Diverse_Loups as Diverse_Loups
import multiprocessing as mp
import csv
from IPython.display import display, Latex
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})
#Konstante Grössen
########################################################################################################
omega_1=0
omega_2=30
omega_3=150

omega_f= omega_2 - omega_1
omega_h=omega_3-omega_1  # frequency of the atomic transition that is coupled to the hot bath
omega_c=omega_3-omega_2

h=1
nph=30    # Maximale Photonen im cavity 
 
Th=100.    # temperature of the hot bath
Tc=20.     # temperature of the cold bath
Tenv=0.0000000000000000000000000001



nh=2.6
nc=0.02

nf=0.02    #Beschreibt den cavity/Photonen. 



gamma_h=1
gamma_c=1
kappa=0.028
kb=1
g=14*kappa
 

b_fock=qutip.states.fock(nph,0) #m)/fock(N,#m)
b_atom=basis(3)
b_comp=tensor( b_atom, b_fock)

#rho2=tensor(basis(6,4),basis(6,4).dag())


# hier ist ein wenig gebastel mit den transitionsoperatoren

va=qutip.Qobj(qutip.qutrit_basis()[2])
vb=qutip.Qobj(qutip.qutrit_basis()[1])
vg=qutip.Qobj(qutip.qutrit_basis()[0])

Trans_13=tensor(vg*va.dag(),qutip.identity(nph))
Trans_23=tensor(vb*va.dag(),qutip.identity(nph))
Trans_12=tensor(vg*vb.dag(),qutip.identity(nph))

proj_1=tensor(vg*vg.dag(),qutip.identity(nph))
proj_2=tensor(vb*vb.dag(),qutip.identity(nph))
proj_3=tensor(va*va.dag(),qutip.identity(nph))

a=qutip.tensor(qutip.identity(3),qutip.destroy(nph))


################################################################
#implementierung von dem Hamilton
H_free=omega_1*proj_1+h*omega_2*proj_2+h*omega_3*proj_3+h*omega_f*a.dag()*a

H_int=h*g*(Trans_12*a.dag()+a*Trans_12.dag())

H=H_free+H_int

print(H-H.dag(),H_int-H_int.dag(),H_free-H_free.dag(),"sollte null geben!!!!!!!!!!!!!!!!!!!!!!!!")

#print(H_int,H_free,H)31
#H=Hfree+Hint
#########################################################################################################



def n(omega,T):
    n=1/(np.exp(h*omega/(kb*T))-1)
    return n
"""
gamma_1=(n(omega_h,Th)+1)*gamma_h #### unsicher wegen vorfaktor 1/2 
gamma_2=(n(omega_h,Th))*gamma_h
gamma_3=(n(omega_c,Tc)+1)*gamma_c
gamma_4=(n(omega_c,Tc))*gamma_c
kappa_5=(n(omega_f,Tenv)+1)*kappa####goes to zero
kappa_6=(n(omega_f,Tenv))*kappa

"""
gamma_1=(nh+1)*gamma_h #### unsicher wegen vorfaktor 1/2 
gamma_2=(nh)*gamma_h
gamma_3=(nc+1)*gamma_c
gamma_4=(nc)*gamma_c
kappa_5=(nf+1)*kappa ####goes to zero
kappa_6=(nf)*kappa
print(gamma_1)

######################################################################################################
#Vorfaktoren rechenr
def T(omega,n):
    T=h*omega/(kb*(np.log((1/n)+1)))
    return T

print("Die Temperatur des warmen Bades ist: ",T(omega_h,nh))
######################################################################################################


A1=Trans_13
A2=Trans_13.dag()
A3=Trans_23
A4=Trans_23.dag()
A5=a
A6=a.dag()
########################################################################################################
c_op_list=[]

c_op_list.append(np.sqrt(gamma_1)*A1)
c_op_list.append(np.sqrt(gamma_2)*A2)
c_op_list.append(np.sqrt(gamma_3)*A3)
c_op_list.append(np.sqrt(gamma_4)*A4)
c_op_list.append(np.sqrt(kappa_5)*A5)
c_op_list.append(np.sqrt(kappa_6)*A6)


#print(c_op_list)

rho = steadystate(H, c_op_list)
#print(rho)
#qutip.plot_wigner_fock_distribution(rho)
#plt.show()



rho_f=rho.ptrace(1)  ### State in the cavity
print(rho_f)
print("Die Temperatur des warmen Bades ist: ",T(omega_h,nh))
print("Die Temperatur des kalten Bades ist: ",T(omega_c,nc))
qutip.plot_wigner_fock_distribution(rho_f,colorbar='colorbar')
plt.show()
##########################################################################################################

#qutip.plot_wigner_fock_distribution(rho2,colorbar='colorbar')
#plt.show()


##########################################################################################################
#Berechnen der Wärme als Tr(H_free*rho) oder Tr[H_free*L(rho,A)] D is one of the Liovillien therme (L)! 
def D(c_op_list,rho):
    D=[]
    D.append(c_op_list[0]*rho*c_op_list[0].dag()-1/2*(c_op_list[0].dag()*c_op_list[0]*rho-rho*c_op_list[0].dag()*c_op_list[0]))
    D.append(c_op_list[1]*rho*c_op_list[1].dag()-1/2*(c_op_list[1].dag()*c_op_list[1]*rho-rho*c_op_list[1].dag()*c_op_list[1]))
    D.append(c_op_list[2]*rho*c_op_list[2].dag()-1/2*(c_op_list[2].dag()*c_op_list[2]*rho-rho*c_op_list[2].dag()*c_op_list[2]))
    D.append(c_op_list[3]*rho*c_op_list[3].dag()-1/2*(c_op_list[3].dag()*c_op_list[3]*rho-rho*c_op_list[3].dag()*c_op_list[3]))
    D.append(c_op_list[4]*rho*c_op_list[4].dag()-1/2*(c_op_list[4].dag()*c_op_list[4]*rho-rho*c_op_list[4].dag()*c_op_list[4]))
    D.append(c_op_list[5]*rho*c_op_list[5].dag()-1/2*(c_op_list[5].dag()*c_op_list[5]*rho-rho*c_op_list[5].dag()*c_op_list[5]))
    return D


    
Liste_von_Q=[]

Liste_von_Q.append(np.trace(H_free*(D(c_op_list,rho)[0]+D(c_op_list,rho)[1])))
Liste_von_Q.append(np.trace(H_free*(D(c_op_list,rho)[2]+D(c_op_list,rho)[3])))
Liste_von_Q.append(np.trace(H_free*(D(c_op_list,rho)[4]+D(c_op_list,rho)[5])))
print(Liste_von_Q)


#loat_list= list(np.float_(Liste_von_Q))
#print(float_list)    
#Liste_von_Q=float_list


g_list=[]

Energie_VS_g=[]
for i in range(200):
    list_temp=[]
    list_temp=Diverse_Loups.EnergieCalculator(i/100, H_free,Trans_12,Trans_13, Trans_23,a,nh,nf,nc,h,kb,gamma_h,gamma_c,kappa,c_op_list)
    g_list.append(i/100)  #Erstellt eine Liste mit Wären von g 
    Energie_VS_g.append(list_temp)

#Liste von Stings in floats konvertieren
#float_list2=list(np.float_(Energie_VS_g))
print(Energie_VS_g)  

#Speicherern der Liste in csv datei
with open('Speicherort.csv','w') as temp_file:
    for item in Energie_VS_g:
        temp_file.write("%s\n" % item)

################################################################################################################################################
#Plotten Wärme VS g 

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.set_xlabel(r' $\frac{g}{\gamma_h}$', fontsize=23)
ax.set_ylabel(r' Heat current', fontsize=15)
plt.title('current/energyflux vs coupling constant')
plt.plot(np.asarray(g_list)[:200],np.asarray(Energie_VS_g)[:200,0],label=r' $\frac{J_h}{\gamma_h \omega_h}$')
plt.plot(np.asarray(g_list)[:200],np.asarray(Energie_VS_g)[:200,1],label=r' $\frac{J_c}{\gamma_c \omega_c}$')
plt.plot(np.asarray(g_list)[:200],np.asarray(Energie_VS_g)[:200,2],label=r' $\frac{J_{cav}}{\gamma_{cav} \omega_{cav}}$')
legend = ax.legend(loc='upper right', shadow=True, fontsize='x-large')
legend.get_frame().set_facecolor('C0')
plt.show()
""" r' $\frac{-e^{i\pi}}{2^n}$!',fontsize=21    """#für LATEX Labels r' $\frac{g}{\gamma}$', fontsize=19

######################################################################################################################################################
#Berechnung
nh_list=[]
Trace_list=[]
nh=0.1 #set nh again to zero
for j in range(100):
    Trace_list_temp=Diverse_Loups.Funktion(nh,proj_1,proj_2,proj_3,H,nc,nf,gamma_h,gamma_c,kappa,A1,A2,A3,A4,A5,A6)
    Trace_list.append(Trace_list_temp)

    nh_list.append(nh)
    nh=nh+0.3

fig2, ax = plt.subplots()
ax.set_xlabel(r' $n_h$', fontsize=21)
ax.set_ylabel('probability')
plt.title('stationary atomic population')
plt.plot(np.asarray(nh_list)[:100],np.asarray(Trace_list)[:100,0],label='P1')
plt.plot(np.asarray(nh_list)[:100],np.asarray(Trace_list)[:100,1],label='P2')
plt.plot(np.asarray(nh_list)[:100],np.asarray(Trace_list)[:100,2],label='P3')
legend = ax.legend(loc='center right', shadow=True, fontsize='x-large')
legend.get_frame().set_facecolor('C0')
#Linien in plt
plt.axvline(x=2.6)
plt.axvline(x=2.6)
plt.axvline(x=5.5)
plt.axvline(x=0.17)
plt.axvline(x=20)
plt.axvline(x=1.7)

plt.show()
#with open("Speicherort.csv", "wb") as f:
#    writer = csv.writer(f)
#    writer.writerows(Energie_VS_g)

######################################################################################################################################################################
#random testing
# Muss ich noch tensoriesieren mit 30 also psi0
#H = 2*np.pi * 0.1 * qutip.sigmax()
psi0 = tensor(basis(3, 1),basis(30))
times = np.linspace(0.0, 10.0, 100)

#result = qutip.sesolve(H, psi0, times, [qutip.sigmaz(), qutip.sigmay()])
result = qutip.sesolve(H, psi0, times,c_op_list)
fig, ax = plt.subplots()
ax.plot(result.times, result.expect[0])
ax.plot(result.times, result.expect[1])
ax.set_xlabel('Time')
ax.set_ylabel('Expectation values')
ax.legend(("Sigma-Z", "Sigma-Y"))
plt.show()


#Entropy. production


nh2=0.1
nh_list2=[]
Entropy=[]
for i in range(100):
    list_temp=[]
    list_temp=Diverse_Loups.Entropy(nh2,Trans_12,a, kb,h,g,H,H_free,nc,nf,gamma_h,gamma_c,kappa,Trans_13,Trans_23,omega_c,omega_h,omega_f)
    #g_list.append(i/100)  #Erstellt eine Liste mit Wären von g 
    Entropy.append(list_temp)
    nh2=nh2+0.3
    nh_list2.append(nh2)

#Liste von Stings in floats konvertieren
#float_list2=list(np.float_(Energie_VS_g))
print(Entropy) 

#result=mesolve(H, rho0, tlist)
#print(D(c_op_list,rho)[3])


print("Die Temperatur des warmen Bades ist: ",T(omega_h,nh))
print("Die Temperatur des kalten Bades ist: ",T(omega_c,nc))
print(Trace_list_temp)

fig3, ax = plt.subplots()
ax.set_xlabel(r' $n_h$', fontsize=19)
ax.set_ylabel('Entropy')
plt.title('Entropy Production')
plt.plot(np.asarray(nh_list2)[:100],np.asarray(Entropy)[:100,0],label=r' $J_h+J_{cav}+J_c$',color='red')
plt.plot(np.asarray(nh_list2)[:100],np.asarray(Entropy)[:100,1],label=r' $J_h$')
plt.plot(np.asarray(nh_list2)[:100],np.asarray(Entropy)[:100,2],label=r' $J_c$')
plt.plot(np.asarray(nh_list2)[:100],np.asarray(Entropy)[:100,3],label=r' $J_{cav}$')
legend = ax.legend(loc='upper right', shadow=True, fontsize='x-large')
legend.get_frame().set_facecolor('C0')
#Linien in plt
plt.axvline(x=2.6)
plt.axvline(x=2.6)
plt.axvline(x=5.5)
plt.axvline(x=0.17)
plt.axvline(x=20)
plt.axvline(x=1.7)

plt.show()
#################################################################################################################
#photon number
nh2=0.1
nh_list2=[]
Photonnumber_list=[]
for i in range(100):
    list_temp=[]
    list_temp=Diverse_Loups.Photonnumber(nh2,a,proj_1,proj_2,proj_3,H,nc,nf,gamma_h,gamma_c,kappa,A1,A2,A3,A4,A5,A6)
    #g_list.append(i/100)  #Erstellt eine Liste mit Wären von g 
    Photonnumber_list.append(list_temp)
    nh2=nh2+0.3
    nh_list2.append(nh2)

    print(Photonnumber_list)

fig4, ax = plt.subplots()
ax.set_xlabel(r' $n_h$', fontsize=21)
ax.set_ylabel(r' $\langle n \rangle$', fontsize=21)
plt.title(r' Photonnumber vs $n_h$',fontsize=21)
plt.plot(np.asarray(nh_list2)[:100],np.asarray(Photonnumber_list)[:100],color='red')

#legend = ax.legend(loc='upper right', shadow=True, fontsize='x-large')
#legend.get_frame().set_facecolor('C0')
#Linien in plt
plt.axvline(x=2.6)
plt.axvline(x=2.6)
plt.axvline(x=5.5)
plt.axvline(x=0.17)
plt.axvline(x=20)
plt.axvline(x=1.7)
plt.show()


