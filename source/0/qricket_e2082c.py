# https://github.com/aarav-rm8/QricKet/blob/d42b10b317d6c0240309aceb13127b73a25b64d5/QricKet.py
#!/usr/bin/env python
# coding: utf-8

# In[24]:


#Importing Required Libraries
from qiskit import *
from qiskit.tools.visualization import *
import numpy as np
from qiskit.circuit.library import *
import pandas as pd


# In[25]:


#Importing Data from CSV File
global df
df = pd.read_csv('playerdata.csv')


# In[26]:


#Defining function which takes in Batsman and Bowler Data, and hence returns the number of runs scored by the particular player.
#We have collected data and used that to calculate the probabilities for each player against each type of bowler.
#We use the probabilities to create a statevector. We then build a quantum circuit in which we initialise 3 qubits as per the statevector.
#We then perform a measurement, which would be random. The measurement determines the number of runs / wicket taken by player on that particular ball.
def ball(Bowl,Bat):
    #We import the data corresponding to specific player and specific bowler type.
    p_dat=df.loc[(df['PlayerCode']==Bat) & (df['BowlerCode']==Bowl)]
    p_d=list(p_dat.to_records()[0])
    
    #Using data to write our statevector:
    SV = [(x)**0.5 for x in p_d[-5:-1] + [p_d[-1]]] + [0,0,0]
    #err = 1 - sum(x**2 for x in SV)
    #print('Error in StateVec:',err)
    #SV[0] = (SV[0]**2+ err)**.5 #To ensure probability always adds up to 1 and hence reduce possibility of error
   # print('New Err:',1 - sum(x**2 for x in SV))
    #Defining a quantum circuit based on above data:
    qr = QuantumRegister(3)
    cr = ClassicalRegister(3)
    qc = QuantumCircuit(qr,cr)
    
    qc.initialize(SV)
    qc.measure(qr,cr)
    
    #Obtaining a single measurement:
    simulator=Aer.get_backend('qasm_simulator')

    RES=execute(qc,backend=simulator,shots=1).result()
    
    Output=list(RES.get_counts(qc).keys())[0]
    
    #Obtaining number of runs / wicket from the result
    wick = 0 #indicated it is not out.
    if Output=='000':
        runs=0
        wick=1
    elif Output=='001':
        runs=0
    elif Output=='010':
        runs=1
    elif Output=='011':
        runs=4
    elif Output=='100':
        runs=6
        
    return [runs,wick]


# In[27]:


#Defining a function which allows us to choose a bowler type based on the following stats:
#Fast Bowler has 96% chance of being chosen in the overs 1-3 and 19-20
#Spin Bowler has 95% chance of being chosen in the overs 9-14
#In all other bowlers, each bowler type is equally likely

def Bowler_Choose(over):
    
     #Defining a 3 qubit quantum circuit:
    qr = QuantumRegister(3)
    cr = ClassicalRegister(3)
    qc = QuantumCircuit(qr,cr)

    
    if (over in range(1,4)) or (over in range(19,21)):
        #QRNG for FAST BOWLER
        s= (0.04/3)**.5
        f = (0.96/5)**.5
        stv = [s,f,f,f,f,s,f,s]
        qc.initialize(stv)
    elif over in range(9,15):
        s= (0.95/3)**.5
        f = (0.05/5)**.5
        stv = [s,f,f,f,f,s,f,s]
        qc.initialize(stv)
    else:
        #Equally Likely
        qc.h(0)
        qc.h(1)
        qc.h(2)
    qc.measure(qr,cr)
    
    #Obtaining a single measurement:
    simulator=Aer.get_backend('qasm_simulator')

    RES=execute(qc,backend=simulator,shots=1).result()
    
    Output=int(list(RES.get_counts(qc).keys())[0],2)+1
    
    return Output
    
        


# In[28]:


B_Names={'VK':'Virat Kohli',
        'RS':'Rohit Sharma',
        'RP':'Rishabh Pant',
        'SKY':'Suryakumar Yadav',
        'KL':'KL Rahul',
        'HP':'Hardik Pandya',
        'AP':'Axar Patel',
        'RA':'Ravichandran Ashwin',
        'BK':'Bhuvaneshwar Kumar',
        'MS':'Mohammed Shami',
        'UY':'Umesh Yadav'}
BowlerType=[None,'Left-arm Chinaman',
'Left-arm Fast',
'Left-arm Medium',
'Left-arm Orthodox',
'Right-arm Fast',
'Right-arm Legbreak',
'Right-arm Medium',
'Right-arm Offbreak']

Indiv_Runs = {}
for i in B_Names:
    Indiv_Runs[i]=0


# In[35]:


#USING A PRE-DEFINED BATTING ORDER:

B_Order = ['RS','KL','VK','SKY','RP','HP','AP','RA','BK','MS','UY'] #ASSUMED
print('Batting Order:')
for i in range(11):
    print(i+1,':',B_Names[B_Order[i]])
    
Indiv_Runs = {}
for i in B_Order:
    Indiv_Runs[i]=0

Overs=20 #ASSUMED
Runs_Net = 0
Wickets_Net = 0
B1 = B_Order[0]
B2 = B_Order[1]
del(B_Order[1])
del(B_Order[0])

match_over=False
for i in range(1,Overs+1):
    print('')
    print('Over #',i)
    Bowler = Bowler_Choose(i) 

    for j in range(6):
        print('')
        print('Ball #',j+1)
        print('On Strike:',B_Names[B1])
        print('Bowler Type:',BowlerType[Bowler])
        Result=ball(Bowler,B1)
        if Result[1]==1:
            print('WICKET!')
            Wickets_Net+=1
            try:
                B1=B_Order[0]
                del(B_Order[0])
                print('Coming in:',B_Names[B1])
            except IndexError:
                match_over=True
            
        if Result[0]==1:
            B1,B2=B2,B1
        
        print('Number of Runs Scored =',Result[0])
        Runs_Net+=Result[0]
        Indiv_Runs[B1] = Indiv_Runs[B1]+Result[0]
        print('Current Score:',Runs_Net,'/',Wickets_Net,)
        
        if match_over:
            break
        
    if match_over:
        print('All Out!')
        break
    print('')
    B1,B2=B2,B1
print('Match Innings Over!')


# In[36]:

print('\nRuns Scored by Individual Players')

for k in Indiv_Runs:
    print(B_Names[k],':',Indiv_Runs[k])
    

# In[ ]:




