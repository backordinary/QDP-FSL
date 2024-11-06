# https://github.com/Alfaxad/Qhack2022_OPen_Challenge/blob/b76d6942e60882ac38345e1fc3dda0430f8d816c/Qhack%20Open%20Challenge/backend.py
#!/usr/bin/env python
# coding: utf-8

# In[32]:


from azure.quantum import Workspace
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
from qiskit.visualization import plot_histogram
from qiskit.tools.monitor import job_monitor
from azure.quantum.qiskit import AzureQuantumProvider
from qiskit import assemble
from qiskit import BasicAer


provider = AzureQuantumProvider (
    resource_id = "/subscriptions/b1d7f7f8-743f-458e-b3a0-3e09734d716d/resourceGroups/aq-hackathons/providers/Microsoft.Quantum/Workspaces/aq-hackathon-01",
    location = "eastus"
)
workspace = Workspace (
    subscription_id = "b1d7f7f8-743f-458e-b3a0-3e09734d716d",
    resource_group = "aq-hackathons",
    name = "aq-hackathon-01",
    location = "eastus"
)

### tic toc toe ###

######################################
#           #             #          #   
#     6     #     7       #    8     # 
#           #             #          # 
######################################
#           #             #          # 
#     3     #     4       #    5     # 
#           #             #          # 
######################################
#           #             #          #  
#     0     #     1       #     2    #   
#           #             #          # 
######################################

print('######################################\n#           #             #          #\n#     6     #     7       #    8     #\n#           #             #          #\n######################################\n#           #             #          #\n#     3     #     4       #    5     #\n#           #             #          #\n######################################\n#           #             #          #\n#     0     #     1       #     2    #\n#           #             #          #\n######################################\n')

######
#This variables can be inputed by the users to change the input of the game!
l_player_1 = [0] #list of qbits cell numbers inicialized to 0
l_player_2 = [2,3] #list of qbits cell numbers inicialized to 1
l_entangled = [(1,6),(7,8),(4,5)] #list of pairs of cells that need to be entangled to |01>+|10>

#translate from lists to initial state matrix
l_initial_ordered = [0,0,0,0,0,0,0,0,0]
for inx in range(len(l_player_1)):
    l_initial_ordered[l_player_1[inx]]=0
for inx in range(len(l_player_2)):
    l_initial_ordered[l_player_2[inx]]=1  
entg_num = 1  
for inx in range(len(l_entangled)):
    l_initial_ordered[l_entangled[inx][0]]='e'+str(entg_num )   
    l_initial_ordered[l_entangled[inx][1]]='e'+str(entg_num  )
    entg_num +=1

print('This the initial table state inputed by the user:\n0 corresponds to player 1, 1 corresponds to player 2,\n e1,e2,e3... corresponds to entangled tiles')
print('-----\n'+str(l_initial_ordered[6])+'|'+str(l_initial_ordered[7])+'|'+str(l_initial_ordered[8]))
print('-----\n'+str(l_initial_ordered[3])+'|'+str(l_initial_ordered[4])+'|'+str(l_initial_ordered[5]))
print('-----\n'+str(l_initial_ordered[0])+'|'+str(l_initial_ordered[1])+'|'+str(l_initial_ordered[2]))
print('-----')


# Create a Quantum Circuit acting on the q register
circuit = QuantumCircuit(9, 9)
circuit.name = "Tic toc toe"

for inx  in range(len(l_player_2)):
    circuit.x(l_player_2[inx])

for inx in range(len(l_entangled)):
    circuit.h(l_entangled[inx][0])
    circuit.x(l_entangled[inx][1])
    circuit.cx(l_entangled[inx][0],l_entangled[inx][1])




#circuit.h(0)
#circuit.cx(0, 1)
#circuit.cx(1, 2)
circuit.measure(list(range(9)), list(range(9)))

# Print out the circuit
print('This is the correspondin quantum circuit from the users input')
circuit.draw()



# In[8]:


#This is only used to simulate with real hardware
#simulator_backend = provider.get_backend("ionq.simulator")
#simulator_backend = provider.get_backend("ionq.qpu")
#job = simulator_backend.run(circuit, shots=1)
#job_id = job.id()
#print("quantum tic", job_id)
#job_monitor(job)
#result = job.result()
#print(result.get_counts())


# In[33]:


def check_winner(board,mark):
    return(((board[0]==mark) and (board[1]== mark) and (board[2]==mark) )or #for row1 

            ((board[3]==mark) and (board[4]==mark) and (board[5]==mark) )or #for row2

            ((board[6]==mark) and (board[7]==mark) and (board[8]==mark) )or #for row3

            ((board[0]==mark) and (board[3]==mark) and (board[6]== mark) )or#for Colm1 

            ((board[1]==mark) and (board[4]==mark) and (board[7]==mark) )or #for Colm 2

            ((board[2]==mark) and (board[5]==mark) and (board[8]==mark) )or #for colm 3

            ((board[0]==mark) and (board[4]==mark) and (board[8]==mark) )or #daignole 1

            ((board[2]==mark) and (board[4]==mark) and (board[6]==mark) )) #daignole 2


flag_p1=1 
flag_p2=0

while (flag_p1 or flag_p2) and not(flag_p1 and flag_p2):
    print('Running the quantum circuit...')
    flag_p1=0 
    flag_p2=0
    job = execute(circuit, BasicAer.get_backend('qasm_simulator'), shots=1)
    result = job.result()
    l_final_ordered=list(map(lambda x: int(x),list(list(result.get_counts().keys())[0][::-1])))
    print(l_final_ordered)
    #list with ordered cells
    print('The colapsed state is:')
    print('-----\n'+str(l_final_ordered[6])+'|'+str(l_final_ordered[7])+'|'+str(l_final_ordered[8]))
    print('-----\n'+str(l_final_ordered[3])+'|'+str(l_final_ordered[4])+'|'+str(l_final_ordered[5]))
    print('-----\n'+str(l_final_ordered[0])+'|'+str(l_final_ordered[1])+'|'+str(l_final_ordered[2]))
    print('-----')

    if (check_winner(l_final_ordered,0) ):## to check if player 1 won
        print('Player 1 won!')
        flag_p1 = 1
    else:
        flag_p1 = 0

    if (check_winner(l_final_ordered,1)): ## to check if player 2 won
        print('Player 2 won!')
        flag_p2 = 1
    else:
        flag_p2 = 0

    if (flag_p1 or flag_p2) and not(flag_p1 and flag_p2):
        break

    if flag_p1 and flag_p2:
        print('The game will repeat until one one player wins')

    if not(flag_p1 and flag_p2):
        print('No winners,\nThe game will repeat until one one player wins')

