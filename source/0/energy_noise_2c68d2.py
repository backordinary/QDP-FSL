# https://github.com/psasanka1729/Grover-NISQ/blob/a410b3273db970396ab0b9ac31cd0cd5b0b1eec4/Energy%20vs%20epsilon%20with%20colormap%20of%20entropy/Plots/L%20=%206/energy_noise.py
#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
from qiskit import*
from qiskit import Aer
import qiskit.quantum_info as qi
import numpy as np
import re
from scipy.sparse import identity
import numpy as np
from scipy import sparse
from scipy.sparse import lil_matrix
import scipy.sparse.linalg


# In[2]:


Target_state = '000000'


# In[3]:


N = len(Target_state)


## The operator U_s.
A = np.ones((2**N, 2**N))
U_s = (2/(2**N))*A - np.identity(2**N, dtype = complex)


## The operator U_w. This is neeed for the sign adjustment of Grover_reconstructed operator.
U_w = - np.identity(2 ** N, dtype=complex) 
Target_index = int(Target_state, 2)
U_w.itemset((Target_index, Target_index),1)


## G is the Grover operator.
#G = np.matmul(U_w, U_s)


# In[4]:


l = []

file1 = open('gates_list_'+Target_state+'.txt', 'r')
Lines = file1.readlines()
 

for line in Lines:
    l.append(line.strip())

gates_list = []

Rz_Number = 0
for i in range(len(l)):
    
    l_temp = []
    gate_name = l[i].split(',')[0]
    if gate_name == 'rz':
        Rz_Number +=1 
    elif gate_name == 'h':
        Rz_Number += 1
    gate_angle = l[i].split(',')[1]
    gate_qubit = l[i].split(',')[2]
    
    l_temp.append(gate_name)
    l_temp.append(gate_angle)
    l_temp.append(gate_qubit)

    gates_list.append(l_temp)


# In[5]:


Rz_Number


# In[6]:


#N = 3
def Hadamard_gate(): # Hadamad gate acting on one qubit.
    
    return 1/np.sqrt(2)*np.array([[1,1],[1,-1]])

def RY(theta):
    return np.array([[np.cos(theta/2), -np.sin(theta/2)], [np.sin(theta/2), np.cos(theta/2)]])

def PauliZ():
    return np.array([[1,0],[0,-1]])

# H = RY(pi/2)*Z

def Hadamard(Angle, Qubit): 

    '''

    List below will hold gates acting on one qubit. For example, for L = 3,
    the Hadamard gate acting on the qubit 1 is given by = 1 x H x 1, where 
    x is the Kronecker product. Then, qubits_list = [1,H,1].

    ''' 

    qubits_list = [] 
    
    for i in range(N):
        
        if i == Qubit: # Qubit^th position in the list is H.
            
            qubits_list.append(np.matmul(RY(Angle),PauliZ()))
            
        else: # Other gates are identity operators.
            
            qubits_list.append(np.identity(2))

    '''
    
    The following loop performs the Kronecker product.

    '''        
    
    M = sparse.csr_matrix(qubits_list[0]) # Initializes the final matrix.
    
    for g in range(1,len(qubits_list)):
        
        M = sparse.kron(qubits_list[g],M) # kronecker product.
        
    return M


# In[7]:


def CNOT(t,c):
    ## Changing the simulator 
    backend = Aer.get_backend('unitary_simulator')

    ## The circuit without measurement
    circ = QuantumCircuit(N)
    circ.cx(t,c)

    ## job execution and getting the result as an object
    job = execute(circ, backend)
    result = job.result()

    ## get the unitary matrix from the result object
    return result.get_unitary(circ) 


# In[8]:


def Rz_matrix(theta):

    return np.matrix([[np.exp(-1j*theta/2),0],[0,np.exp(1j*theta/2)]])

def Rz(Angle, Qubit):
    
    if Qubit > N -1 :
        
        print("Qubit number exceeds N")
        
    else:    
    
        qubits_list = []
    
        for i in range(N):
        
            if i == Qubit:
            
                qubits_list.append(Rz_matrix(Angle))
            
            else:
            
                qubits_list.append(np.matrix(np.identity(2)))
    
        M = sparse.csr_matrix(qubits_list[0])
    
        for g in range(1,len(qubits_list)):
        
            M = sparse.kron(qubits_list[g], M) # kronecker product.
        
        return M


# In[9]:


def Grover_reconstructed1(epsilon):
    

    #Rz_Noise = 2*(np.random.rand(Rz_Number)-0.5)
    ## Initializing the oracle U_w as an identity matrix.
    
    Or = identity(2**N) 

    ## In the following loop we multiply all the 1 and 2 qubit gates with (or without) noise.
    
    j = 0 # Index for the random noise list.
    
    for i in range(len(gates_list)): # l is the list with all the gates.
    
        if gates_list[i][0] == 'rz':
            
            Rz_s = sparse.csr_matrix(Rz(float(gates_list[i][1])  +

                 epsilon * Rz_Noise[j], int(gates_list[i][2])))
            
            Or = Or*Rz_s
            
            j = j + 1
        
        
        
        elif gates_list[i][0] == 'h':
            
            
            Hs = sparse.csr_matrix(Hadamard(np.pi/2+epsilon*Rz_Noise[j], int(gates_list[i][2])))
            Or = Or*Hs
            j = j + 1
            
        
        elif gates_list[i][0] == 'cx':
        
            CNOTs = sparse.csr_matrix(CNOT(int(gates_list[i][1]), int(gates_list[i][2])))
            Or = Or*CNOTs
     
    Or = Or.todense()
    ## In the following we will fix the phase of the reconstructed Oracle.
    # First we will make all the elements
    # 1 or -1.
    Or = Or/Or[0,0]
    
    ## The sign of the reconstructed Oracle should be same as that of original U_w.
    if np.sign(Or[0,0]) == np.sign(U_w[0,0]):
        
        pass # If the sign is same, then pass.
    
    else:
        
        Or = -Or # Otherwise change the sign.
    Gr = np.matmul(Or, U_s) ## The Grover operator G = U_w * U_s.
    
    return Gr


# In[10]:


import numpy
import numpy.linalg

sigma_x=numpy.zeros((2,2),dtype=complex)
sigma_y=numpy.zeros((2,2),dtype=complex)
sigma_z=numpy.zeros((2,2),dtype=complex)
sigma_0=numpy.identity(2,dtype=complex)
sigma_x[0,1]=1.
sigma_x[1,0]=1.
sigma_y[0,1]=-1.j
sigma_y[1,0]=1.j
sigma_z[0,0]=1.
sigma_z[1,1]=-1.
sigma_plus=(sigma_x+1.j*sigma_y)/2.
sigma_minus=(sigma_x-1.j*sigma_y)/2.

def adjoint(psi):
    return psi.conjugate().transpose()

def psi_to_rho(psi):
    return numpy.outer(psi,psi.conjugate())

def exp_val(psi, op):
    return numpy.real(numpy.dot(adjoint(psi),op.dot(psi)))

def norm_sq(psi):
    return numpy.real(numpy.dot(adjoint(psi),psi))

def normalize(psi,tol=1e-9):
    ns=norm_sq(psi)**0.5
    if ns < tol:
        raise ValueError
    return psi/ns

def comm(a,b):
    return a.dot(b)-b.dot(a)

def anti_comm(a,b):
    return a.dot(b)+b.dot(a)

def is_herm(M,tol=1e-9):
    if M.shape[0]!=M.shape[1]:
        return False
    diff=M-adjoint(M)
    return max(numpy.abs(diff.flatten())) < tol

def is_unitary(M,tol=1e-9):
    if M.shape[0]!=M.shape[1]:
        return False
    diff=M.dot(adjoint(M))-numpy.identity((M.shape[0]))
    return max(numpy.abs(diff.flatten())) < tol

def eigu(U,tol=1e-9):
    (E_1,V_1)=numpy.linalg.eigh(U+adjoint(U))
    U_1=adjoint(V_1).dot(U).dot(V_1)
    H_1=adjoint(V_1).dot(U+adjoint(U)).dot(V_1)
    non_diag_lst=[]
    j=0
    while j < U_1.shape[0]:
        k=0
        while k < U_1.shape[0]:
            if j!=k and abs(U_1[j,k]) > tol:
                if j not in non_diag_lst:
                    non_diag_lst.append(j)
                if k not in non_diag_lst:
                    non_diag_lst.append(k)
            k+=1
        j+=1
    if len(non_diag_lst) > 0:
        non_diag_lst=numpy.sort(numpy.array(non_diag_lst))
        U_1_cut=U_1[non_diag_lst,:][:,non_diag_lst]
        (E_2_cut,V_2_cut)=numpy.linalg.eigh(1.j*(U_1_cut-adjoint(U_1_cut)))
        V_2=numpy.identity((U.shape[0]),dtype=V_2_cut.dtype)
        for j in range(len(non_diag_lst)):
            V_2[non_diag_lst[j],non_diag_lst]=V_2_cut[j,:]
        V_1=V_1.dot(V_2)
        U_1=adjoint(V_2).dot(U_1).dot(V_2)

    # Sort by phase
    U_1=numpy.diag(U_1)
    inds=numpy.argsort(numpy.imag(numpy.log(U_1)))

    return (U_1[inds],V_1[:,inds]) # = (U_d,V) s.t. U=V*U_d*V^\dagger


# In[11]:


np.random.seed(2022)
#Rz_Noise = np.zeros(Rz_Number)
Rz_Noise = 2*(np.random.rand(Rz_Number)-0.5)


# In[ ]:





# In[12]:


def Phi_F(operator): 
    
    return (1j*np.log(eigu(operator)[0])).real  # eigu(Gr)[0] = exp(-i * phi_F)


# In[ ]:


#Phi_F(Grover_reconstructed1(0.0))


# In[14]:


L = N // 2 # Length of half cut number of qubits.



'''
    The following function takes a wavefunction as input and returns its entropy.

'''

def Entropy(Wavefunction):




    # Converting the list to a numpy matrix.
    Psi = np.matrix(Wavefunction).reshape(len(Wavefunction),1) # Psi column matrix.

    # Normalizing Psi.
    Psi = Psi/np.linalg.norm(Psi)


      
    
    def psi(s):
        return Psi[(2**L)*s:(2**L)*s + 2**L]   
    
      
    '''
        psi(s_p) is a row matrix/vector. psi(s) is a column matrix/vector.      
        Dimension of rhoA is N/2 x N/2. 
        The element <s|rhoA|sp> is given by psi_sp^\dagger * psi_s.
        
    ''' 

    def rhoA(s,s_p): # <s|rho_A|s_p>

        # psi(s_p)^\dagger * psi(s) is the element of (s,s_p) of rho_AB.  
        return psi(s_p).getH() * psi(s)
    
    
    
    def rhoA_Matrix(N):
    
        M = np.zeros((N,N), dtype = complex) # 0 to N-1.
    
        '''
            rho is Hermitian, it is sufficient to calculate the elements above the diagonal.
            The the elements below the diagonal can be replace by the complex cpnjugate of the
            elements above the diagonal.
        '''
        for i in range(N):
            for j in range(N):
            
                if i <= j : # Above the diagonal (i,j) i<j.
                
                    M[i,j] = rhoA(i,j)[0,0]
                
                else: # Below the diagonal (i,j) i>j.
                
                    M[i,j] = np.conjugate(M[j,i])
        return M    
    
    
    '''
        w is the diagonal of the diagonalized matrix rhoA.

    '''
    
    w, v = np.linalg.eig(rhoA_Matrix(N))
    
    w = w.real

    '''
        The following loop calculates S = - sum \lamba_i * log(\lambda_i).

    '''
    
    DL = np.zeros(N) # Creating an array for log w with zeros.
    
    for i in range(len(w)):
    
        if abs(w[i]) < 1.e-8: # log of zero gives nan.
        
            pass # Leave the log(zero) element as zero.
    
        else:
        
            DL[i] = np.log(w[i])
        
    # Entropy = -Tr(rho * log(rho)).        
    return -sum(w*DL)




def Bin2Dec(BinaryNumber): # Converts binary to decimal numbers.
    return int(str(BinaryNumber),2)


def Dec2Bin(DecimalNumber): # Converts decimal to binary numbers.
    return bin(DecimalNumber).replace("0b", "")



List = [i for i in range(2**N)] 


'''
The following function converts all numbers in decimals in the above list 
 from 0 to 2^N -1 to binary.

''' 
def List_Bin(List):
    
    l = []
    
    for i in List:
        
        i_Bin = Dec2Bin(i)
              
        
        '''
        While converting numbers from decimal to binary, for example, 1
         is mapped to 1, to make sure that
        every numbers have N qubits in them, the following loop adds leading 
        zeros to make the
        length of the binary string equal to N. Now, 1 is mapped to 000.....1
         (string of length N).
        
        '''
        
        while len(i_Bin) < N: 
            
            i_Bin = '0'+i_Bin # This loop adds leading zeros.
            
        l.append(i_Bin)
        
    return l





'''
    The following function takes a binary string as input and rolls the qubits by one and
    returns the rolled string.

'''

def Roll_String(Binary_String):
    
    return Binary_String[-1] + Binary_String[:-1]







'''
    The following function takes a wavefunction as input and performs one roll
     on the qubits and
    returns the resultant wavefunction.

'''

def Psi_Roll(Inital_Psi):
    
    
    
    '''
        The following list contains all possible 2^N qubits after one roll 
        is performed on them.
        For example, the first position 0001 is changed to 1000.
    
    '''
    
    # Rolls every string in the list List by one qubit.
    Rl = [Roll_String(i) for i in List_Bin(List)] 

   

    
    ''' 
        The following list contains the previous list but in decimal numbers. For example,
        for N =4, the first position 1 is changed to 8.
        
    
    '''
    
    Rl_d = [Bin2Dec(i) for i in Rl] # Converts the rolled binary string to decimal number.


    '''
        The following loop rearranges the coefficients of Psi after rolling. 
        For example, for N = 4,
        if the first coefficient 0001 is mapped to the eighth coefficient 1000 after
         one rotation of
        the qubits. The coefficient of the rolled Psi in the i ^ th position is in the
         Rl_d[i] ^ th positon
        of the initial Psi.
    
    '''
    
    
    Psi_Rolled = []

    for i in range(2**N): 
        # Rearranging the coefficients according to the list l_d.
        Psi_Rolled.append(Inital_Psi[Rl_d[i]]) 
        
    return Psi_Rolled






'''
    The following function performs specified number of rolls Num on the qubits.

'''

def N_Rolled(Num, Initial_Psi): # Use loop only for postive N.
    
    if Num == 0:
        
        return Initial_Psi
    
    else:
    
        s = Psi_Roll(Initial_Psi) # One roll.
    
        for i in range(Num-1): # Loop performing remaining N-1 rolls.
        
            s = Psi_Roll(s)
        
        return np.matrix(s).reshape(2**N,1) # Returning the rolled wavefunction as a matrix.

def Average_Entropy(Initial_Psi):
    
    list_of_entropies = []
    
    '''
    The loop calculates all the entropies and returns a list containing them.
    
    '''
    for i in range(N):
        
        S = Entropy(N_Rolled(i, Initial_Psi))
        list_of_entropies.append(S)
        
    # Returns the average entropy    
    return sum(list_of_entropies) / len(list_of_entropies)


def V_Matrix(operator):
    return eigu(operator)[1]
    
def Array2List(Arr):
    Arr_l = Arr.tolist()
    l = []
    for i in Arr_l:
        l.append(i[0])
    return l


# In[15]:


np.random.seed(2022)
Rz_Noise = 2*(np.random.rand(Rz_Number)-0.5)


# In[ ]:


f = open('plot_data'+Target_state+'.txt', 'w')
Num = 400

for i in range(1,Num):
    eps = 0.1*(i/(Num))
    print(i)
    
    f = open('plot_data'+Target_state+'.txt', 'a')
    Op = Grover_reconstructed1(eps)
    X = str(eps)
    Y = Phi_F(Op)
    V = eigu(Op)[1]
            
    # file -> epsilon phi_f entropy    
    for j in range(2**N):
        f.write(X +'\t'+ str(Y[j].real)+ '\t' + str(Average_Entropy(Array2List(V[:,j:j+1]))) +'\n')   

