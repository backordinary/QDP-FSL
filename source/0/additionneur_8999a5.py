# https://github.com/3gaspo/guide-infoQ/blob/ae8ec94a5bfb715168017518abb4beb51c969713/codes/additionneur.py
##Additionneur
from qiskit import QuantumCircuit, execute, Aer

#additionneur 1 bit tout seul
def add(a,b):
    '''
    ajoute les bits a et b, qui valent 0 ou 1
    '''
    qc = QuantumCircuit(4,2)
    
    if a == 1:
        qc.initialize([0,1],0)
    if b == 1:
        qc.initialize([0,1],1)
    
    qc.barrier() #pour la visiblité

    #XOR
    qc.cx(0,2)
    qc.cx(1,2)
    #Retenue
    qc.ccx(0,1,3)
    
    qc.barrier()
    
    qc.measure(2,0)
    qc.measure(3,1)
    
    return qc


from qiskit.visualization import plot_histogram
def counts_add(qc):
    backend = Aer.get_backend('qasm_simulator')
    counts = execute(qc, backend).result().get_counts()
    return(counts)

#exemple avec 1+1
qc = add(1,1)
plot_histogram(counts_add(qc))


#Additionneur 1bit pour additionneur complet
def add(qc,ia,ib, ic, ir):
    '''
    ajoute les bits a et b entre eux (avec retenue)
    qc : le circuit complet
    ia : rang du bit a dans qc
    ib : rang du bit b dans qc
    ic : rang du bit du résultat de a+b
    ir : rang du bit de retenue
    '''
    qc.barrier()
    
    #XOR
    qc.cx(ia,ic)
    qc.cx(ib,ic)
    
    #Retenue
    qc.ccx(ia,ib,ir)
    
    qc.barrier()

def add_retenue(qc,ic,ir1,ir2):
    '''
    ajoute la retenue précédente au résultat précédent
    qc : circuit complet
    ic : rang du résultat
    ir1 : rang retenue précédente
    ir2 : rang seconde retenue
    '''
    qc.barrier()
    
    #Retenue
    qc.ccx(ir1,ic,ir2)
    #XOR
    qc.cx(ir1,ic)

    qc.barrier()
  
    
def state(a):
    '''
    fonction qui renvoie l'état initial de a
    ex : a = '1' renvoie le ket [0,1] = |1>
    '''
    if int(a)==1:
        return([0,1])
    else :
        return([1,0])


def add_multi(a,b):
    '''
    a et b sous forme de chaine de caractère
    ex : a = '101'
    '''

    n = len(a)
    assert(len(b)==n)
    
    qc = QuantumCircuit(4*n,n+1)
    
    #initialisation des entrées
    for i in range(n):
        sa = state(a[n-1-i])
        sb = state(b[n-1-i])
        
        qc.initialize(sa,i)
        qc.initialize(sb,n+i)
    
    #additions
    for i in range(n):
        
        add(qc, i, n+i, 2*n+i,3*n+i)
        
        #addition des retenues
        if i >= 1:
            
            add_retenue(qc, 2*n+i,3*n+i-1,3*n+i)
    
    #mesures
    qc.measure(4*n-1,n)
    for i in range(n):
        qc.measure(2*n+n-i-1,n-i-1)
        
    return qc

#exemple :
qc = add_multi('101','011')
plot_histogram(counts_add(qc))

            

        
        

    
    
