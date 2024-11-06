# https://github.com/ignaziopedone/shor-analysis/blob/597a5e1068742a4d4d90ed4ba2181b10cb849fb3/ECDLP/ECC_library.py
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit import Aer, execute, IBMQ, BasicAer
from qiskit.aqua import QuantumInstance
import math

def addition(circuit,x,y,c,N):
    for i in range(1,N):
        circuit.cx(x[i],y[i])
    circuit.cx(x[N-1],c)
    for i in range(N-2,0,-1):
        circuit.cx(x[i],x[i+1])
    for i in range(0,N-1):
        circuit.ccx(y[i],x[i],x[i+1])
    circuit.ccx(y[N-1],x[N-1],c)
    for i in range(N-1,0,-1):
        circuit.cx(x[i],y[i])
        circuit.ccx(y[i-1],x[i-1],x[i])
    for i in range(1,N-1):
        circuit.cx(x[i],x[i+1])
    for i in range(0,N):
        circuit.cx(x[i],y[i])
        
def subtraction(circuit,x,y,c,N):
    for i in range(N-1,-1,-1):
        circuit.cx(x[i],y[i])
    for i in range(N-2,0,-1):
        circuit.cx(x[i],x[i+1])
    for i in range(1,N):
        circuit.ccx(y[i-1],x[i-1],x[i])
        circuit.cx(x[i],y[i])
    circuit.ccx(y[N-1],x[N-1],c)
    for i in range(N-2,-1,-1):
        circuit.ccx(y[i],x[i],x[i+1])
    for i in range(1,N-1):
        circuit.cx(x[i],x[i+1])
    circuit.cx(x[N-1],c)
    for i in range(N-1,0,-1):
        circuit.cx(x[i],y[i])

def bitwise_complement(circuit,g,n):
    for i in range(0,n):
        circuit.x(g[i])

"""This function calculates just the carry bit of the addition between the value encoded into register a and the classical value c.
a in the register of n qubits to which we have to add the constant;
c is the constant;
g is the register of n-1 dirty ancilla qubits;
m is the qubit where we will save the carry."""
def carry_gate(circuit,a,g,m,c,n):
    circuit.cx(g[n-2],m)
    for i in range(n-1,1,-1):
        if(c[i]=='1'):
            circuit.cx(a[i],g[i-1])
            circuit.x(a[i])
        circuit.ccx(g[i-2],a[i],g[i-1])
    if(c[1]=='1'):
        circuit.cx(a[1],g[0])
        circuit.x(a[1])
    if(c[0]=='1'):
        circuit.ccx(a[0],a[1],g[0])
    for i in range(2,n):
        circuit.ccx(g[i-2],a[i],g[i-1])
    circuit.cx(g[n-2],m)
    for i in range(n-1,1,-1):
        circuit.ccx(g[i-2],a[i],g[i-1])
    if(c[0]=='1'):
        circuit.ccx(a[0],a[1],g[0])
    if(c[1]=='1'):
        circuit.x(a[1])
        circuit.cx(a[1],g[0])
    for i in range(2,n):
        circuit.ccx(g[i-2],a[i],g[i-1])
        if(c[i]=='1'):
            circuit.x(a[i])
            circuit.cx(a[i],g[i-1])
            
def controlled_carry_gate(circuit,ctrl1,a,g,m,c,n):
    circuit.ccx(ctrl1,g[n-2],m)
    for i in range(n-1,1,-1):
        if(c[i]=='1'):
            circuit.cx(a[i],g[i-1])
            circuit.x(a[i])
        circuit.ccx(g[i-2],a[i],g[i-1])
    if(c[1]=='1'):
        circuit.cx(a[1],g[0])
        circuit.x(a[1])
    if(c[0]=='1'):
        circuit.ccx(a[0],a[1],g[0])
    for i in range(2,n):
        circuit.ccx(g[i-2],a[i],g[i-1])
    circuit.ccx(ctrl1,g[n-2],m)
    for i in range(n-1,1,-1):
        circuit.ccx(g[i-2],a[i],g[i-1])
    if(c[0]=='1'):
        circuit.ccx(a[0],a[1],g[0])
    if(c[1]=='1'):
        circuit.x(a[1])
        circuit.cx(a[1],g[0])
    for i in range(2,n):
        circuit.ccx(g[i-2],a[i],g[i-1])
        if(c[i]=='1'):
            circuit.x(a[i])
            circuit.cx(a[i],g[i-1])

"""This circuit performs the increment of the value encoded into register x. We used the control qubit as the first alement of the register to increment, placing an X-gate on it at the end of the circuit, since:
-if it was 1, also the first qubit of the register to increment will toggle and so on;
-if it was 0, it will be the only one to be toggled.
The mathematical passages of the computation are: |x>|g> -> |x-g>|g> -> |x-g>|g'-1> -> |x-g-g'+1>|g'-1> -> |x+1>|g>, where:
-x is the register to increment with the control qubit as first element, so n qubits;
-g is a register with n-1 dirty qubits;
-g'-1 is the bitwise complement of g."""
def incremental_gate_controlled(circuit,x,g,n):
    if(n==2):
        circuit.cx(x[0],x[1])
    elif(n==3):
        circuit.ccx(x[0],x[1],x[2])
        circuit.cx(x[0],x[1])
    else:
        subtraction(circuit,g,x[:n-1],x[n-1],n-1)
        bitwise_complement(circuit,g,n-1)
        subtraction(circuit,g,x[:n-1],x[n-1],n-1)
        bitwise_complement(circuit,g,n-1)
        circuit.x(x[0])
        circuit.x(x[n-1])
        
"""The next 2 functions perform respectively the addition and the subtraction of a constant value (c) to the value encoded in the quantum register (x) of n qubits. The constant addition will need a dirty ancilla qubit g for the computation, that will be returned in the same state."""
def constant_addition(circuit,x,g,c,n):
    if(n==2):
        if(c[0]=='1'):
            circuit.cx(x[0],x[1])
        constant_addition(circuit,x[:1],g,c[:1],1)
        constant_addition(circuit,x[1:],g,c[1:],1)
    elif(n==1):
        if(c[0]=='1'):
            circuit.x(x[0])
    else:
        if(n%2==0):
            l=n//2
        else:
            l=n//2+1
        incremental_gate_controlled(circuit,g[0:1]+x[l:n],x[:l],n-l+1)
        for i in range(l,n):
            circuit.cx(g,x[i])
        carry_gate(circuit,x[:l],x[l:],g,c[:l],l)
        incremental_gate_controlled(circuit,g[0:1]+x[l:n],x[:l],n-l+1)
        carry_gate(circuit,x[:l],x[l:],g,c[:l],l)
        for i in range(l,n):
            circuit.cx(g,x[i])
        constant_addition(circuit,x[:l],g,c[:l],l)
        constant_addition(circuit,x[l:],g,c[l:],n-l)
        
"""The subtraction is calculated performing the inverse computation of the addition."""
def constant_subtraction(circuit,x,g,c,n):
    sub_x = QuantumRegister(n)
    sub_g = QuantumRegister(1)
    sub_circ = QuantumCircuit(sub_x,sub_g, name='adder')
    constant_addition(sub_circ,sub_x,sub_g,c,n)
    sub_inst = sub_circ.inverse().to_gate()
    circuit.append(sub_inst,x[0:n]+g[0:1])

"""The next 6 functions are just useful to create controlled versions of previous ones"""
def controlled_addition(circuit,ctrl,x,y,c,n):
    sub_x = QuantumRegister(n)
    sub_y = QuantumRegister(n)
    sub_c = QuantumRegister(1)
    sub_circ = QuantumCircuit(sub_x,sub_y,sub_c, name='adder')
    addition(sub_circ,sub_x,sub_y,sub_c,n)
    sub_inst = sub_circ.to_gate().control()
    circuit.append(sub_inst,ctrl[0:1]+x[0:n]+y[0:n]+c[0:1])
    
def controlled_subtraction(circuit,ctrl,x,y,c,n):
    sub_x = QuantumRegister(n)
    sub_y = QuantumRegister(n)
    sub_c = QuantumRegister(1)
    sub_circ = QuantumCircuit(sub_x,sub_y,sub_c, name='sub')
    subtraction(sub_circ,sub_x,sub_y,sub_c,n)
    sub_inst = sub_circ.to_gate().control()
    circuit.append(sub_inst,ctrl[0:1]+x[0:n]+y[0:n]+c[0:1])
    
def doubly_controlled_addition(circuit,ctrl1,ctrl2,x,y,c,n):
    sub_x = QuantumRegister(n)
    sub_y = QuantumRegister(n)
    sub_c = QuantumRegister(1)
    sub_circ = QuantumCircuit(sub_x,sub_y,sub_c, name='adder')
    addition(sub_circ,sub_x,sub_y,sub_c,n)
    sub_inst = sub_circ.to_gate().control().control()
    circuit.append(sub_inst,ctrl1[0:1],ctrl2[0:1]+x[0:n]+y[0:n]+c[0:1])
    
def doubly_controlled_subtraction(circuit,ctrl1,ctrl2,x,y,c,n):
    sub_x = QuantumRegister(n)
    sub_y = QuantumRegister(n)
    sub_c = QuantumRegister(1)
    sub_circ = QuantumCircuit(sub_x,sub_y,sub_c, name='sub')
    subtraction(sub_circ,sub_x,sub_y,sub_c,n)
    sub_inst = sub_circ.to_gate().control().control()
    circuit.append(sub_inst,ctrl1[0:1],ctrl2[0:1]+x[0:n]+y[0:n]+c[0:1])
    
def controlled_constant_addition(circuit,ctrl,x,g,c,n):
    if(n==2):
        if(c[0]=='1'):
            circuit.ccx(ctrl,x[0],x[1])
        controlled_constant_addition(circuit,ctrl,x[:1],g,c[:1],1)
        controlled_constant_addition(circuit,ctrl,x[1:],g,c[1:],1)
    elif(n==1):
        if(c[0]=='1'):
            circuit.cx(ctrl,x[0])
    else:
        if(n%2==0):
            l=n//2
        else:
            l=n//2+1
        incremental_gate_controlled(circuit,g[0:1]+x[l:n],x[:l],n-l+1)
        for i in range(l,n):
            circuit.cx(g,x[i])
        controlled_carry_gate(circuit,ctrl,x[:l],x[l:],g,c[:l],l)
        incremental_gate_controlled(circuit,g[0:1]+x[l:n],x[:l],n-l+1)
        controlled_carry_gate(circuit,ctrl,x[:l],x[l:],g,c[:l],l)
        for i in range(l,n):
            circuit.cx(g,x[i])
        controlled_constant_addition(circuit,ctrl,x[:l],g,c[:l],l)
        controlled_constant_addition(circuit,ctrl,x[l:],g,c[l:],n-l)

"""To remove after test of the improved method"""
def basic_controlled_constant_addition(circuit,ctrl,x,g,c,n):
    sub_x = QuantumRegister(n)
    sub_g = QuantumRegister(1)
    sub_circ = QuantumCircuit(sub_x,sub_g, name='const_adder')
    constant_addition(sub_circ,sub_x,sub_g,c,n)
    sub_inst = sub_circ.to_gate().control()
    circuit.append(sub_inst,ctrl[0:1]+x[0:n]+g[0:1])
    
def controlled_constant_subtraction(circuit,ctrl,x,g,c,n):
    sub_ctrl = QuantumRegister(1)
    sub_x = QuantumRegister(n)
    sub_g = QuantumRegister(1)
    sub_circ = QuantumCircuit(sub_ctrl,sub_x,sub_g, name='ctrl_adder')
    controlled_constant_addition(sub_circ,sub_ctrl,sub_x,sub_g,c,n)
    sub_inst = sub_circ.inverse().to_gate()
    circuit.append(sub_inst,ctrl[0:1]+x[0:n]+g[0:1])

"""This function will become useful as part of a comparator"""   
def addition_without_carry(circuit,x,y,N):
    for i in range(1,N):
        circuit.cx(x[i],y[i])
    for i in range(N-2,0,-1):
        circuit.cx(x[i],x[i+1])
    for i in range(0,N-1):
        circuit.ccx(y[i],x[i],x[i+1])
    for i in range(N-1,0,-1):
        circuit.cx(x[i],y[i])
        circuit.ccx(y[i-1],x[i-1],x[i])
    for i in range(1,N-1):
        circuit.cx(x[i],x[i+1])
    for i in range(0,N):
        circuit.cx(x[i],y[i])
        
def controlled_addition_without_carry(circuit,ctrl,x,y,n):
    sub_x = QuantumRegister(n)
    sub_y = QuantumRegister(n)
    sub_circ = QuantumCircuit(sub_x,sub_y,name='add_no_car')
    addition_without_carry(sub_circ,sub_x,sub_y,n)
    sub_inst = sub_circ.to_gate().control()
    circuit.append(sub_inst,ctrl[0:1]+x[0:n]+y[0:n])
    
def doubly_controlled_addition_without_carry(circuit,ctrl1,ctrl2,x,y,n):
    sub_x = QuantumRegister(n)
    sub_y = QuantumRegister(n)
    sub_circ = QuantumCircuit(sub_x,sub_y,name='add_no_car')
    addition_without_carry(sub_circ,sub_x,sub_y,n)
    sub_inst = sub_circ.to_gate().control().control()
    circuit.append(sub_inst,ctrl[0:1]+x[0:n]+y[0:n])
    
def subtraction_without_carry(circuit,x,y,N):
    for i in range(N-1,-1,-1):
        circuit.cx(x[i],y[i])
    for i in range(N-2,0,-1):
        circuit.cx(x[i],x[i+1])
    for i in range(1,N):
        circuit.ccx(y[i-1],x[i-1],x[i])
        circuit.cx(x[i],y[i])
    for i in range(N-2,-1,-1):
        circuit.ccx(y[i],x[i],x[i+1])
    for i in range(1,N-1):
        circuit.cx(x[i],x[i+1])
    for i in range(N-1,0,-1):
        circuit.cx(x[i],y[i])

def controlled_subtraction_without_carry(circuit,ca,x,y,n):
    sub_x = QuantumRegister(n)
    sub_y = QuantumRegister(n)
    sub_circuit = QuantumCircuit(sub_x,sub_y,name='-')
    subtraction_without_carry(sub_circuit,sub_x,sub_y,n)
    sub_inst=sub_circuit.to_gate().control()
    circuit.append(sub_inst,ca[0:1]+x[0:n]+y[0:n])

"""This gate flips the qubit c if the value encoded in y is <= that the value encoded in x"""
def comparator(circuit,x,y,c,n):
    subtraction(circuit,y,x,c,n)
    addition_without_carry(circuit,y,x,n)
    
def controlled_comparator(circuit,ctrl,x,y,c,n):
    controlled_subtraction(circuit,ctrl,y,x,c,n)
    controlled_addition_without_carry(circuit,ctrl,y,x,n)
    
def doubly_controlled_comparator(circuit,ctrl1,ctrl2,x,y,c,n):
    doubly_controlled_subtraction(circuit,ctrl1,ctrl2,y,x,c,n)
    doubly_controlled_addition_without_carry(circuit,ctrl1,ctrl2,y,x,n)

def binary_halving_cyclic(circuit,x,n):
    for i in range(1,n):
        circuit.cx(x[i-1],x[i])
        circuit.cx(x[i],x[i-1])
        circuit.cx(x[i-1],x[i])
    
def binary_doubling_cyclic(circuit,x,n):
    for i in range(n-1,0,-1):
        circuit.cx(x[i-1],x[i])
        circuit.cx(x[i],x[i-1])
        circuit.cx(x[i-1],x[i])
        
"""This is useful for the controlled modular doubling"""
def controlled_binary_doubling_cyclic(circuit,ctrl,x,n):
    for i in range(n-1,0,-1):
        circuit.ccx(ctrl,x[i-1],x[i])
        circuit.ccx(ctrl,x[i],x[i-1])
        circuit.ccx(ctrl,x[i-1],x[i])
        
def controlled_binary_halving_cyclic(circuit,ctrl,x,n):
    for i in range(1,n):
        circuit.ccx(ctrl,x[i-1],x[i])
        circuit.ccx(ctrl,x[i],x[i-1])
        circuit.ccx(ctrl,x[i-1],x[i])
    
"""The addition modulo p of the registers x and y is perfomed using a clear qubit 'c' and a dirty qubit 'g'. The result is stored in y"""
def addition_modp(circuit,x,y,c,g,p,n):
    p1 = p+'0'
    addition(circuit,x,y,c,n)
    constant_subtraction(circuit,y[0:n]+c[0:1],g,p1,n+1)
    controlled_constant_addition(circuit,c,y,g,p,n)
    comparator(circuit,y,x,c,n)
    circuit.x(c)

"""The doubling modulo p (only odd) of the registers x is perfomed using a clear qubit 'c' and a dirty qubit 'g'. The result is stored in x"""   
def doubling_modp(circuit,x,c,g,p,n):
    p1 = p+'0'
    binary_doubling_cyclic(circuit,x[0:n]+c[0:1],n+1)
    constant_subtraction(circuit,x[0:n]+c[0:1],g,p1,n+1)
    controlled_constant_addition(circuit,c,x,g,p,n)
    circuit.x(x[0])
    circuit.cx(x[0],c)
    circuit.x(x[0])
    
"""The next 3 functions are just useful to create controlled versions of previous ones"""
def controlled_addition_modp(circuit,ctrl,x,y,c,g,p,n):
    p1 = p+'0'
    controlled_addition(circuit,ctrl,x,y,c,n)
    constant_subtraction(circuit,y[0:n]+c[0:1],g,p1,n+1)
    controlled_constant_addition(circuit,c,y,g,p,n)
    controlled_comparator(circuit,ctrl,y,x,c,n)
    circuit.x(c)
    
def doubly_controlled_addition_modp(circuit,ctrl1,ctrl2,x,y,c,g,p,n):
    p1 = p+'0'
    doubly_controlled_addition(circuit,ctrl,x,y,c,n)
    constant_subtraction(circuit,y[0:n]+c[0:1],g,p1,n+1)
    controlled_constant_addition(circuit,c,y,g,p,n)
    doubly_controlled_comparator(circuit,ctrl1,ctrl2,y,x,c,n)
    circuit.x(c)
    
def controlled_doubling_modp(circuit,ctrl,x,c,g,p,n):
    controlled_binary_doubling_cyclic(circuit,ctrl,x[0:n]+c[0:1],n+1)
    constant_subtraction(circuit,x[0:n]+c[0:1],g,s_p1,n+1)
    controlled_constant_addition(circuit,c,x,g,s_p,n)
    circuit.x(x[0])
    circuit.ccx(ctrl,x[0],c)
    circuit.x(x[0])
    circuit.x(ctrl)
    circuit.cx(ctrl,c)
    circuit.x(ctrl)

"""The multiplication modulo p of the registers x and y is perfomed using a clear register 'z' of n qubits, a clear qubit 'c' and a dirty qubit 'g'. The result is stored in z(=x*y mod p)"""
def multiplication_modp(circuit,x,y,z,c,g,p,n):
    for i in range(n-1,0,-1):
        controlled_addition_modp(circuit,x[i:i+1],y,z,c,g,p,n)
        doubling_modp(circuit,z,c,g,p,n)
    controlled_addition_modp(circuit,x[0:1],y,z,c,g,p,n)
    
"""The squaring modulo p of the registers x is perfomed using a clear qubit 'ca', a clear register 'z' of n qubits, a clear qubit 'cb' and a dirty qubit 'g'. The result is stored in z(=x*x mod p)"""
def square_modp(circuit,ca,x,z,cb,g,p,n):
    for i in range(n-1,0,-1):
        circuit.cx(x[i],ca)
        controlled_addition_modp(circuit,ca,x,z,cb,g,s_p,n)
        circuit.cx(x[i],ca)
        doubling_modp(circuit,z,cb,g,s_p,n)
    circuit.cx(x[0],ca)
    controlled_addition_modp(circuit,ca,x,z,cb,g,s_p,n)
    circuit.cx(x[0],ca)
       
"""Quantum circuit for the forward Montgomery modular multiplication |x> |y> |0> --> |x> |y> |r = x * y * R_dag mod p>. 
'z','m' and 'r' are clear registers of n qubits, 'z' and 'm' are returned clear, while 'r' stores the result.
The qubit 'g' is a dirty ancilla qubit in an unknown state, while the qubit 'c' is a clear qubit.
The sub_circuit stores the result to 'z', that is copied to 'r', and then everything is uncomputed running the sub_circuit backward."""
def montgomery_multiplication_modp(circuit,x,y,c,z,g,m,r,p,n):
    p1 = p+'0'
    sub_x = QuantumRegister(n)
    sub_y = QuantumRegister(n)
    sub_c = QuantumRegister(1)
    sub_z = QuantumRegister(n)
    sub_g = QuantumRegister(1)
    sub_m = QuantumRegister(n)
    sub_circuit = QuantumCircuit(sub_x,sub_y,sub_c,sub_z,sub_g,sub_m, name='mul_mod_p')
    for i in range (0,n):
        controlled_addition(sub_circuit,sub_x[i:i+1],sub_y,sub_z,sub_c,n)
        sub_circuit.cx(sub_z[0],sub_m[i])
        controlled_constant_addition(sub_circuit,sub_m[i:i+1],sub_z[0:n]+sub_c[0:1],sub_g,p1,n+1)
        binary_halving_cyclic(sub_circuit,sub_z[0:n]+sub_c[0:1],n+1)
    constant_subtraction(sub_circuit,sub_z[0:n]+sub_c[0:1],sub_g,p1,n+1)
    controlled_constant_addition(sub_circuit,sub_c,sub_z,sub_g,p,n)
    sub_inst = sub_circuit.to_gate()
    circuit.append(sub_inst,x[0:n]+y[0:n]+c[0:1]+z[0:n]+g[0:1]+m[0:n])
    for i in range (0,n):
        circuit.cx(z[i],r[i])
    circuit.append(sub_inst.inverse(),x[0:n]+y[0:n]+c[0:1]+z[0:n]+g[0:1]+m[0:n])

    
"""Quantum circuit for the forward Montgomery modular squaring |x> |0> --> |x> |r = x * x * R_dag mod p>. 
'z','m' and 'r' are clear registers of n qubits, 'z' and 'm' are returned clear, while 'r' stores the result.
The qubit 'g' is a dirty ancilla qubit in an unknown state, while qubits 'c' and 'ca' are clear qubits.
The sub_circuit stores the result to 'z', that is copied to 'r', and then everything is uncomputed running the sub_circuit backward."""
def montgomery_squaring_modp(circuit,ca,x,c,z,g,m,r,p,n):
    p1=p+'0'
    sub_ca = QuantumRegister(1)
    sub_x = QuantumRegister(n)
    sub_c = QuantumRegister(1)
    sub_z = QuantumRegister(n)
    sub_g = QuantumRegister(1)
    sub_m = QuantumRegister(n)
    sub_circuit = QuantumCircuit(sub_ca,sub_x,sub_c,sub_z,sub_g,sub_m, name='mul_mod_p')
    for i in range (0,n):
        sub_circuit.cx(sub_x[i],sub_ca)
        controlled_addition(sub_circuit,sub_ca[0:1],sub_x,sub_z,sub_c,n)
        sub_circuit.cx(sub_x[i],sub_ca)
        sub_circuit.cx(sub_z[0],sub_m[i])
        controlled_constant_addition(sub_circuit,sub_m[i:i+1],sub_z[0:n]+sub_c[0:1],sub_g,p1,n+1)
        binary_halving_cyclic(sub_circuit,sub_z[0:n]+sub_c[0:1],n+1)
    constant_subtraction(sub_circuit,sub_z[0:n]+sub_c[0:1],sub_g,p1,n+1)
    controlled_constant_addition(sub_circuit,sub_c,sub_z,sub_g,p,n)
    sub_inst = sub_circuit.to_gate()
    circuit.append(sub_inst,ca[0:1]+x[0:n]+c[0:1]+z[0:n]+g[0:1]+m[0:n])
    for i in range (0,n):
        circuit.cx(z[i],r[i])
    circuit.append(sub_inst.inverse(),ca[0:1]+x[0:n]+c[0:1]+z[0:n]+g[0:1]+m[0:n])
    
"""The next functions (until the 'round' one) will be auxiliary functions to compute the modular inversion"""
    
"""Auxiliary function that creates a controlled X gate acting on qubit t depending on register x. If x contains all |0> and the type varible is set to '0', the t qubit will toggle. If x contains all |1> and the type varible is set to '1', the t qubit will toggle."""
def register_cnot(circuit,x,t,type,n_ctrl):
    if type=='0':
        for i in range(0,n_ctrl):
            circuit.x(x[i])
        circuit.mcx(x,t)
        for i in range(0,n_ctrl):
            circuit.x(x[i])
    else:
        circuit.mcx(x,t)
        
def controlled_swap(circuit,ctrl,x,y,n):
    for i in range(0,n):
        circuit.cx(x[i],y[i])
        circuit.ccx(ctrl,y[i],x[i])
        circuit.cx(x[i],y[i])
               
"""Just 4 auxiliary functions that are doubly controlled versions of already written functions. The doubly controlled addition and subtraction does not calculate the carry, since it should not be used in the Kaliski's algorithm."""            

def doubly_controlled_halving(circuit,ca,cb,x,n):
    sub_x = QuantumRegister(n)
    sub_circuit = QuantumCircuit(sub_x,name='/2')
    binary_halving_cyclic(sub_circuit,sub_x,n)
    sub_inst=sub_circuit.to_gate().control().control()
    circuit.append(sub_inst,ca[0:1]+cb[0:1]+x[0:n])
    
def doubly_controlled_doubling(circuit,ca,cb,x,n):
    sub_x = QuantumRegister(n)
    sub_circuit = QuantumCircuit(sub_x,name='*2')
    binary_doubling_cyclic(sub_circuit,sub_x,n)
    sub_inst=sub_circuit.to_gate().control().control()
    circuit.append(sub_inst,ca[0:1]+cb[0:1]+x[0:n])
    
def doubly_controlled_addition_no_carry(circuit,ca,cb,x,y,n):
    sub_x = QuantumRegister(n)
    sub_y = QuantumRegister(n)
    sub_circuit = QuantumCircuit(sub_x,sub_y,name='+')
    addition_without_carry(sub_circuit,sub_x,sub_y,n)
    sub_inst=sub_circuit.to_gate().control().control()
    circuit.append(sub_inst,ca[0:1]+cb[0:1]+x[0:n]+y[0:n])
    
def doubly_controlled_subtraction_no_carry(circuit,ca,cb,x,y,n):
    sub_x = QuantumRegister(n)
    sub_y = QuantumRegister(n)
    sub_circuit = QuantumCircuit(sub_x,sub_y,name='-')
    addition_without_carry(sub_circuit,sub_x,sub_y,n)
    sub_inst=sub_circuit.to_gate().inverse().control().control()
    circuit.append(sub_inst,ca[0:1]+cb[0:1]+x[0:n]+y[0:n])
    
"""The block and the round functions perform the operations described in https://arxiv.org/abs/1706.06752 (par. 3.4)"""    

def block(circuit,u,v,ca,s,r,cb,cc,m,n):
    circuit.x(u[0])
    circuit.cx(u[0],cc)
    circuit.x(u[0])
    circuit.x(v[0])
    circuit.x(cc)
    circuit.ccx(v[0],cc,m)
    circuit.x(cc)
    circuit.x(v[0])
    circuit.cx(cc,cb)
    circuit.cx(m,cb)
    subtraction(circuit,u,v,ca,n)
    circuit.x(cb)
    circuit.ccx(ca,cb,cc)
    circuit.ccx(ca,cb,m)
    circuit.x(cb)
    circuit.cx(m,cb)
    circuit.cx(cc,cb)
    addition(circuit,u,v,ca,n)
    circuit.x(m)
    doubly_controlled_halving(circuit,cc,m,u,n)
    doubly_controlled_doubling(circuit,cc,m,s,n)
    circuit.x(m)
    circuit.x(cc)
    doubly_controlled_halving(circuit,cc,m,v,n)
    doubly_controlled_doubling(circuit,cc,m,r,n)
    circuit.x(cc)
    doubly_controlled_subtraction_no_carry(circuit,cc,m,v,u,n)
    doubly_controlled_addition_no_carry(circuit,cc,m,s,r,n)
    doubly_controlled_halving(circuit,cc,m,u,n)
    doubly_controlled_doubling(circuit,cc,m,s,n)
    circuit.x(m)
    circuit.x(cc)
    doubly_controlled_subtraction_no_carry(circuit,cc,m,u,v,n)
    doubly_controlled_addition_no_carry(circuit,cc,m,r,s,n)
    doubly_controlled_halving(circuit,cc,m,v,n)
    doubly_controlled_doubling(circuit,cc,m,r,n)
    circuit.x(m)
    circuit.x(cc)
    circuit.cx(r[0],cc)

"""Controlled block created from improved version with swaps explained at https://arxiv.org/abs/2001.09580"""    
def controlled_block(circuit,ctrl,u,v,ca,s,r,cb,cc,m,n):
    circuit.x(u[0])
    circuit.ccx(ctrl,u[0],cc)
    circuit.x(u[0])
    circuit.x(v[0])
    circuit.x(cc)
    circuit.mcx(ctrl[0:1]+v[0:1]+cc[0:1],m)
    circuit.x(cc)
    circuit.x(v[0])
    circuit.cx(cc,cb)
    circuit.cx(m,cb)
    comparator(circuit,v,u,ca,n)
    circuit.x(cb)
    circuit.ccx(ca,cb,cc)
    circuit.ccx(ca,cb,m)
    circuit.x(cb)
    circuit.cx(m,cb)
    circuit.cx(cc,cb)
    circuit.x(ca)
    comparator(circuit,u,v,ca,n)
    controlled_swap(circuit,cc,u,v,n)
    controlled_swap(circuit,cc,r,s,n)
    circuit.cx(ctrl,cc)
    circuit.cx(m,cc)
    controlled_subtraction_without_carry(circuit,cc,u,v,n)
    controlled_addition_without_carry(circuit,cc,r,s,n)
    circuit.cx(m,cc)
    circuit.cx(ctrl,cc)
    controlled_binary_halving_cyclic(circuit,ctrl,v,n)
    controlled_binary_doubling_cyclic(circuit,ctrl,r,n)
    controlled_swap(circuit,cc,u,v,n)
    controlled_swap(circuit,cc,r,s,n)
    circuit.cx(r[0],cc)
    
def round(circuit,u,v,ca,s,r,cb,cc,m,f,k,l,n):
    register_cnot(circuit,k,f,'0',l) #the qubit is toggled if k is 0, so we add another x-gate
    circuit.x(f)
    register_cnot(circuit,v,f,'0',n)
    circuit.x(f)
    incremental_gate_controlled(circuit,f[0:1]+k[0:l],v[0:l],l+1)
    circuit.x(f)
    #creating sub_circuit in order to apply the control to the block
    """Version using basic block"""
    """sub_u = QuantumRegister(n)
    sub_v = QuantumRegister(n)
    sub_ca = QuantumRegister(1)
    sub_s = QuantumRegister(n)
    sub_r = QuantumRegister(n)
    sub_cb = QuantumRegister(1)
    sub_cc = QuantumRegister(1)
    sub_m = QuantumRegister(1)
    sub_circuit = QuantumCircuit(sub_u,sub_v,sub_ca,sub_s,sub_r,sub_cb,sub_cc,sub_m,name='block')
    block(sub_circuit,sub_u,sub_v,sub_ca,sub_s,sub_r,sub_cb,sub_cc,sub_m,n)
    sub_inst = sub_circuit.to_gate().control()
    circuit.append(sub_inst,f[0:1]+u[0:n]+v[0:n]+ca[0:1]+s[0:n]+r[0:n]+cb[0:1]+cc[0:1]+m[0:1])"""
    """Version using block with swaps"""
    controlled_block(circuit,f,u,v,ca,s,r,cb,cc,m,n)
    
"""The Montgomery modular inversion will perform the transformation 'x mod p -> (x^-1) * (2^n) mod p'.
The output of the circuit shown below is '- (x^-1) * (2^k) mod p', encoded in register r.
As a future work, the register r has to be copied into another register, and n-k (a value that will be already encoded into register 'k' after the aplication of this circuit) controlled modular doublings have to be applied.
In order to do this, a suggestion is to create n+1 controlled modular doublings gates, activating each one depending on the value encoded into register 'k'.
After that, a sign flip has to be applied.
Then, we can multiply the result for (2^n mod p)^(-1) mod p, that is a classical value, in order to obtain x^(-1) mod p.
Now the register will contain the correct value encoded inside it, so we can apply the inverse circuit of the one shown below in order to uncompute every register.
The inputs of this function are:
- u, a register of n qubits that will contain the modulo p encoded;
- v, a register of n qubits that will contain the value x encoded;
- ca, a clean ancilla qubit;
- s, a register of n qubits that will start with the value 1 encoded;
- r, a register of n qubits that will start with the value 0 encoded;
- cb, a clean ancilla qubit;
- cc, a clean ancilla qubit;
- m, a register of 2n clean ancilla qubits that will be useful to save the operation performed in every application of the block circuit, so it will be useful for the uncomputation;
- f, a clean ancilla qubit that will be toggled when the correct result is reached;
- k, a register of l = log(n) qubits that will contain the value n-k at the end of the circuit."""
def mod_inv_function(circuit,u,v,ca,s,r,cb,cc,m,f,k,l,n):
    circuit.x(s[0])
    circuit.x(f)
    for i in range(0,2*n):
        round(circuit,u,v,ca,s,r,cb,cc,m[i:i+1],f,k,n,l)
    
"""The capabilities of the simulators used for testing permit to perform the computation up to this point."""

def mod_inv_function_inverse(circuit,u,v,ca,s,r,cb,cc,m,f,k,n,l):
    sub_u = QuantumRegister(n,'u')
    sub_v = QuantumRegister(n,'v')
    sub_ca = QuantumRegister(1,'ca')
    sub_s = QuantumRegister(n,'s')
    sub_r = QuantumRegister(n,'r')
    sub_cb = QuantumRegister(1,'cb')
    sub_cc = QuantumRegister(1,'cc')
    sub_m = QuantumRegister(2*n,'m')
    sub_f = QuantumRegister(1,'f')
    sub_k = QuantumRegister(l,'k')
    sub_circ = QuantumCircuit(sub_u,sub_v,sub_ca,sub_s,sub_r,sub_cb,sub_cc,sub_m,sub_f,sub_k)
    mod_inv_function(sub_circ,sub_u,sub_v,sub_ca,sub_s,sub_r,sub_cb,sub_cc,sub_m,sub_f,sub_k,n,l)
    sub_inst = sub_circ.inverse().to_gate()
    circuit.append(sub_inst,u[0:n]+v[0:n]+ca[0:1]+s[0:n]+r[0:n]+cb[0:1]+cc[0:1]+m[0:2*n]+f[0:1]+k[0:l])

"""This function permits to calculate the modular inversion of the number encoded into register v, while the modulo is encoded into register u.
The inputs are:
- u, a clear register of n qubits that will contain the modulo p encoded;
- v, a register of n qubits that contains the value x encoded;
- ca, a clean ancilla qubit;
- s, a register of n+1* qubits that will start with the value 1 encoded;
- r, a register of n+1* qubits that will start with the value 0 encoded;
- cb, a clean ancilla qubit;
- cc, a clean ancilla qubit;
- m, a register of 2n clean ancilla qubits that will be useful to save the operation performed in every application of the block circuit, so it will be useful for the uncomputation;
- f, a clean ancilla qubit that will be toggled when the correct result is reached;
- k, a register of l = log(n)+1 qubits that will contain the value n-k at the end of the circuit;
- y_r, a clear register that will contain the output of the function at the end of the circuit;
- y_k, a clear register of l qubits that will be used to copy the register k at the end of the circuit.
Total number of qubits: 7n + 2l + 7 = 7n + 2log(n) + 9"""
def mod_inv(circuit,u,v,ca,s,r,cb,cc,m,f,k,y_r,y_k,p,l,n):
    for i in range(0,n):
        if p[i] == '1':
            circuit.x(u[i])
    mod_inv_function(circuit,u,v,ca,s,r,cb,cc,m,f,k,n,l)
    for i in range(0,n):
        circuit.cx(r[i],y_r[i])
        if(i<l):
            circuit.cx(k[i],y_k[i])
    for i in range(0,n+1):
        """if k>0, then y*=2. We can use f as ancilla qubit, creating a register of l+1 qubits with f as msb (control concatenation)."""
        """Every iteration, we perform k-i and check if f is toggled"""
        constant_subtraction(circuit,y_k[0:l]+f[0:1],ca,i+1,l+1)
        controlled_doubling_modp(circuit,f,y_r,ca,cb,p,n)
        constant_addition(circuit,y_k[0:l]+f[0:1],ca,i+1,l+1)
    bitwise_complement(circuit,y_r,n)
    for i in range(0,l):
        circuit.cx(k[i],y_k[i])
    mod_inv_function_inverse(circuit,u,v,ca,s,r,cb,cc,m,f,k,n,l)
        
