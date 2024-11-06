# https://github.com/Northerneye/QuantumErrorCorrectedSensorCodes/blob/e133fcac549c5c8fec67e00722341d3d44e21f7c/KLStein.py
import qiskit as qs
import numpy as np
import sympy as sy
import cmath
import math

sy.var('phi x ', real=True)

global myphi
myphi = float(0.1)
#degree = 0.1

ZeroLogical = [0 for i in range(2**7)]
ZeroLogical[0] = 1/np.sqrt(8)
ZeroLogical[2**6 + 2**4 + 2**2 + 2**0] = 1/np.sqrt(8)
ZeroLogical[2**5 + 2**4 + 2**1 + 2**0] = 1/np.sqrt(8)
ZeroLogical[2**6 + 2**5 + 2**2 + 2**1] = 1/np.sqrt(8)
ZeroLogical[2**3 + 2**2 + 2**1 + 2**0] = 1/np.sqrt(8)
ZeroLogical[2**6 + 2**4 + 2**3 + 2**1] = 1/np.sqrt(8)
ZeroLogical[2**5 + 2**4 + 2**3 + 2**2] = 1/np.sqrt(8)
ZeroLogical[2**6 + 2**5 + 2**3 + 2**0] = 1/np.sqrt(8)

OneLogical = [0 for i in range(2**7)]
OneLogical[2**7-1] = 1/np.sqrt(8)
OneLogical[2**5 + 2**3 + 2**1] = 1/np.sqrt(8)
OneLogical[2**6 + 2**3 + 2**2] = 1/np.sqrt(8)
OneLogical[2**4 + 2**3 + 2**0] = 1/np.sqrt(8)
OneLogical[2**6 + 2**5 + 2**4] = 1/np.sqrt(8)
OneLogical[2**5 + 2**2 + 2**0] = 1/np.sqrt(8)
OneLogical[2**6 + 2**1 + 2**0] = 1/np.sqrt(8)
OneLogical[2**4 + 2**2 + 2**1] = 1/np.sqrt(8)



myOperator = [[float(0) for i in range(2**7)] for j in range(2**7)]

myOperator[0][0] = 1
myOperator[2**6 + 2**4 + 2**2 + 2**0][2**6 + 2**4 + 2**2 + 2**0] = 1
myOperator[2**5 + 2**4 + 2**1 + 2**0][2**5 + 2**4 + 2**1 + 2**0] = 1
myOperator[2**6 + 2**5 + 2**2 + 2**1][2**6 + 2**5 + 2**2 + 2**1] = 1
myOperator[2**3 + 2**2 + 2**1 + 2**0][2**3 + 2**2 + 2**1 + 2**0] = 1
myOperator[2**6 + 2**4 + 2**3 + 2**1][2**6 + 2**4 + 2**3 + 2**1] = 1
myOperator[2**5 + 2**4 + 2**3 + 2**2][2**5 + 2**4 + 2**3 + 2**2] = 1
myOperator[2**6 + 2**5 + 2**3 + 2**0][2**6 + 2**5 + 2**3 + 2**0] = 1

myOperator[2**7-1][2**7-1] = sy.exp(sy.I*phi)
myOperator[2**5 + 2**3 + 2**1][2**5 + 2**3 + 2**1] = sy.exp(sy.I*phi)
myOperator[2**6 + 2**3 + 2**2][2**6 + 2**3 + 2**2] = sy.exp(sy.I*phi)
myOperator[2**4 + 2**3 + 2**0][2**4 + 2**3 + 2**0] = sy.exp(sy.I*phi)
myOperator[2**6 + 2**5 + 2**4][2**6 + 2**5 + 2**4] = sy.exp(sy.I*phi)
myOperator[2**5 + 2**2 + 2**0][2**5 + 2**2 + 2**0] = sy.exp(sy.I*phi)
myOperator[2**6 + 2**1 + 2**0][2**6 + 2**1 + 2**0] = sy.exp(sy.I*phi)
myOperator[2**4 + 2**2 + 2**1][2**4 + 2**2 + 2**1] = sy.exp(sy.I*phi)

A = sy.IndexedBase('A')
l = 0
for i in range(2**7):
    if(myOperator[i][i] == 0):
        myOperator[i][i] = A[l]
        l += 1
myOperator = sy.Matrix(myOperator)

#TEST IF THE GATE COMMUTES WITH STABILIZERS
stabilizersData = np.array([
    [0,0,0,3,3,3,3],
    [0,3,3,0,0,3,3],
    [3,0,3,0,3,0,3],
    [0,0,0,1,1,1,1],
    [0,1,1,0,0,1,1],
    [1,0,1,0,1,0,1]])
ZeroMatrix = np.array([[float(0) for i in range(2**7)] for j in range(2**7)])

print("Creating KL conditions...")
#<0|ZxZx|0> - <1|ZxZx|1> = 0
identity = np.array([
    [1,0],
    [0,1]])
X = np.array([
    [0,1],
    [1,0]])
Y = np.array([
    [0, -1*sy.I],
    [sy.I, 0]])
Z = np.array([
    [1, 0],
    [0, -1]])

X0 = np.kron(np.kron(np.kron(np.kron(np.kron(np.kron(X,identity),identity),identity),identity),identity),identity)
X1 = np.kron(np.kron(np.kron(np.kron(np.kron(np.kron(identity,X),identity),identity),identity),identity),identity)
X2 = np.kron(np.kron(np.kron(np.kron(np.kron(np.kron(identity,identity),X),identity),identity),identity),identity)
X3 = np.kron(np.kron(np.kron(np.kron(np.kron(np.kron(identity,identity),identity),X),identity),identity),identity)
X4 = np.kron(np.kron(np.kron(np.kron(np.kron(np.kron(identity,identity),identity),identity),X),identity),identity)
X5 = np.kron(np.kron(np.kron(np.kron(np.kron(np.kron(identity,identity),identity),identity),identity),X),identity)
X6 = np.kron(np.kron(np.kron(np.kron(np.kron(np.kron(identity,identity),identity),identity),identity),identity),X)
Y0 = np.kron(np.kron(np.kron(np.kron(np.kron(np.kron(Y,identity),identity),identity),identity),identity),identity)
Y1 = np.kron(np.kron(np.kron(np.kron(np.kron(np.kron(identity,Y),identity),identity),identity),identity),identity)
Y2 = np.kron(np.kron(np.kron(np.kron(np.kron(np.kron(identity,identity),Y),identity),identity),identity),identity)
Y3 = np.kron(np.kron(np.kron(np.kron(np.kron(np.kron(identity,identity),identity),Y),identity),identity),identity)
Y4 = np.kron(np.kron(np.kron(np.kron(np.kron(np.kron(identity,identity),identity),identity),Y),identity),identity)
Y5 = np.kron(np.kron(np.kron(np.kron(np.kron(np.kron(identity,identity),identity),identity),identity),Y),identity)
Y6 = np.kron(np.kron(np.kron(np.kron(np.kron(np.kron(identity,identity),identity),identity),identity),identity),Y)
Z0 = np.kron(np.kron(np.kron(np.kron(np.kron(np.kron(Z,identity),identity),identity),identity),identity),identity)
Z1 = np.kron(np.kron(np.kron(np.kron(np.kron(np.kron(identity,Z),identity),identity),identity),identity),identity)
Z2 = np.kron(np.kron(np.kron(np.kron(np.kron(np.kron(identity,identity),Z),identity),identity),identity),identity)
Z3 = np.kron(np.kron(np.kron(np.kron(np.kron(np.kron(identity,identity),identity),Z),identity),identity),identity)
Z4 = np.kron(np.kron(np.kron(np.kron(np.kron(np.kron(identity,identity),identity),identity),Z),identity),identity)
Z5 = np.kron(np.kron(np.kron(np.kron(np.kron(np.kron(identity,identity),identity),identity),identity),Z),identity)
Z6 = np.kron(np.kron(np.kron(np.kron(np.kron(np.kron(identity,identity),identity),identity),identity),identity),Z)



#complete set of KL conditions (still only find 103 conditions total)
expr = x
Ab = []
Elist = [X0]#, X1, X2, X3, X4, X5, X6, Y0, Y1, Y2, Y3, Y4, Y5, Y6, Z0, Z1, Z2, Z3, Z4, Z5, Z6]
Elistname = ["X0"]#, "X1", "X2", "X3", "X4", "X5", "X6", "Y0", "Y1", "Y2", "Y3", "Y4", "Y5", "Y6", "Z0", "Z1", "Z2", "Z3", "Z4", "Z5", "Z6"]
localcounter = 0
o = 0
for error1 in Elist:
    for error2 in Elist:
        errorinnerproduct = np.matmul(np.matmul(np.matrix(ZeroLogical),np.matmul(np.matrix(myOperator),np.matrix(error1))),np.matmul(np.matrix(ZeroLogical),np.matmul(np.matrix(error2),np.matrix(myOperator))).getH()).tolist()[0][0] - np.matmul(np.matmul(np.matrix(OneLogical),np.matmul(np.matrix(myOperator),np.matrix(error1))),np.matmul(np.matrix(OneLogical),np.matmul(np.matrix(error2),np.matrix(myOperator))).getH()).tolist()[0][0]
        Ab.append(errorinnerproduct)
        print(errorinnerproduct)
        #input(Ab)
        o += 1
        errorinnerproduct = np.matmul(np.matmul(np.matrix(ZeroLogical),np.matmul(np.matrix(error1),np.matrix(myOperator))),np.matmul(np.matrix(ZeroLogical),np.matmul(np.matrix(error2),np.matrix(myOperator))).getH()).tolist()[0][0] - np.matmul(np.matmul(np.matrix(OneLogical),np.matmul(np.matrix(error1),np.matrix(myOperator))),np.matmul(np.matrix(OneLogical),np.matmul(np.matrix(error2),np.matrix(myOperator))).getH()).tolist()[0][0]
        Ab.append(errorinnerproduct)
        print(errorinnerproduct)
        #input(Ab)
        o += 1
        errorinnerproduct = np.matmul(np.matmul(np.matrix(ZeroLogical),np.matmul(np.matrix(myOperator),np.matrix(error1))),np.matmul(np.matrix(ZeroLogical),np.matmul(np.matrix(myOperator),np.matrix(error2))).getH()).tolist()[0][0] - np.matmul(np.matmul(np.matrix(OneLogical),np.matmul(np.matrix(myOperator),np.matrix(error1))),np.matmul(np.matrix(OneLogical),np.matmul(np.matrix(myOperator),np.matrix(error2))).getH()).tolist()[0][0]
        Ab.append(errorinnerproduct)
        print(errorinnerproduct)
        #input(Ab)
        
        print("Completed Error "+str(Elistname[localcounter%len(Elist)])+" "+str(Elistname[math.floor(localcounter/len(Elist))]))
        localcounter += 1
        o += 1
#"""
print()
Ab = sy.Tuple.fromiter(Ab)
input(Ab)

solution = sy.solve(Ab,Ab.atoms(sy.Indexed))
print(solution)

global currentmatrix
global Unitary

Unitary = [[expr.subs(x,0) for i in range(2**7)] for j in range(2**7)]

Unitary[0][0] = 1
Unitary[2**6 + 2**4 + 2**2 + 2**0][2**6 + 2**4 + 2**2 + 2**0] = 1
Unitary[2**5 + 2**4 + 2**1 + 2**0][2**5 + 2**4 + 2**1 + 2**0] = 1
Unitary[2**6 + 2**5 + 2**2 + 2**1][2**6 + 2**5 + 2**2 + 2**1] = 1
Unitary[2**3 + 2**2 + 2**1 + 2**0][2**3 + 2**2 + 2**1 + 2**0] = 1
Unitary[2**6 + 2**4 + 2**3 + 2**1][2**6 + 2**4 + 2**3 + 2**1] = 1
Unitary[2**5 + 2**4 + 2**3 + 2**2][2**5 + 2**4 + 2**3 + 2**2] = 1
Unitary[2**6 + 2**5 + 2**3 + 2**0][2**6 + 2**5 + 2**3 + 2**0] = 1

Unitary[2**7-1][2**7-1] = sy.exp(sy.I*phi)
Unitary[2**5 + 2**3 + 2**1][2**5 + 2**3 + 2**1] = sy.exp(sy.I*phi)
Unitary[2**6 + 2**3 + 2**2][2**6 + 2**3 + 2**2] = sy.exp(sy.I*phi)
Unitary[2**4 + 2**3 + 2**0][2**4 + 2**3 + 2**0] = sy.exp(sy.I*phi)
Unitary[2**6 + 2**5 + 2**4][2**6 + 2**5 + 2**4] = sy.exp(sy.I*phi)
Unitary[2**5 + 2**2 + 2**0][2**5 + 2**2 + 2**0] = sy.exp(sy.I*phi)
Unitary[2**6 + 2**1 + 2**0][2**6 + 2**1 + 2**0] = sy.exp(sy.I*phi)
Unitary[2**4 + 2**2 + 2**1][2**4 + 2**2 + 2**1] = sy.exp(sy.I*phi)

theoretical = True
l = 0
for i in range(2**7):
    if(Unitary[i][i] == 0):
        Unitary[i][i] = solution.args[0][l]#.subs([(var101, -var108),(var102, -var107),(var103,-var106)])#SEE EXPRESSION FOR GATES as a function of free variables CCZ(var110,var109...)
        l += 1
Unitary = sy.Matrix(Unitary)
identity = np.array([
    [1,0],
    [0,1]])
currentmatrix = sy.Matrix(np.kron(np.kron(np.kron(np.kron(np.kron(np.kron(identity,identity),identity),identity),identity),identity),identity))

global chargates 
chargates = []

def makeqiskitgate(curbinary, i):
    global currentmatrix
    global Unitary
    global chargates
    global myphi
    global Unitarysubbed
    mybits = []
    identity = np.array([
        [1,0],
        [0,1]])
    while(len(curbinary) < 7):
        curbinary = "0"+curbinary
    zflag = False
    workingmatrix = np.array([1])
    for j in range(len(curbinary)):
        if(curbinary[j] == "1"):
            if(zflag==False):
                #need to make current diagonal entry equal to desired
                nextmatrix = np.array([[0, 0], #Z((Unitary*currentmatrix.H)[i,i])
                 [0, 1-sy.simplify((Unitarysubbed*currentmatrix.H)[i,i])]])
                workingmatrix = np.kron(workingmatrix,nextmatrix)
                zflag = True
                chargates.append("(Z_"+str(j)+"="+str(sy.simplify((Unitarysubbed*currentmatrix.H)[i,i]))+")")
                mybits.append(j)
            else: #kronecker product a control signal
                nextmatrix = np.array([
                    [0,0],
                    [0,1]])
                workingmatrix = np.kron(workingmatrix, nextmatrix)
                chargates[len(chargates)-1] = chargates[len(chargates)-1] + "C_"+str(j)
                mybits.append(j)
        else: # kronecker product an identity
            workingmatrix = np.kron(workingmatrix, identity)
    
    workingmatrix = sy.Matrix(np.kron(np.kron(np.kron(np.kron(np.kron(np.kron(identity,identity),identity),identity),identity),identity),identity)) - sy.Matrix(workingmatrix)
    
    if(len(mybits) > 1):
        nextgate = qs.circuit.library.PhaseGate(float(sy.simplify(-sy.I*sy.ln((Unitarysubbed*currentmatrix.H)[i,i])).subs(phi,myphi).evalf())).control(len(mybits)-1)
    else:
        nextgate = qs.circuit.library.PhaseGate(float(sy.simplify(-sy.I*sy.ln((Unitarysubbed*currentmatrix.H)[i,i])).subs(phi,myphi).evalf()))

    #return identity minus calculated matrix   
    #return sy.Matrix(np.kron(identity,identity)) - sy.Matrix(workingmatrix)
    
    # 7 qubit return   
    return [nextgate, mybits, sy.Matrix(workingmatrix)]

def makegate(curbinary, i):
    global currentmatrix
    global Unitary
    global chargates
    global myphi
    identity = np.array([
        [1,0],
        [0,1]])
    while(len(curbinary) < 7):
        curbinary = "0"+curbinary
    zflag = False
    workingmatrix = np.array([1])
    for j in range(len(curbinary)):
        if(curbinary[j] == "1"):
            if(zflag==False):
                #need to make current diagonal entry equal to desired
                nextmatrix = np.array([[0, 0], #Z((Unitary*currentmatrix.H)[i,i])
                 [0, 1-sy.simplify((Unitary*currentmatrix.H)[i,i])]])
                workingmatrix = np.kron(workingmatrix,nextmatrix)
                zflag = True
                chargates.append("(Z_"+str(j)+"="+str(sy.simplify((Unitary*currentmatrix.H)[i,i]))+")")
            else: #kronecker product a control signal
                nextmatrix = np.array([
                    [0,0],
                    [0,1]])
                workingmatrix = np.kron(workingmatrix, nextmatrix)
                chargates[len(chargates)-1] = chargates[len(chargates)-1] + "C_"+str(j)
        else: # kronecker product an identity
            workingmatrix = np.kron(workingmatrix, identity)
    

    
    #return identity minus calculated matrix   
    #return sy.Matrix(np.kron(identity,identity)) - sy.Matrix(workingmatrix)
    
    # 7 qubit return   
    return sy.Matrix(np.kron(np.kron(np.kron(np.kron(np.kron(np.kron(identity,identity),identity),identity),identity),identity),identity)) - sy.Matrix(workingmatrix)


"""#BRUTE FORCE CHECKS THE FREE VARIABLES
myphases = [1, phi]
qc = qs.QuantumCircuit(7)
varleft = [var98, var99, var100, var101, var102, var103, var106, var107, var108, var109, var110, var111]
numgates = 1000
numgatesindex = [0 for i in range(len(varleft))]
print("Searching for smallest circuit...")
for j in range(len(myphases)**len(varleft)):
    mybinary = format(j, 'b')
    while(len(mybinary) < len(varleft)):
        mybinary = "0"+mybinary
    localUnitary = Unitary.subs([(varleft[0], myphases[int(mybinary[0])]),(varleft[1], myphases[int(mybinary[1])]),(varleft[2], myphases[int(mybinary[2])]),(varleft[3], myphases[int(mybinary[3])]),(varleft[4], myphases[int(mybinary[4])]),(varleft[5], myphases[int(mybinary[5])]),(varleft[6], myphases[int(mybinary[6])]),(varleft[7], myphases[int(mybinary[7])]),(varleft[8], myphases[int(mybinary[8])]),(varleft[9], myphases[int(mybinary[9])]),(varleft[10], myphases[int(mybinary[10])]),(varleft[11], myphases[int(mybinary[11])]),])
    print(mybinary)
    localnumgates = 0
    for i in range(2**7):
        if(Unitary[i,i] != currentmatrix[i,i]):
            #Use i to find necessary gate(from binary decomp)(01101 is c2c3z5)
            curbinary = format(i,'b')
            newMatrix = makegate(curbinary, i)
            currentmatrix = currentmatrix * newMatrix
            localnumgates += 1
    if(localnumgates < numgates):
        print(localnumgates)
        numgates = localnumgates
        numgatesindex = mybinary
            
print(curbinary)
#"""

qc = qs.QuantumCircuit(7)
print("ASSEMBLING CIRCUIT...")
global Unitarysubbed

#NEED TO SUB I INDEXED SYMPY VARIABLES (A[1] rather than var1)
Unitarysubbed = Unitary.subs([(var101, -var108),(var102, -var107),(var103,-var106)]).subs([(var98, -1),(var99, 1),(var100, -1),(var106, -1),(var107, 1),(var108, -1),(var109, 1),(var110, -1),(var111, 1)]).evalf()
for i in range(2**7):
    if(Unitarysubbed[i,i] != currentmatrix[i,i]):
        #Use i to find necessary gate(from binary decomp)(01101 is c2c3z5)
        curbinary = format(i,'b')
        [nextgate, mybits, newMatrix] = makeqiskitgate(curbinary, i)
        qc.append(nextgate, mybits)
        currentmatrix = currentmatrix * newMatrix
        
print(chargates)
print("gates(phi) ^^\n")



#Elist = [X0, X1, X2, X3, X4, X5, X6, Y0, Y1, Y2, Y3, Y4, Y5, Y6, Z0, Z1, Z2, Z3, Z4, Z5, Z6]
#Elistname = ["X0", "X1", "X2", "X3", "X4", "X5", "X6", "Y0", "Y1", "Y2", "Y3", "Y4", "Y5", "Y6", "Z0", "Z1", "Z2", "Z3", "Z4", "Z5", "Z6"]
Elist = [X3]
Elistname = ["X3"]
localcounter = 0
for error1 in Elist:
    for error2 in Elist:
        print()
        print(np.matmul(np.matmul(np.matrix(ZeroLogical),np.matmul(np.matrix(myOperator),np.matrix(error1))),np.matmul(np.matrix(ZeroLogical),np.matmul(np.matrix(error2),np.matrix(myOperator))).getH()).tolist()[0][0] - np.matmul(np.matmul(np.matrix(OneLogical),np.matmul(np.matrix(myOperator),np.matrix(error1))),np.matmul(np.matrix(OneLogical),np.matmul(np.matrix(error2),np.matrix(myOperator))).getH()).tolist()[0][0])
        print(np.matmul(np.matmul(np.matrix(ZeroLogical),np.matmul(np.matrix(Unitary),np.matrix(error1))),np.matmul(np.matrix(ZeroLogical),np.matmul(np.matrix(error2),np.matrix(Unitary))).getH()).tolist()[0][0] - np.matmul(np.matmul(np.matrix(OneLogical),np.matmul(np.matrix(Unitary),np.matrix(error1))),np.matmul(np.matrix(OneLogical),np.matmul(np.matrix(error2),np.matrix(Unitary))).getH()).tolist()[0][0])
        print(np.matmul(np.matmul(np.matrix(ZeroLogical),np.matmul(np.matrix(Unitarysubbed),np.matrix(error1))),np.matmul(np.matrix(ZeroLogical),np.matmul(np.matrix(error2),np.matrix(Unitarysubbed))).getH()).tolist()[0][0] - np.matmul(np.matmul(np.matrix(OneLogical),np.matmul(np.matrix(Unitarysubbed),np.matrix(error1))),np.matmul(np.matrix(OneLogical),np.matmul(np.matrix(error2),np.matrix(Unitarysubbed))).getH()).tolist()[0][0])
        print()
        print(np.matmul(np.matmul(np.matrix(ZeroLogical),np.matmul(np.matrix(error1),np.matrix(myOperator))),np.matmul(np.matrix(ZeroLogical),np.matmul(np.matrix(error2),np.matrix(myOperator))).getH()).tolist()[0][0] - np.matmul(np.matmul(np.matrix(OneLogical),np.matmul(np.matrix(error1),np.matrix(myOperator))),np.matmul(np.matrix(OneLogical),np.matmul(np.matrix(error2),np.matrix(myOperator))).getH()).tolist()[0][0])
        print(np.matmul(np.matmul(np.matrix(ZeroLogical),np.matmul(np.matrix(error1),np.matrix(Unitary))),np.matmul(np.matrix(ZeroLogical),np.matmul(np.matrix(error2),np.matrix(Unitary))).getH()).tolist()[0][0] - np.matmul(np.matmul(np.matrix(OneLogical),np.matmul(np.matrix(error1),np.matrix(Unitary))),np.matmul(np.matrix(OneLogical),np.matmul(np.matrix(error2),np.matrix(Unitary))).getH()).tolist()[0][0])
        print(np.matmul(np.matmul(np.matrix(ZeroLogical),np.matmul(np.matrix(error1),np.matrix(Unitarysubbed))),np.matmul(np.matrix(ZeroLogical),np.matmul(np.matrix(error2),np.matrix(Unitarysubbed))).getH()).tolist()[0][0] - np.matmul(np.matmul(np.matrix(OneLogical),np.matmul(np.matrix(error1),np.matrix(Unitarysubbed))),np.matmul(np.matrix(OneLogical),np.matmul(np.matrix(error2),np.matrix(Unitarysubbed))).getH()).tolist()[0][0])
        print()
        print(np.matmul(np.matmul(np.matrix(ZeroLogical),np.matmul(np.matrix(myOperator),np.matrix(error1))),np.matmul(np.matrix(ZeroLogical),np.matmul(np.matrix(myOperator),np.matrix(error2))).getH()).tolist()[0][0] - np.matmul(np.matmul(np.matrix(OneLogical),np.matmul(np.matrix(myOperator),np.matrix(error1))),np.matmul(np.matrix(OneLogical),np.matmul(np.matrix(myOperator),np.matrix(error2))).getH()).tolist()[0][0])
        print(np.matmul(np.matmul(np.matrix(ZeroLogical),np.matmul(np.matrix(Unitary),np.matrix(error1))),np.matmul(np.matrix(ZeroLogical),np.matmul(np.matrix(Unitary),np.matrix(error2))).getH()).tolist()[0][0] - np.matmul(np.matmul(np.matrix(OneLogical),np.matmul(np.matrix(Unitary),np.matrix(error1))),np.matmul(np.matrix(OneLogical),np.matmul(np.matrix(Unitary),np.matrix(error2))).getH()).tolist()[0][0])
        print(np.matmul(np.matmul(np.matrix(ZeroLogical),np.matmul(np.matrix(Unitarysubbed),np.matrix(error1))),np.matmul(np.matrix(ZeroLogical),np.matmul(np.matrix(Unitarysubbed),np.matrix(error2))).getH()).tolist()[0][0] - np.matmul(np.matmul(np.matrix(OneLogical),np.matmul(np.matrix(Unitarysubbed),np.matrix(error1))),np.matmul(np.matrix(OneLogical),np.matmul(np.matrix(Unitarysubbed),np.matrix(error2))).getH()).tolist()[0][0])
        print("Error X3X3 not "+str(Elistname[localcounter%len(Elist)])+" "+str(Elistname[math.floor(localcounter/len(Elist))])+"\n")
        localcounter += 1
input("ALL KL Condtions Calculated")

print(qc)
print("qiskit gates with phi = "+str(myphi))

mygate = qs.quantum_info.Operator(np.array(Unitarysubbed.subs(phi,myphi)))

"""#example of gate decomposition algorithm
currentmatrix = sy.Matrix(np.kron(identity,identity))

Unitary = sy.Matrix([
    [1,  0,  0,  0],
    [0,  sy.exp(sy.I*a),  0,  0],
    [0,  0,  sy.exp(sy.I*a),  0],
    [0,  0,  0, sy.exp(sy.I*a)]
])

for i in range(2**2):
    if(Unitary[i,i] != currentmatrix[i,i]):
        #Use i to find necessary gate(from binary decomp)(01101 is c2c3z5)
        curbinary = format(i,'b')
        newMatrix = makegate(curbinary, i)
        print()
        print(newMatrix)
        currentmatrix = currentmatrix * newMatrix
        print(currentmatrix)
        print(chargates)
        input(Unitary)
input("finished")
"""


"""
l = 0
for stabilizer in stabilizers:
    l += 1
    localflag = False
    commutator = np.matmul(np.matrix(mygate), np.matrix(stabilizer)) - np.matmul(np.matrix(stabilizer), np.matrix(mygate)).tolist()
    for i in range(len(commutator)):
        for j in range(len(commutator[i].tolist()[0])):
            if(round(commutator[i].tolist()[0][j].real, 5) != 0 and round(commutator[i].tolist()[0][j].imag, 5) != 0):
                localflag = True
                #print("problem, "+str(commutator[i].tolist()[0][j])+" != 0,     i="+str(i)+", j="+str(j)+"\n")
    if(localflag):
        print("Logical Rotation Does Not Commute With Operator "+str(l)+"\n"+str(stabilizersData[l-1]))
#"""
"""#TEST OVERLAPS AND OUTPUT STATE
qc.save_statevector()
sv_job = sv_sim.run(qc)
state = sv_job.result().get_statevector().data
ZeroOverlap = np.matmul(np.matrix(np.array(ZeroLogical)), np.matrix(np.array(state)).getH()) #<0|psi>
OneOverlap = np.matmul(np.matrix(np.array(OneLogical)), np.matrix(np.array(state)).getH()) #<1|psi>
PlusOverlap = np.matmul(np.matrix((np.array(ZeroLogical)+np.array(OneLogical))/np.sqrt(2)), np.matrix(np.array(state)).getH()) #<+|psi>
MinusOverlap = np.matmul(np.matrix((np.array(ZeroLogical)-np.array(OneLogical))/np.sqrt(2)), np.matrix(np.array(state)).getH()) #<-|psi>
PlusIOverlap = np.matmul(np.matrix((np.array(ZeroLogical)+np.array(OneLogical)*1j)/np.sqrt(2)), np.matrix(np.array(state)).getH()) #<+i|psi>
MinusIOverlap = np.matmul(np.matrix((np.array(ZeroLogical)-np.array(OneLogical)*1j)/np.sqrt(2)), np.matrix(np.array(state)).getH()) #<-i|psi>
print("|<0|psi>|^2 = "+str(ZeroOverlap*np.conjugate(ZeroOverlap))+"\n")
print("|<1|psi>|^2 = "+str(OneOverlap*np.conjugate(OneOverlap))+"\n")
print("|<+|psi>|^2 = "+str(PlusOverlap*np.conjugate(PlusOverlap))+"\n")
print("|<-|psi>|^2 = "+str(MinusOverlap*np.conjugate(MinusOverlap))+"\n")
print("|<+i|psi>|^2 = "+str(PlusIOverlap*np.conjugate(PlusIOverlap))+"\n")
print("|<-i|psi>|^2 = "+str(MinusIOverlap*np.conjugate(MinusIOverlap))+"\n")

#aer_job = aer_sim.run(qc, shots=100)
#print(aer_job.result().get_counts())
#"""

qubitorder = [0,1,2,3,4,5,6]

KLZeroStates = []#[ZeroLogical]
KLOneStates = []#[OneLogical]

logicalStates = [ZeroLogical, OneLogical]
for i in range(42):
    qc = qs.QuantumCircuit(7)

    qc.initialize(logicalStates[0], qc.qubits)

    if(i<21):
        if(i<7):
            qc.x(i)
        if(i>=7 and i < 14):
            qc.z(i%7)
        if(i>=14):
            qc.y(i%7)

    qc.unitary(mygate, qubitorder)

    if(i >= 21 and i < 42):
        if(i<28):
            qc.x(i%7)
        if(i>=28 and i < 35):
            qc.z(i%7)
        if(i>=35):
            qc.y(i%7)

    sv_sim = qs.Aer.get_backend('aer_simulator')

    qc.save_statevector()
    job = sv_sim.run(qc)

    state = job.result().get_statevector().data
    #print(state)
    #input(qc)
    currentState = np.array(state)
    if(0==0):
        KLZeroStates.append(currentState)
    elif(0==1):
        KLOneStates.append(currentState)
    #########################################################################Getting rotated Z_L|0>
qc = qs.QuantumCircuit(7)
qc.initialize(logicalStates[0], qc.qubits)

qc.unitary(mygate, qubitorder)

sv_sim = qs.Aer.get_backend('aer_simulator')

qc.save_statevector()
job = sv_sim.run(qc)

state = job.result().get_statevector().data
#print(state)
#input(qc)
currentState = np.array(state)
if(0==0):
    KLZeroStates.append(currentState)
elif(0==1):
    KLOneStates.append(currentState)
for i in range(42):
    qc = qs.QuantumCircuit(7)

    qc.initialize(logicalStates[1], qc.qubits)

    if(i<21):
        if(i<7):
            qc.x(i)
        if(i>=7 and i < 14):
            qc.z(i%7)
        if(i>=14):
            qc.y(i%7)

    qc.unitary(mygate, qubitorder)

    if(i >= 21 and i < 42):
        if(i<28):
            qc.x(i%7)
        if(i>=28 and i < 35):
            qc.z(i%7)
        if(i>=35):
            qc.y(i%7)

    sv_sim = qs.Aer.get_backend('aer_simulator')

    qc.save_statevector()
    job = sv_sim.run(qc)

    state = job.result().get_statevector().data
    #print(state)
    #input(qc)
    currentState = np.array(state)
    if(1==0):
        KLZeroStates.append(currentState)
    elif(1==1):
        KLOneStates.append(currentState)
    #########################################################################Getting rotated Z_L|0>
qc = qs.QuantumCircuit(7)
qc.initialize(logicalStates[1], qc.qubits)

qc.unitary(mygate, qubitorder)

sv_sim = qs.Aer.get_backend('aer_simulator')

qc.save_statevector()
job = sv_sim.run(qc)

state = job.result().get_statevector().data
#print(state)
#input(qc)
currentState = np.array(state)
if(1==0):
    KLZeroStates.append(currentState)
elif(1==1):
    KLOneStates.append(currentState)

flag = True
l = 0
k = 0
for zerostate in KLZeroStates:
    for onestate in KLOneStates:
        if(np.matmul(np.matrix(zerostate), np.matrix(onestate).getH()) > 0):
            flag = False
            print()
            print("UH OH  l="+str(l)+", k="+str(k))
            input(np.matmul(np.matrix(zerostate), np.matrix(onestate).getH()))
            #print(zerostate)
            #print(onestate)
        k += 1
    l += 1
for l in range(len(KLZeroStates)):
    for m in range(len(KLOneStates)):
        num1 = np.matmul(np.matrix(KLZeroStates[l]), np.matrix(KLZeroStates[m]).getH()).tolist()[0][0]
        num1 = round(num1.real, 5) + round(num1.imag, 5) * 1j
        num2 = np.matmul(np.matrix(KLOneStates[l]), np.matrix(KLOneStates[m]).getH()).tolist()[0][0]
        num2 = round(num2.real, 5) + round(num2.imag, 5) * 1j
        if(num1 != num2):
            print("Uh Oh!! Unequal "+str(l)+", "+str(m))
            print(str(num1)+" != "+str(num2)+"\n")
            flag = False
#print(len(KLZeroStates))
if(flag):
    print("NONE Found!!")  
else:
    print("FAILED")  
print("END")