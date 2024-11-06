# https://github.com/AlexQkit/AlexQkit/blob/72e77094c290a4c0ab524e2541fb929fba6a4275/server/results.py
from qiskit import Aer
from qiskit import IBMQ
from qiskit import execute
from qiskit.quantum_info import partial_trace
from qiskit.visualization import plot_bloch_vector
        
import io
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


class Results():

    def __init__(self,circuit):
        self.defaultBlochSphere=self.figToResponse(plot_bloch_vector([0,0,1]))
        self.circuit=circuit
        self.num_qubits=self.circuit.num_qubits
        #self.circutDrawing=self.draw()
        self.statevector=self.stateVector()
        self.reversedStatevector=self.reversedStateVector()
        
###############################################################################################################################   
        
    def setter(self,shots,API_TOKEN,device,circuit):
        self.shots=int(shots)
        self.API_TOKEN=API_TOKEN
        self.device=device
        self.circuit=circuit
        self.num_qubits=self.circuit.num_qubits
        #self.circutDrawing=self.draw()
        self.statevector=self.stateVector()
        self.reversedStatevector=self.reversedStateVector()
        
###############################################################################################################################   
    
    def setCircuit(self,circuit):
        self.circuit=circuit
        self.num_qubits=self.circuit.num_qubits
        #self.circutDrawing=self.draw()
        self.statevector=self.stateVector()
        self.reversedStatevector=self.reversedStateVector()
        
###############################################################################################################################  

    def stateVector(self):    
        """
        returns the state vector of the circuit
        """
        simulator=Aer.get_backend('statevector_simulator')
        result=execute(self.circuit,backend=simulator).result()
        statevector=result.get_statevector(decimals=4) #"decimals=4" doesn't work in version 0.20.0 
        return statevector.tolist()
    
###############################################################################################################################
           
    def reversedStateVector(self):
        """
        returns the state vector for the reversed wires 
        ex.. factor of |001⟩ will be the factor of |100⟩ 
        
        to correct qiskit convention
        according to Qiskit’s convention, first qubit is on the right-hand side
        ex: |01⟩  .. 1st qubit is 1 and 2nd qubit is 0
        """
        reversedStatevector=[]
        for i in range(len(self.statevector)):
            #the next line finds the inverse position in decimal ex.. 3 -> 011 -> 110 -> 6
            pos=int(''.join(reversed(str(("{0:0"+str(self.num_qubits).replace('.0000','')+"b}").format(i)))),2)
            reversedStatevector.append(round(self.statevector[pos].real, 4) + round(self.statevector[pos].imag, 4)*1j)
        return reversedStatevector
        
###############################################################################################################################

    def numberFormat(self,num,isImag=False):
        """
        to enhance dirac notation and matrix representation numbers
        """
        string=str(num)
        if num!=0:
            if num>0:
                string="+" if num==1 else "+"+string
            else:
                string="-" if num==-1 else string
            return string+"i" if isImag else string
        return ""

###############################################################################################################################
        
    def diracNotation(self):
        """
        returns dirac notation of the circuit
        neglects terms with zero probability
        four digits after floating point
        """
        diracNotation=""
        for i in range(len(self.reversedStatevector)):
            if self.reversedStatevector[i]==0:
                continue
            diracNotation+=self.numberFormat(self.reversedStatevector[i].real)
            diracNotation+=self.numberFormat(self.reversedStatevector[i].imag,True)
            #the next line generates the state .. ex circuit with 3 wires -> i=2 => state:010
            diracNotation+="|"+str(("{0:0"+str(self.num_qubits).replace('.0000','')+"b}").format(i))+"⟩ " 
        return diracNotation.lstrip("+")

###############################################################################################################################
        
    def matrixRepresentation(self,decimals=8):
        """
        returns <str> readable matrix representation of the whole system
        including the initialization gates
        
        four digits after floating point
        
        circuit mustn't be measured
        """
        temp = self.circuit.copy()
        temp.remove_final_measurements()
    
        simulator = Aer.get_backend('unitary_simulator')
        result = execute(temp, backend=simulator).result()
        unitary = result.get_unitary(decimals=decimals).tolist()
        for i in range(len(unitary)):
            for j in range(len(unitary[i])):
                if unitary[i][j]==0:
                    unitary[i][j]="0"
                else:
                    string=str(unitary[i][j].real).replace(".0", "")
                    string="" if unitary[i][j].real==0 else string
                    string+=self.numberFormat(unitary[i][j].imag,True)
                    unitary[i][j]=string.lstrip("+")
        return unitary
        
###############################################################################################################################

    def separatedProbabilities(self):
        """
        returns probability of |1⟩ for every wire in a list
        
        rule.. probability(qubit(i)==1)= sum( probability(state with qubit(i)==1))
        
        ex.. system= 1/√2 |10⟩ + 1/√2 |11⟩
             prob( |1*⟩ ) = prob(1st qubit == 1)
                          = sum( prob( states[qubit(0)]==1 ) )
                          = prob(|10⟩) + prob(|11⟩)
                          = (1/√2)**2  +  (1/√2)**2
                          = 1
                          means the 1st qubit will always be true 
        """
        res = []
        for j in range(self.num_qubits):
            val = 0
            for i in range(len(self.statevector)):
                #generating the state .. ex circuit with 3 wires -> i=2 => state:010
                pos = str(("{0:0"+str(self.num_qubits).replace('.0000', '')+"b}").format(i)) 
                if pos[j] == '1':
                    val += abs(self.statevector[i])**2
            val = round(val*100, 1)
            res.insert(0, val)
        return res

###############################################################################################################################

    def graph(self):
        """
        returns the probability of state as 2D list [ [state, probability], ... ] to be presented on the chart
        probability of a state = its count / total number of shots
        neglects zero probabilities
        
        we apply measurments to all wires to prevent losing counts
        unmeasured circuits will return all counts as zeros
        
        measurments in reverse to correct qiskit convention
        according to Qiskit’s convention, first qubit is on the right-hand side
        ex: |01⟩  .. 1st qubit is 1 and 2nd qubit is 0
        """
        temp = self.circuit.copy()
        temp.measure(list(range(self.num_qubits)),list(range(self.num_qubits-1,-1,-1)))
        simulator = Aer.get_backend('qasm_simulator')
        result = execute(temp, backend=simulator,shots=self.shots).result()
        counts = result.get_counts(temp)
        #the next line to convert dictionary {state:counts} to list [ [state, counts/shots ] ]
        #we need that representation of data to display it on google charts vue component
        graphData=[["|"+state+"⟩",count/self.shots] for state,count in counts.items()]
        return graphData

###############################################################################################################################       
   
    def figToResponse(self,fig):
        """
        converts figure type to response object
        """
        output = io.BytesIO()
        FigureCanvas(fig).print_png(output)
        return Response(output.getvalue(), mimetype='image/png')
 
############################################################################################################################### 
        
    def draw(self):
        """
        returns image of the circuit as response object
        """
        fig=self.circuit.draw(output='mpl')
        return self.figToResponse(fig)
    
###############################################################################################################################
        
    def separatedBlochSpheres(self):
        """
        returns list of response object for the bloch sphere of every wire
        """
        pos=list(range(self.num_qubits))
        res={}
        for i in range(self.num_qubits):
            #density matrix of the wire
            [[a, b], [c, d]] = partial_trace(self.statevector, pos[:i]+pos[i+1:]).data 
            #polar coordinate
            x = 2*b.real
            y = 2*c.imag
            z = a.real-d.real
            #blochSphere figure
            fig=plot_bloch_vector([x,y,z])#,title="qubit "+str(i)
            #appending response object of the figure
            res[i]=self.figToResponse(fig)
        return res

###############################################################################################################################

    def runOnIBMQ(self):
        """
        runs a circuit on a real quantum computer (IBM Q devices) and returns the link of the results
        """
        IBMQ.save_account(self.API_TOKEN)
        IBMQ.load_account()
        provider = IBMQ.get_provider('ibm-q')
        qcomp = provider.get_backend(self.device)
        job = execute(self.circuit, backend=qcomp, shots=self.shots)
        return "https://quantum-computing.ibm.com/results/"+job.job_id()

############################################################################################################################### 
        
    def draggableCircuitResults(self):
        """
        returns all draggable circuit results in a dictionary
        """
        returnedDictionary={}
        self.blochSpheres=self.separatedBlochSpheres()
        returnedDictionary["probabilities"] = self.separatedProbabilities()
        #returnedDictionary["blochSpheres"] = self.separatedBlochSpheres()
        returnedDictionary["diracNotation"] = self.diracNotation()
        returnedDictionary["link"] = ""
        returnedDictionary['chart'] = self.graph()
        try:
            returnedDictionary["qasm"] = self.circuit.qasm()
        except Exception:
            #str(Exception)
            returnedDictionary["qasm"] = "//You are using custom gate\n//with size more than 2 qubits\n//sorry, this version doesn't support that\n//qiskit version 0.19.1"
            
        if self.API_TOKEN != "":
            returnedDictionary["link"] = self.runOnIBMQ()
        
        return returnedDictionary
    
###############################################################################################################################
    
    def qasmCircuitResults(self):
        """
        returns all qasm circuit results in a dictionary
        """
        returnedDictionary={}
        self.circutDrawing = self.draw()
        self.blochSpheres=self.separatedBlochSpheres()
        returnedDictionary["wires"]=self.num_qubits
        returnedDictionary["probabilities"] = self.separatedProbabilities()
        #returnedDictionary["blochSpheres"] = self.separatedBlochSpheres()
        returnedDictionary["diracNotation"] = self.diracNotation()
        returnedDictionary['chart'] = self.graph()
        returnedDictionary["link"] = ""
        #returnedDictionary["qasmRows"] = np.transpose(cols).tolist()
        
        if self.API_TOKEN != "":
            returnedDictionary["link"] = self.runOnIBMQ()
        
        return returnedDictionary
    
############################################################################################################################### 