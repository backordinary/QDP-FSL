# https://github.com/Karen-Shekyan/Variational-Quantum-Eigensolver/blob/a28525f7c5ea8bafb318044eeddcd26ad02b4522/qiskit%20code%20(Python)/main.py
import numpy as np
from qiskit.algorithms.optimizers import COBYLA
from qiskit.algorithms import VQE
from qiskit_nature.algorithms import (GroundStateEigensolver,
                                      NumPyMinimumEigensolverFactory)
from qiskit_nature.drivers import Molecule
from qiskit_nature.drivers.second_quantization import (
    ElectronicStructureMoleculeDriver, ElectronicStructureDriverType)
from qiskit_nature.transformers.second_quantization.electronic import FreezeCoreTransformer
from qiskit_nature.problems.second_quantization import ElectronicStructureProblem
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import ParityMapper
import numpy as np
from qiskit_nature.circuit.library import UCCSD, HartreeFock
from qiskit.circuit.library import EfficientSU2
from qiskit.algorithms.optimizers import COBYLA, SPSA, SLSQP
from qiskit.opflow import TwoQubitReduction
from qiskit import BasicAer, Aer
from qiskit.utils import QuantumInstance
from qiskit.utils.mitigation import CompleteMeasFitter
from qiskit.providers.aer.noise import NoiseModel
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute, Aer
from qiskit.utils import QuantumInstance
from qiskit.opflow import PauliExpectation, CircuitSampler, StateFn, X, Y, Z, I, CircuitStateFn
from qiskit import IBMQ 

#Input:
    #params - numpy array containing parameters to be optimized
#Output:
    #returns a numpy array containing optimized parameters
def optimizeParams(params=np.array([])):
    optimizer = COBYLA(maxiter=500, tol=0.0001)
    result = optimizer.minimize(fun=objective_function, x0=params)
    return result.x

#Input:
    #electrons - number of electrons in simulated molecule
    #orbitals - number of orbitals in simulated molecule. This is also the size of the
    #quantum register initialized
    #circuit - quantum circuit in which the Hartree-Fock state will be initialized
#Outcome:
    #creates a quantum register in the Hartree-Fock state and adds it to the circuit
def getHartreeFockState(electrons, circuit, register):
    for i in range(electrons):
        circuit.x(register[i])
    return

#function to change bool String to integer
def boolStringToInt(array):
    integer = 0
    for i in range(len(array)):
        if (array[i] == "1"):
            integer += 2**i
    return integer

# Measures the expectation value more effeciently by grouping certain Paulis
def rotateAndMeasure(pauliOperator, stateFunc):
    # as of now, pauliOperator should be type PauliSumOp from qiskit
    # stateFunc is the circuit that represents ansatz
    
    op = pauliOperator
    state = stateFunc
    state = CircuitStateFn(state)
    
    backend = Aer.get_backend('aer_simulator') 
    qInstance = QuantumInstance(backend, shots=1024)
    measurableExp = StateFn(op, is_measurement=True).compose(state) 
    
    expectationVal = PauliExpectation().convert(measurableExp)
    sampler = CircuitSampler(qInstance).convert(expectationVal) 
    
    return sampler.eval().real  

def VQE(electrons, orbitals, standardError, thetaArray, pauliOperator):
    #thetaArray has orbitals*2 rows
    #target standard error is given to function
    #function should return expectationvalue for array of angles
    register = QuantumRegister(orbitals)
    circuit = QuantumCircuit(register)
    getHartreeFockState(electrons, circuit, register)
    expectationValue = 0.00
    measured = []
    lastMin = 0
    minExp = 1
    
    while (lastMin < 100):
        for j in range(orbitals**4):
            for l in range(round((1/standardError)**2.0)):
                print(l)
                #prepare state function of theta[i]
                for i in range(orbitals):
                    # print(i)
                    circuit.rx(thetaArray[2*i], register[i])
                    circuit.ry(thetaArray[2*i+1], register[i])

                #entangled ansatz states preparation
                for k in range(len(register) - 1):
                    circuit.cx(register[k], register[k+1])
                
                #measures expectation value, adds to list, compares with min
                exp = rotateAndMeasure(pauliOperator, circuit)
                measured.append(exp)
                if exp < minExp:
                    minExp = exp
                    lastMin = 0
                else:
                    lastMin += 1
            print("\n")
            print(j)
        #gets rid of all expectation values after the minimum value
        upToMin = measured[:len(measured)-lastMin]
    
        #adds up the remaining expectation values
        expectationValue = sum(upToMin)
        thetaArray = optimizeParams(thetaArray)
        print("\n")
        print(lastMin)
    return expectationValue

# backup measeurement algorithm for energy
def measureExpectationValue(array, standardError, circuit):
    totalH = 0.00
    counter = 0

    #simulation of measurement results
    measurement = ClassicalRegister(len(array))
    circuit.measure(input, measurement)
    simulator = provider.get_backend('simulator_stabilizer')

    #runs 1/standard error squared times for standard error given
    simulation = execute(circuit, simulator, shots=(1/standardError)^2)
    mresult = simulation.result()
    counts = mresult.get_counts(circuit)
    for(measured_state, count) in counts.items():
        counter += count
        intM = boolStringToInt(measured_state)
        totalH += intM * count
    #Find expectationValue for given set of Pauli Strings for energies                
    expectationValue = totalH / counter   
    return expectationValue 



##################################################################################################
#                                     HELPER METHODS BENEATH                                     #
##################################################################################################

np.random.seed(999999)
p0 = np.random.random()
target_distr = {0: p0, 1: 1-p0}
provider = IBMQ.load_account()
backend = provider.get_backend('simulator_stabilizer')

def get_var_form(params):
    qr = QuantumRegister(1, name="q")
    cr = ClassicalRegister(1, name='c')
    qc = QuantumCircuit(qr, cr)
    qc.u(params[0], params[1], params[2], qr[0])
    qc.measure(qr, cr[0])
    return qc


def counts_to_distr(counts):
    """Convert Qiskit result counts to dict with integers as
    keys, and pseudo-probabilities as values."""
    n_shots = sum(counts.values())
    return {int(k, 2): v/n_shots for k, v in counts.items()}


def objective_function(params):
    """Compares the output distribution of our circuit with
    parameters `params` to the target distribution."""
    # Create circuit instance with paramters and simulate it
    qc = get_var_form(params)
    result = backend.run(qc).result()
    # Get the counts for each measured state, and convert
    # those counts into a probability dict
    output_distr = counts_to_distr(result.get_counts())
    # Calculate the cost as the distance between the output
    # distribution and the target distribution
    cost = sum(
        abs(target_distr.get(i, 0) - output_distr.get(i, 0))
        for i in range(2**qc.num_qubits)
    )
    return cost