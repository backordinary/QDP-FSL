# https://github.com/R0J0-13/Semiprime-Factorisation-using-Shors/blob/36ebb75ae174ae0ad5a6dfe74c8125b6e2826eae/Shor's.py
#Modules

from qiskit import IBMQ #Imports the IBMQ Quantum Experience package from the qiskit Library Module from Python's Library Directory 
from qiskit import Aer #Imports the AER Quantum Experience package from the qiskit Library Module from Python's Library Directory 
from qiskit.aqua import QuantumInstance #Imports the Quantum Instance package from the qiskit's aqua package from Python's Library Directory 
from qiskit.aqua.algorithms import Shor #Imports the Shor Function package from the qiskit's aqua package from Python's Library Directory  
import csv #Imports the csv Library Module from Python's Library Directory 
import time #Imports the time Library Module from Python's Library Directory 

#Readying the Quantum Computer for Processing

IBMQ.enable_account("") #The API Token for accessing the IBMQ Machine
provider = IBMQ.get_provider(hub = "ibm-q") #Specifies the Quantum Provider
#backend = provider.get_backend('ibmq_qasm_simulator') #Specifies the Quantum Computer
backend = Aer.get_backend("qasm_simulator") #Specifies the Quantum Computer
#backend = provider.get_backend('ibmq_vigo') #Specifies the Quantum Computer

#Main

lis = [] #A List to store the elements retrieved from the csv file
with open('rtcQuantum.csv', 'r') as file: #Opens the csv file named rtcQuantum in Read mode
    reader = csv.reader(file) #Collates all elements of the file to a variable
    for row in reader: #Parses the file elements 
        #print(row) #FOR TESTING PURPOSES
        lis.append(row) #Adds each row of the csv file to a list
        
#print(lis) #FOR TESTING PURPOSES
lst2 = [item[0] for item in lis] #Extracts the elements in the first column of the csv file into a list

userInput = int(input("Please enter an Odd Number to factorise: ")) #User Input to get a number to be factorised

while str(userInput) in lst2: #A check to ensure already factorised numbers are not repeated
    print("This number has already been computed")
    userInput = int(input("Please enter an Odd Number to factorise: "))

while userInput % 2 == 0: #A check to ensure the entered number is odd
    print("The number must be Odd")
    userInput = int(input("Please enter an Odd Number to factorise: "))


print("\n Shors Algorithm")
print("--------------------")
print("\nExecuting...\n")    

import time 
startTime = time.time() #A marker to denote the beginning of the time measurement
  
N = Shor(userInput) #Function to run Shor's algorithm with the user input passed as the parameter
result_dict = N.run(QuantumInstance(backend, shots = 1, skip_qobj_validation = False)) #Store the factors generated
primeFactors = result_dict["factors"] #Get factors from results

print(primeFactors) #Outputs the prime factors of the user input onto the screen
endTime = time.time() #A marker to denote the end of the time measurement
time = endTime - startTime #The execution time is determined
print(time) #Outputs the execution time onto the screen

print("The Prime Factor pair for the number", str(userInput), "is:", primeFactors)

digits = len(str(userInput)) #Stores the number of digits in the inputted number

if len(primeFactors) > 0: #Condition to check if the primeFactors is empty

    with open(r'rtcQuantum.csv', 'a', newline='') as csvfile: #Opens the csv file named rtcQuantum in Append mode
        fieldnames = ['Number','Digits', 'Time'] #Stores the row heading names
        writer = csv.DictWriter(csvfile, fieldnames = fieldnames) #Creates the row headings for the csv file


        writer.writerow({'Number': str(userInput), 'Digits': str(digits), 'Time': str(time)}) #Writes to the csv file in the respective fields
        


