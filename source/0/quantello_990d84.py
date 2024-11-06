# https://github.com/RutvikJ77/Quantello/blob/426bae93317b393a0fa4c4f362ce457e10be90d3/Quantello.py
from time import sleep
import webbrowser
import qiskit as qk
import warnings
import os

#Load your account if running locally
# from qiskit import IBMQ
# IBMQ.save_account('YOUR_API_KEY')


warnings.filterwarnings("ignore") #Used to ignore the Numpy deprecated warnings for the qiskit package

n = 7   #Number of bits are calculated using 2^n - 1
q = qk.QuantumRegister(n)
c = qk.ClassicalRegister(n)
circ = qk.QuantumCircuit(q, c)
provider = qk.IBMQ.load_account()


for j in range(n):
    circ.h(q[j])
    
circ.measure(q,c)
print(circ)

backend = qk.BasicAer.get_backend('qasm_simulator') 

def rand_int():
    """
    Generates random number based on quantum computation
    returns integer
    """
    new_job = qk.execute(circ, backend, shots=1)
    bitstring = new_job.result().get_counts()
    bitstring = list(bitstring.keys())[0]
    integer = int(bitstring, 2)
    return integer


def print_string(current_string, expected_character):
    while True:
        #os.system("afplay assets/ambient.mp3&") Audio play Especially for macos
        some_character = rand_int()
        print("\r" + current_string + chr(some_character), end="")
        #sleep(0.1)  Uncomment to increase the runtime of the program
        if chr(some_character) == expected_character:
            break


my_string = "Hello " + input("Enter your name: ") + " ,thanks for wasting both our time.- Quantello XD"
current_str = ""

# Try to get the matching character for every character in the string.
for letter in my_string:
    print_string(current_str, letter)
    current_str += letter
print("\nNow that you have wasted your time, enjoy this video as well XD")
webbrowser.open("https://www.youtube.com/watch?v=DLzxrzFCyOs",new=1)