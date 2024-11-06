# https://github.com/ishan616/qiskit-projects-by-ishan/blob/dac9bf25f2ebf1728c691619f34291d14af8b379/password_gen.py
#This code generates a random password based on user's preference of password length using ibm's single qubit processor

#import relevant libraries and modules
from qiskit import *
from qiskit.providers.ibmq.job import job_monitor

#load your IBMQ account and get the backends ready
provider = IBMQ.load_account()
qcomp = provider.get_backend("ibmq_armonk")
M_simulator = Aer.get_backend("qasm_simulator") #local qiskit simulator package as an alternative to ibm's quantum devices

#set up our one qubit circuit intialized in the |+> state and measured in the measurement basis
qc = QuantumCircuit(1, 1)
qc.h(0)
qc.measure(0, 0)

password = ""
p_len = input("How long do you want your password to be? ")


#the allowable password characters reside in ASCII values between 33 and 126
#we run the random quantum measurement 7 times for each character in our password creating a 7 bit binary number 
#which is converted into an ASCII character if it is between 33 and 127 
#if not, we simply add 33 to the number and then convert it into the ASCII character
if not p_len.isnumeric():
    print("Please enter a number.")
else:
    p_len = int(p_len)

    #run our quantum experiment multiple times, store result from each run in local memory
    job = execute(qc, backend=qcomp, shots=7 * p_len, memory=True)
    job_monitor(job)
    result = job.result()
    memory = result.get_memory()
    bin_str = ""

    #bin_str will be a binary string of all our quantum meaurement results together
    for ch in memory:
        bin_str += ch

    #bins devides our bin_str in 7 bit numbers
    bins = [bin_str[i : i + 7] for i in range(0, len(bin_str), 7)]

    #conversion to ASCII conditioned on the bins being between 33 and 126
    chars = []
    for bin in bins:
        num = int(bin, 2)
        if num < 33:
            num += 33
        elif num == 127:
            num = 33
        chars += chr(num)

    #Assembling the password from ASCII characters
    password = ""
    for char in chars:
        password += char

print(password)
