# https://github.com/carbasemin/qiskit_/blob/d8f6d164bd6e3d6b1e70039d4c7b9d51ae0bc0ad/algorithm_generalizations/simon_joker.py
from qiskit import*
from qiskit.visualization import plot_histogram
from random import randint

#Asking the user for the number of qbits.

try:
    n = int(input('How many qbits (in a register) you\'d like have sir? Give us a number between 1 and 8: '))
    print('')
except:
    try:
    	n = int(input("A number sir, not something else. A NUMBER between 1 and 8 sir please: "))
    except:
    	exit('Fucking idiot.')
    
    
if n < 9:
    pass
else:
    try:
    	n = int(input("Between 1 and 8 sir, 1 and 8. Let's try again."))
    	if n < 9:
    		pass
    	else:
    		exit('Fuck you.')
    except:
    	print('...')

#And now the period.
print('')

try:
	s_integer = int(input('''And what about the period sir, what should it be? 
Give us a number between 0 and 2^(number of qbits): '''))
except:
	try:
		s_integer = int(input('I said, a number, sir, not anything else but a number. Please: '))
	except:
		exit('...')

if s_integer < 2**n:
    pass
else:
    try:
    	s_integer = int(input('Between 0 and 2^(number of qbits) sir please if you will: '))
    	if s_integer < 2**n:
    		pass
    	else:
    		exit('Why god why?')
    except:
    	print('...')

s = bin(s_integer)[2:]

if len(s) != n:
    for i in range(n - len(s)):
        s = '0' + s
else:
    pass

print('Here is the period you\'ve kindly provided us in binary sir:', s)

#Oracle

print('\n', end = 'Now, some magic! This oracle something will guess your number!')

sc = QuantumCircuit(n*2, n)

sc.h(range(n))
sc.barrier()

#This is our oracle, the Simon function, that is. 
for i,j in enumerate(reversed(s)):
    if j == '1':
        for q in range(n):
            sc.cx(i, (n) + q)
        sc.barrier()

sc.h(range(n))
sc.barrier()

sc.measure(range(n), range(n))

sim = Aer.get_backend('qasm_simulator')
results = execute(sc, sim).result().get_counts()
del results['0'*n] #I'd rather not have the 00000... result. It comes by default and has no use.

s_guess = list(results)[0]

#Guessing.

print('\n')
print('The period is:', s_guess, 'Is it really? Let\'s check.')
print('Is', s_guess, 'equal to', s + '?')

if s_guess == s:
    print('Yes, it indeed is. I\'m good at guessing stuff then. That\'s a proven fact now.')
else:
    print('No, it sure is not. Bad luck...')