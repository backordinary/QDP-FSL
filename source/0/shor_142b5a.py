# https://github.com/AramDevops/QURSA/blob/431c1335f80bf7a70cbf90df6dbddf47981526cd/apps/Shor.py
"""
@author: Nasr Akram
"""
# imports for the App
from hydralit import HydraHeadApp
from pathlib import Path
import streamlit as st
import base64

# imports for Quantum part
from qiskit.visualization import plot_histogram
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit.library import QFT

#imports for Classical part
from datetime import timedelta
from sympy import sieve
import random
import math
import time


def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

class CheatApp(HydraHeadApp):

    def __init__(self, title='', **kwargs):
        self.__dict__.update(kwargs)
        self.title = title

    def run(self):

        st.markdown("""
            <style>
            .css-1v0mbdj.etr89bj1  img {
            -ms-transform: scale(0.7);
            -webkit-transform: scale(0.5);
            -o-transform: scale(0.5);
            -moz-transform: scale(0.5);
            transform: scale(0.5);
            border: 2px solid #fff;
            box-shadow: 10px 10px 5px  rgb(38, 255, 4);
            -moz-box-shadow: 10px 10px 5px  rgb(38, 255, 4);
            -webkit-box-shadow: 10px 10px 5px  rgb(38, 255, 4);
            -khtml-box-shadow: 10px 10px 5px  rgb(38, 255, 4);
            margin-top: -32%;
            margin-bottom: -18%;
            border-radius: 20px;}
            </style>""" , unsafe_allow_html=True)

        st.markdown(
            "<center><img style='margin-top:-200px; width:15%'src='https://i.gifer.com/VIcR.gif'></center>",
            unsafe_allow_html=True)

        def generate_primes(start, end):
            # Use a sieve algorithm to generate a list of prime numbers in the specified range
            primes = list(sieve.primerange(start, end))
            return primes

        def is_prime(n):
            if n < 2:
                return False
            for i in range(2, int(math.sqrt(n)) + 1):
                if n % i == 0:
                    return False
            return True

        st.sidebar.markdown(f"Voici des nombres premiers générés aléatoirement avec leurs niveaux de difficulté")

        st.sidebar.write(f"Très facile : {random.sample(generate_primes(300, 700), 3)}")
        st.sidebar.write(f"Facile : {random.sample(generate_primes(3000, 7000), 3)}")
        st.sidebar.write(f"Moyen : {random.sample(generate_primes(30000, 70000), 3)}")
        st.sidebar.write(f"Moyen/Long : {random.sample(generate_primes(300000, 700000), 3)}")

        st.sidebar.markdown("Entrez les nombres premiers P&Q : ")
        p = st.sidebar.number_input("P : ", key="number_1", min_value=1)
        if not is_prime(p):
            st.sidebar.error("P doit être un nombre premier")
        q = st.sidebar.number_input("Q : ", key="number_2", min_value=1)
        if not is_prime(q):
            st.sidebar.error("Q doit être un nombre premier")

        # Display the selected or entered number
        st.markdown("Vous avez sélectionné : ")
        st.write("Votre valeur P : ",p)
        st.write("Votre valeur Q : ", q)

        # Easy test:
        #p = 433
        #q = 619

        # Easy-Medium test:
        #p = 12281
        #q = 13297

        # Medium test:
        #p = 450563
        #q = 462263

        # list of Hard tests:
        # big_primes_numb = [978270003817, 1000000005721, 978270003427, 999999993029, 989999983979,1000000001491, 1000000005677, 978270002791]

        N = p*q

        st.write(f'N : ', N)

        st.write(f'Votre clé publique est : {[N,65537]}')

        message = st.text_input(f"Le message que vous souhaitez chiffrer : ")

        def Crypter(clef_publique, mon_message):
            key, n = clef_publique
            msg_chiffre = [(ord(char) ** key) % n for char in message]
            return msg_chiffre

        msg_ssl = Crypter((65537, N), message)
        st.write(f"Le message chiffré avec la clé publique : {msg_ssl}")

        def value_a(N):

            while not False:
                a = random.randint(2, N)
                if math.gcd(a, N) == 1:
                    return a


        def initialize_qubits(qc, n, m):
            qc.h(range(n))  # apply hadamard gates
            qc.x(n + m - 1)  # set qubit to 1

        st.write(f"Fonction: \n\tU(x) = a^x mod {N}")

        def c_modN(a, power):
            U = QuantumCircuit(5)
            for iteration in range(power):
                if a & 1:  # check if a is odd
                    for q in range(5 // 2):
                        U.rx(math.pi / 2, q)  # use rx gate for qubit rotation
                else:
                    continue  # skip rest of loop iteration if a is even

            U = U.to_gate()
            U.name = "%i^%i mod N" % (a, power)
            c_U = U.control()
            return c_U

        def modular_exponentiation(qc, n, m, a):
            #st.write(qc,n,m,a)
            for x in range(n):
                exponent = 2 ** x
                qc.append(c_modN(a, exponent),
                          [x] + list(range(n, n + m)))


        def qft_dagger(qc, measurement_qubits):
            qc.append(QFT(len(measurement_qubits),
                          do_swaps=False).inverse(),
                      measurement_qubits)
            qc.name = "QFT†"

        def measure(qc, n):
            qc.measure(n, n)

        def error_correction(qc, n):
            qc.reset(range(n + 3))
            qc.barrier()
            for i in range(n):
                # apply CNOT gates to correct bit flip errors
                qc.cx(i, n)
                qc.cx(i, n + 1)
                qc.cx(i, n + 2)
                qc.x(n)
                qc.x(n + 1)
                qc.x(n + 2)
                qc.cx(i, n)
                qc.cx(i, n + 1)
                qc.cx(i, n + 2)
                qc.x(n)
                qc.x(n + 1)
                qc.x(n + 2)
            qc.barrier()

        def period_finder(n, m, a):
            # set up quantum circuit
            qc = QuantumCircuit(n + m, n)

            # initialize the qubits
            initialize_qubits(qc, n, m)
            qc.barrier()

            # apply modular exponentiation
            modular_exponentiation(qc, n, m, a)
            qc.barrier()

            # apply error correction
            error_correction(qc, n)

            # apply inverse QFT
            qft_dagger(qc, range(n))
            qc.barrier()

            # measure the n measurement qubits
            measure(qc, range(n))


            return qc

        def Modular_multiplicative_inverse(a, n):
            t = 0
            newt = 1
            r = n
            newr = a
            while newr != 0:
                quotient = r // newr
                t, newt = newt, t - quotient * newt
                r, newr = newr, r - quotient * newr
            if r > 1:
                return "a is not invertible"
            if t < 0:
                t = t + n
            return t

        def run_shor():
            start_time = time.time()
            factor_found = False
            attempt = 0
            st.write("Chargement...", end='\t')
            st.write("Connexion à l'ordinateur quantique...", end='\t')
            st.write("Calcul...", end='\t')

            while not factor_found:
                try :
                    attempt += 1
                    a = value_a(N)
                    qc = period_finder(5, 5, a)

                    simulator = Aer.get_backend('qasm_simulator')
                    counts = execute(qc, backend=simulator).result().get_counts(qc)


                    # convert and add binary periods to list
                    counts_dec = sorted([int(measured_value[::-1], 2)
                                         for measured_value in counts])

                    factors = set()

                    for measured_value in counts_dec:

                        guesses = [math.gcd(int((a ** (measured_value / 2))) + 1, N),
                                   math.gcd(int((a ** (measured_value / 2))) - 1, N)]


                        for guess in guesses:
                            # ignore trivial factors
                            if guess != 1 and guess != N and N % guess == 0:
                                factors.add(guess)
                                # Print the value of r
                                r = int(measured_value / 2)
                                st.write('La valeur de la période "r" est:', {r})
                                f = math.gcd(int((a ** (measured_value / 2))) + 1, N)
                                k = math.gcd(int((a ** (measured_value / 2))) - 1, N)
                                st.write('On teste les deux formules pour trouver les facteurs avec:')
                                st.write('N_1 = gcd(' , a , '^(', r ,'/2) + 1, ', N ,') ')
                                st.write('N_2 = gcd(' , a , '^(', r ,'/2) - 1, ', N ,') ')
                                if f > 1:
                                    st.write(f'Le facteur trouvé avec gcd(' , a , '^(', r ,'/2) + 1, ', N ,') = ',{f})
                                if k > 1:
                                    st.write(f'Le facteur trouvé avec gcd(' , a , '^(', r ,'/2) - 1, ', N ,') = ',{k})

                                factor_found = True

                    if len(factors)!=0:

                        P = factors.pop()
                        Q = factors.pop() if len(factors) else N // P

                        st.write("Chargement des résultats...")
                        st.write("\nTentative %i:" % attempt)
                        st.write(qc.draw(output='mpl'))
                        st.write(plot_histogram(counts))

                        st.write("P et Q trouvé avec l'ordinateur quantique : ")
                        st.write('N = ', Q, ' x ', P)

                        phi = (P - 1) * (Q - 1)
                        cle_p = Modular_multiplicative_inverse(65537, phi)
                        st.write("La clé privée trouvée : ")
                        st.write('d = ', cle_p)

                        elapsed_time_secs = time.time() - start_time
                        msg_time = "L'exécution a pris : %s secs" % timedelta(seconds=round(elapsed_time_secs))
                        st.write(msg_time)
                except:
                    st.write("Pas assez de qubits !")
                    st.write("Arrêt en cours...")
                    break
        if len(msg_ssl)>0 and is_prime(p) and is_prime(q) :
            if st.button("Démarrer l'algorithme"):
                run_shor()
