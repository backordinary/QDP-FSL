# https://github.com/goulov/qauth/blob/bfafcc5e9630b84b7d6031ac563d207ad734bd6a/qauth.py
from qiskit import BasicAer
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute
import secrets
from numpy import random, pi, sin, cos, arcsin, sqrt
from scipy.optimize import minimize

# quantum computer simulator backend
backend = BasicAer.get_backend('qasm_simulator')

# protocol implementation
class qauth_simulator:
    def __init__(self, N, eps, db, user_access, levels2concurr):
        self._N = N
        self._eps = eps
        self._db = db
        self._user_access = user_access
        self._levels2concurr = levels2concurr
        self._resources = {}
        # generate resources for each user:
        for user in self._user_access.keys():
            self._resources[user] = self._generate_resources(user)

    def _get_angles(self, C):
        def W(x):
            a0, a1, b0, b1 = x
            return -0.25*( \
                    ((1+C)/2) * (cos(a0-b0)**2 + cos(a0-b1)**2 + cos(a1-b0)**2 \
                        + sin(a1-b1)**2) + \
                    ((1-C)/2) * (cos(a0+b0)**2 + cos(a0+b1)**2 + cos(a1+b0)**2 \
                        + sin(a1+b1)**2) )
        x0 = random.uniform(0, 2*pi, 4)
        sol = minimize(W,x0)
        return sol.x

    def _gen_entangled_2qbits(self, C):
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)
        theta = 0.5*arcsin(C)
        qc.ry(2*theta, qr[0]) # *2 because bloch sphere representation
        qc.cx(qr[0], qr[1])
        return qc

    def _generate_resources(self, user):
        level = self._user_access[user]
        return self._gen_entangled_2qbits(self._levels2concurr[level])

    def chsh_predicate(self, s, t, ca, cb):
        return s*t == ca^cb

    def expected_nrwins_chsh(self, level):
        C = self._levels2concurr[level]
        w = 0.5 + 0.25*sqrt(1+C**2)
        return round(self._N*w)

    def play_chsh_run(self, user):
        C = self._levels2concurr[self._user_access[user]]
        qc = self._resources[user]
        angles = self._get_angles(C)

        # alice lab
        a = angles[:2]
        s = secrets.randbits(1)
        cra = ClassicalRegister(1)
        qc = qc.combine(QuantumCircuit(cra))
        qa = qc.qregs[0][0]
        alpha = a[s]
        qc.ry(2*alpha, qa) # *2 because bloch sphere representation

        # bob lab
        b = angles[2:]
        t = secrets.randbits(1)
        crb = ClassicalRegister(1)
        qc = qc.combine(QuantumCircuit(crb))
        qb = qc.qregs[0][1]
        beta = b[t]
        qc.ry(2*beta, qb) # *2 because bloch sphere representation

        # run the circuit -- "announce phase"
        qc.measure(qa, cra)
        qc.measure(qb, crb)
        job = execute(qc, backend, shots=1)
        res = job.result().get_counts()
        ca, cb = [int(i) for i in list(res.keys())[0].split()]
        return s, t, ca, cb

    def play_all_chsh(self, user):
        nr_wins = 0
        for _ in range(self._N):
            s, t, ca, cb = self.play_chsh_run(user)
            nr_wins += self.chsh_predicate(s, t, ca, cb)
        return nr_wins

    def authorize(self, user):
        nr_wins = self.play_all_chsh(user)
        user_level = self._user_access[user]
        expected_wins = self.expected_nrwins_chsh(user_level)
        print("runs won:", nr_wins)
        print("expected:", expected_wins)
        print("epsilon is:", self._eps)
        # authorizer checks if the number of wins is as predicted (up to eps)
        if abs(nr_wins - expected_wins) <= self._eps:
            print(self._db[:user_level])
        else:
            print("FORBIDDEN: failed to prove entanglement level")


if __name__ == "__main__":
    # lambda = 128, l = 4
    N = 16384 # number of iterations
    epsilon = 1536 # error interval allowed

    db = ["dblevel1", "dblevel2", "dblevel3", "dblevel4"]

    levels2concurr = {1: 0.25,
                      2: 0.5,
                      3: 0.75,
                      4: 1}

    user_access = {"user1": 1,
                   "user2": 2,
                   "user3": 3,
                   "user4": 4}

    q = qauth_simulator(N, epsilon, db, user_access, levels2concurr)
    q.authorize('user4')
