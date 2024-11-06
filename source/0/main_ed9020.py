# https://github.com/Linueks/QuantumComputing/blob/c5876baad39b9337e7e50549f3f1c7c9d3de53dc/Fys4411/src/main.py
#!/usr/bin/env python3

def basicEncoder(qc, qr, features):
    for i,f in enumerate(features):
        qc.ry(f,qr[i])

def basicAnsatz(qc, qr, cr, features, theta):

    for i, f in enumerate(features):
            qc.rx(theta[i],qr[i])

    for i in range(len(qr) - 1):
        qc.cx(qr[i],qr[i-1])

    qc.ry(theta[-1],qr[-1])
    qc.measure(qr[-1],cr)


class Main:

    def __init__(self,features,theta):
        self.nq = features.shape[0];self.nc=1
        self.features = features
        self.theta = theta

    def model(self, encoder, ansatz, shots=1000):
        from qiskit import QuantumRegister,\
                ClassicalRegister,\
                QuantumCircuit,\
                execute,\
                Aer
        if (self.nc != 1):
            raise Exception("needs 1 classical measurement")

        qr = QuantumRegister(self.nq)
        cr = ClassicalRegister(self.nc)
        qc = QuantumCircuit(qr,cr)

        encoder(qc,qr,self.features)
        ansatz(qc,qr, cr, self.features, self.theta)

        #print(qc)

        job = execute(
                qc,
                backend=Aer.get_backend("qasm_simulator"),
                shots = shots,
                seed_simulator = 2021
                )
        
        result = job.result().get_counts(qc)
        prediction = result["0"]/shots
        return prediction

    def train(self, target, epochs=100, learning_rate=0.1):
        from numpy import zeros_like, pi

        for epoch in range(epochs):
            prediction = self.model(basicEncoder, basicAnsatz)
            mse = (prediction - target)**2
            mse_deriv = 2*(prediction - target)
            theta_gradient = zeros_like(self.theta)

            for i in range(self.features.shape[0]):
                
                self.theta[i] += pi/2
                o1 = self.model(basicEncoder, basicAnsatz)
                self.theta[i] -= pi
                o2 = self.model(basicEncoder, basicAnsatz)
                self.theta[i] += pi/2
                
                theta_gradient[i] = (o1 - o2)/2

            self.theta -= learning_rate*mse_deriv*theta_gradient
            print(mse)





if __name__ == "__main__":
    import numpy as np
    features = np.array([1.0, 1.5, 2.0, 0.3])
    parameters = 5
    theta = np.random.randn(parameters)

    qml = Main(features,theta)
    #qml.model(basicEncoder,basicAnsatz)
    target = .7
    qml.train(target)
