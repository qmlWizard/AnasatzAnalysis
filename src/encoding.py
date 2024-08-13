import pennylane as qml
from pennylane import numpy as np
import math


class Encoding:

    def __init__(self, encoding = 'naqss'):
        
        self.encoding = encoding

    def calculate_theta(self, image, N):
        thetas = np.pi * ((image - 1) / (N - 1))

        return thetas

    def calculate_alpha(self, theta):
        # Initialize an empty list to store alpha values
        alpha = []
        qubits = int(math.log2(theta.shape[0] * theta.shape[1]))
        for i in range(qubits):
            numerator = sum(np.abs(theta[1])**2)
            denominator = sum(np.abs(theta[0])**2)
            if i == 0:
                alpha.append(np.arctan(np.sqrt(numerator / denominator)))
            else:
                alpha.append(np.arctan(np.sqrt(np.abs(theta[i % 2][1])**2 / np.abs(theta[i % 2][0])**2)))
		
        return alpha
    
    def naqss_circuit(self, theta, alpha):
        qubits = int(math.log2(theta.shape[0] * theta.shape[1]))
        qml.RX(alpha[0], 0)
		
        for i in range(1, qubits - 1):
            for j in range(np.power(2, i)):
                qml.RX(alpha[i], wires = i)
                qml.MultiControlledX(control_wires = [qubit for qubit in range(i)], wires = i)

        for i in range(self.image.shape[0]):
            for j in range(self.image.shape[0]):
                qml.RX(theta[i][j], wires = qubits - 1)
                qml.MultiControlledX(control_wires = [qubit for qubit in range(qubits - 1)], wires = qubits - 1)


    def encode(self, image):

        if self.encoding == 'naqss':

            if len(image.shape) == 2:
                N = 256
            elif len(image.shape) == 3:
                 N = pow(2, 24)   

            thetas = self.calculate_theta(image)

            alphas = self.calculate_alpha(thetas)

            self.naqss_circuit(thetas, alphas)

        elif self.encoding == 'PCA':
            pass