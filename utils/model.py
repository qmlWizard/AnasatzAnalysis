import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils.ansatz import he
import json
import os

torch.manual_seed(42)
np.random.seed(42)

class QNN(nn.Module):
    
    def __init__(self, device, n_qubits, trainable, input_scaling, data_reuploading, ansatz, ansatz_layers, decoding = None):
        super().__init__()
        
        self._device = device
        self._n_qubits = n_qubits
        self._trainable = trainable
        self._input_scaling = input_scaling
        self._data_reuploading = data_reuploading
        self._ansatz = ansatz
        self._layers = ansatz_layers
        self._wires = range(self._n_qubits)
        self._projector = torch.zeros((2**self._n_qubits,2**self._n_qubits))
        self._decoding = decoding
        
        if self._decoding in (None, "round"):
            self._projector[0, 0] = 1
        
        if self._ansatz == 'he':
            if self._input_scaling:
                self.register_parameter(name="input_scaling", param= nn.Parameter(torch.ones(self._layers, self._n_qubits), requires_grad=True))
            else:
                self.register_parameter(name="input_scaling", param= nn.Parameter(torch.ones(self._layers, self._n_qubits), requires_grad=True))
            self.register_parameter(name="variational", param= nn.Parameter(torch.ones(self._layers, self._n_qubits * 2) * 2 * torch.pi, requires_grad=True))

        dev = qml.device(self._device, wires = range(self._n_qubits))
        if self._ansatz_circuit == 'he':
            self._ansatz = qml.QNode(he, dev, diff_method='adjoint', interface='torch')
        else:
            "Please select the Ansatz from (Hardware Efficient)"

    def _decode(self, probabilities):
        "Decoding the probabilities according to probabilities"
        if self._decoding == None:
            return probabilities
        elif self._decoding == 'round':
            return torch.round(probabilities)
    
    def forward(self, x):
        probabilities = self._ansatz_circuit(x, self._parameters, self._wires, self._layers, self._projector, self._data_reuploading)
        output = self._decode(probabilities)
        return output
    
    def summary(self):
        "printing the summary about the model"

    