import pennylane as qml
from pennylane import numpy as np
import torch

def _he_layer(x, _scaling_params, _variational_params, _wires, _embedding, _data_reuploading):
    
    if _embedding or _data_reuploading:
        for i, wire in enumerate(_wires):
            qml.RX(_scaling_params[i] * x[:,i], wires = [wire])
    for i, wire in enumerate(_wires):
        qml.RY(_variational_params[i], wires = [wire])
    for i, wire in enumerate(_wires):
        qml.RZ(_variational_params[i+len(_wires)], wires = [wire])

    if len(_wires) == 2:
        qml.broadcast(unitary=qml.CZ, pattern = "chain", wires = _wires)
    else:
        qml.broadcast(unitary=qml.CZ, pattern = "ring", wires = _wires)

def _he(x, weights, wires, layers, use_data_reuploading):
    first_layer = True
    for layer in range(layers):
        _he_layer(x, weights["input_scaling"][layer], weights["variational"][layer], wires, first_layer, use_data_reuploading)
        first_layer = False

def he(x1, weights, wires, layers, projector, data_reuploading):
    x1 = x1.repeat(1, len(wires) // len(x1[0]) + 1)[:, :len(wires)]
    _he(x1,weights,wires,layers,data_reuploading)
    return qml.expval(qml.Hermitian(projector, wires = wires))
