import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
from Arguments import get_args
args = get_args()

dev = qml.device("default.qubit", wires=args.n_qubits)
@qml.qnode(dev, interface="torch")
def quantum_net(q_input_features, q_weights_flat, **kwargs):
    current_design = kwargs['design']
    q_weights = q_weights_flat.reshape(args.q_depth, args.n_qubits)
    for idx in range(args.n_qubits):
        qml.Hadamard(wires=idx)
    for layer in range(args.q_depth):
        for node in range(args.n_qubits):
            if args.small_design:
                layer_node0 = current_design[str(node % 4) + '0']
                layer_node1 = current_design[str(node % 4) + '1']
                layer_node2 = current_design[str(node % 4) + '2']
            else:
                layer_node0 = current_design[str(layer % 6) + str(node % 4) + '0']
                layer_node1 = current_design[str(layer % 6) + str(node % 4) + '1']
                layer_node2 = current_design[str(layer % 6) + str(node % 4) + '2']
            if layer_node0:
                qml.RY(q_input_features[node], wires=node)
            if layer_node1 == 'x':
                qml.RX(q_weights[layer][node], wires=node)
            if layer_node1 == 'y':
                qml.RY(q_weights[layer][node], wires=node)
            if layer_node1 == 'z':
                qml.RZ(q_weights[layer][node], wires=node)
            if layer_node2 == 'H':
                qml.Hadamard(wires=node)
            if layer_node2 == 'Px':
                qml.PauliX(wires=node)
            if layer_node2 == 'Py':
                qml.PauliY(wires=node)
            if layer_node2 == 'Pz':
                qml.PauliZ(wires=node)
            if layer_node2 == 'CNot':
                qml.CNOT(wires=[node, (node + 1) % args.n_qubits])
            if layer_node2 == 'CSwap':
                qml.CSWAP(wires=[node, (node + 1) % args.n_qubits, (node + 2) % args.n_qubits])
            if layer_node2 == 'Tof':
                qml.Toffoli(wires=[node, (node + 1) % args.n_qubits, (node + 2) % args.n_qubits])
            if layer_node2 == 'CZ':
                qml.CZ(wires=[node, (node + 1) % args.n_qubits])
    exp_vals = [qml.expval(qml.PauliZ(position)) for position in range(args.n_output)]
    return tuple(exp_vals)


##############################################################################

class QuantumLayer(nn.Module):

    def __init__(self, arguments):
        super().__init__()
        self.args = arguments
        self.q_params = nn.Parameter(torch.randn(self.args.q_depth * self.args.n_qubits))

    def forward(self, input_features, design):
        q_out = torch.Tensor(0, self.args.n_output)
        q_out = q_out.to(self.args.device)
        for elem in input_features:
            q_out_elem = quantum_net(elem, self.q_params, design=design).float().unsqueeze(0)
            q_out = torch.cat((q_out, q_out_elem))
        return q_out


class QNet(nn.Module):
    def __init__(self, arguments):
        super(QNet, self).__init__()
        self.args = arguments
        self.QuantumLayer = QuantumLayer(self.args)

    def forward(self, x, design):
        x = self.QuantumLayer(x, design)
        output = F.log_softmax(x, dim=1)
        return output
