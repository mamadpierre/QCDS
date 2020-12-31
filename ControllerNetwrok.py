import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tdist
import pennylane as qml






class Controller(torch.nn.Module):
    def __init__(self, args):
        torch.nn.Module.__init__(self)
        self.args = args
        self.rotations = ['x', 'y', 'z']
        self.operations = ['H', 'Px', 'Py', 'Pz', 'CNot', 'CSwap', 'Tof', 'CZ']
        self.shared_fc1 = nn.Linear(1, 48)
        self.shared_fc2 = nn.Linear(48, 12)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.BN1 = nn.BatchNorm1d(48)
        self.BN2 = nn.BatchNorm1d(12)
        self.fcAction = nn.ModuleDict()

        for layer in range(self.args.q_depth):
            for node in range(self.args.n_qubits):
                self.fcAction[str(layer % 6) + str(node % 4) + '0'] = nn.Linear(12, 2) #reUploading or Not
                self.fcAction[str(layer % 6) + str(node % 4) + '1'] = nn.Linear(12, len(self.rotations)) #which rotations
                self.fcAction[str(layer % 6) + str(node % 4) + '2'] = nn.Linear(12, len(self.operations)) #which operations


    def forward(self):
        design = np.empty([self.args.q_depth, self.args.n_qubits, 3])
        log_prob_list = []
        entropy_list = []
        x = self.shared_fc1(torch.tensor([[1.0]]))
        x = F.leaky_relu(self.BN1(x))
        x = self.dropout1(x)
        x = self.shared_fc2(x)
        x = F.leaky_relu(self.BN2(x))
        x = self.dropout2(x)
        for layer in range(self.args.q_depth):
            for node in range(self.args.n_qubits):
                for decision in range(3):
                    logits = self.fcAction[str(layer % 6) + str(node % 4) + str(decision)](x)
                    probs = F.softmax(logits, dim=1)
                    m = tdist.Categorical(probs)
                    action = m.sample()
                    instant_log_prob = m.log_prob(action)
                    instant_entropy = m.entropy()
                    design[layer, node, decision] = action
                    log_prob_list.append(instant_log_prob)
                    entropy_list.append(instant_entropy)
        design = torch.tensor(design)
        log_prob = torch.sum(torch.stack(log_prob_list))
        entropy = torch.sum(torch.stack(entropy_list))
        return self.post_process(design), entropy, log_prob


    def post_process(self, design):
        updated_design = {}
        for l in range(self.args.q_depth):
            for n in range(self.args.n_qubits):
                layer = str(l)
                node = str(n)
                if design[l, n, 0] == 0:
                    updated_design[layer + node + '0'] = False
                else:
                    updated_design[layer + node + '0'] = True

                if design[l, n, 1] == 0:
                    updated_design[layer + node + '1'] = 'x'
                elif design[l, n, 1] == 1:
                    updated_design[layer + node + '1'] = 'y'
                else:
                    updated_design[layer + node + '1'] = 'z'

                if design[l, n, 2] == 0:
                    updated_design[layer + node + '2'] = 'H'
                elif design[l, n, 2] == 1:
                    updated_design[layer + node + '2'] = 'Px'
                elif design[l, n, 2] == 2:
                    updated_design[layer + node + '2'] = 'Py'
                elif design[l, n, 2] == 3:
                    updated_design[layer + node + '2'] = 'Pz'
                elif design[l, n, 2] == 4:
                    updated_design[layer + node + '2'] = 'CNot'
                elif design[l, n, 2] == 5:
                    updated_design[layer + node + '2'] = 'CSwap'
                elif design[l, n, 2] == 6:
                    updated_design[layer + node + '2'] = 'Tof'
                else:
                    updated_design[layer + node + '2'] = 'CZ'

        return updated_design



class ControllerSmall(torch.nn.Module):
    def __init__(self, args):
        torch.nn.Module.__init__(self)
        self.args = args
        self.rotations = ['x', 'y', 'z']
        self.operations = ['H', 'Px', 'Py', 'Pz', 'CNot', 'CSwap', 'Tof', 'CZ']
        self.shared_fc1 = nn.Linear(1, 48)
        self.shared_fc2 = nn.Linear(48, 12)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.BN1 = nn.BatchNorm1d(48)
        self.BN2 = nn.BatchNorm1d(12)
        self.fcAction = nn.ModuleDict()

        for node in range(self.args.n_qubits):
            self.fcAction[str(node % 4) + '0'] = nn.Linear(12, 2) #reUploading or Not
            self.fcAction[str(node % 4) + '1'] = nn.Linear(12, len(self.rotations)) #which rotations
            self.fcAction[str(node % 4) + '2'] = nn.Linear(12, len(self.operations)) #which operations


    def forward(self):
        design = np.empty([self.args.n_qubits, 3])
        log_prob_list = []
        entropy_list = []
        x = self.shared_fc1(torch.tensor([[1.0]]))
        x = F.leaky_relu(self.BN1(x))
        x = self.dropout1(x)
        x = self.shared_fc2(x)
        x = F.leaky_relu(self.BN2(x))
        x = self.dropout2(x)
        for node in range(self.args.n_qubits):
            for decision in range(3):
                logits = self.fcAction[str(node % 4) + str(decision)](x)
                probs = F.softmax(logits, dim=1)
                # print(node, decision, "probs: ", probs)
                m = tdist.Categorical(probs)
                action = m.sample()
                instant_log_prob = m.log_prob(action)
                instant_entropy = m.entropy()
                design[node, decision] = action
                log_prob_list.append(instant_log_prob)
                entropy_list.append(instant_entropy)
        # print("-"*80)
        design = torch.tensor(design)
        log_prob = torch.sum(torch.stack(log_prob_list))
        entropy = torch.sum(torch.stack(entropy_list))
        return self.post_process(design), entropy, log_prob


    def post_process(self, design):
        updated_design = {}
        for n in range(self.args.n_qubits):
            node = str(n)
            if design[n, 0] == 0:
                updated_design[node + '0'] = False
            else:
                updated_design[node + '0'] = True

            if design[n, 1] == 0:
                updated_design[node + '1'] = 'x'
            elif design[n, 1] == 1:
                updated_design[node + '1'] = 'y'
            else:
                updated_design[node + '1'] = 'z'

            if design[n, 2] == 0:
                updated_design[node + '2'] = 'H'
            elif design[n, 2] == 1:
                updated_design[node + '2'] = 'Px'
            elif design[n, 2] == 2:
                updated_design[node + '2'] = 'Py'
            elif design[n, 2] == 3:
                updated_design[node + '2'] = 'Pz'
            elif design[n, 2] == 4:
                updated_design[node + '2'] = 'CNot'
            elif design[n, 2] == 5:
                updated_design[node + '2'] = 'CSwap'
            elif design[n, 2] == 6:
                updated_design[node + '2'] = 'Tof'
            else:
                updated_design[node + '2'] = 'CZ'
        return updated_design

    
    


class QuantumLayerController(nn.Module):
    def __init__(self, q_depth, n_qubits, n_output):
        super().__init__()
        self.q_depth = q_depth
        self.n_qubits = n_qubits
        self.n_output = n_output

        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch")
        def quantum_net(q_weights_flat, **kwargs):
            q_depth = kwargs['q_depth']
            n_qubits = kwargs['n_qubits']
            n_output = kwargs['n_output']
            q_weights = q_weights_flat.reshape(q_depth, n_qubits)
            for idx in range(n_qubits):
                qml.Hadamard(wires=idx)
            for layer in range(q_depth):
                for node in range(n_qubits):
                    qml.RY(q_weights[layer][node], wires=node)
                    qml.CNOT(wires=[node, (node + 1) % n_qubits])
            exp_vals = [qml.expval(qml.PauliZ(position)) for position in range(n_output)]
            return tuple(exp_vals)

        self.quantum_net = quantum_net
        self.q_params = nn.Parameter(torch.randn(self.q_depth * self.n_qubits))
    def forward(self):
        output = self.quantum_net(self.q_params, q_depth=self.q_depth, n_qubits=self.n_qubits, n_output=self.n_output).float().unsqueeze(0)
        return F.log_softmax(output, dim=1)





class QControllerSmall(torch.nn.Module):
    def __init__(self, args):
        torch.nn.Module.__init__(self)
        self.args = args
        self.fcAction = nn.ModuleDict()
        for node in range(self.args.n_qubits):
            self.fcAction[str(node % 4) + '0'] = QuantumLayerController(3, 3, 2)
            self.fcAction[str(node % 4) + '1'] = QuantumLayerController(3, 3, 3)
            self.fcAction[str(node % 4) + '2'] = QuantumLayerController(3, 4, 4)

    def forward(self):
        design = np.empty([self.args.n_qubits, 3])
        log_prob_list = []
        entropy_list = []
        for node in range(self.args.n_qubits):
            for decision in range(3):
                probs = self.fcAction[str(node % 4) + str(decision)]()
                m = tdist.Categorical(probs)
                action = m.sample()
                instant_log_prob = m.log_prob(action)
                instant_entropy = m.entropy()
                design[node, decision] = action
                log_prob_list.append(instant_log_prob)
                entropy_list.append(instant_entropy)
        design = torch.tensor(design)
        log_prob = torch.sum(torch.stack(log_prob_list))
        entropy = torch.sum(torch.stack(entropy_list))
        return self.post_process(design), entropy, log_prob


    def post_process(self, design):
        updated_design = {}
        for n in range(self.args.n_qubits):
            node = str(n)
            if design[n, 0] == 0:
                updated_design[node + '0'] = False
            elif design[n, 0] == 1:
                updated_design[node + '0'] = True
            else:
                raise Exception('Wrong reupload value')

            if design[n, 1] == 0:
                updated_design[node + '1'] = 'x'
            elif design[n, 1] == 1:
                updated_design[node + '1'] = 'y'
            elif design[n, 1] == 2:
                updated_design[node + '1'] = 'z'
            else:
                raise Exception('Wrong rotation value')

            if design[n, 2] == 0:
                updated_design[node + '2'] = 'CNot'
            elif design[n, 2] == 1:
                updated_design[node + '2'] = 'CZ'
            elif design[n, 2] == 2:
                updated_design[node + '2'] = 'CSwap'
            elif design[n, 2] == 3:
                updated_design[node + '2'] = 'Tof'
            else:
                raise Exception('Wrong non parametric value')
        return updated_design

























