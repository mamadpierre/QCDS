import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tdist






class Controller(torch.nn.Module):
    def __init__(self, args):
        torch.nn.Module.__init__(self)
        self.args = args
        self.rotations = ['x', 'y', 'z']
        self.operations = ['H', 'Px', 'Py', 'Pz', 'CNot', 'CSwap', 'Tof']
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
                else:
                    updated_design[layer + node + '2'] = 'Tof'

        return updated_design



class ControllerSmall(torch.nn.Module):
    def __init__(self, args):
        torch.nn.Module.__init__(self)
        self.args = args
        self.rotations = ['x', 'y', 'z']
        self.operations = ['H', 'Px', 'Py', 'Pz', 'CNot', 'CSwap', 'Tof']
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
            else:
                updated_design[node + '2'] = 'Tof'

        return updated_design
