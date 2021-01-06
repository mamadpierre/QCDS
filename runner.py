from __future__ import print_function
import torch
import torch.optim as optim
from Arguments import get_args
from QuantumNetwork import QNet
from datasets import GlassDataLoaders
from schemes import train, test
args = get_args()
import pickle


def design_set(design):
    if isinstance(design, dict):
        return design
    elif design is None:
        with open("designs/BenchMarkDesigns", 'rb') as file:
            benchMarkDesigns = pickle.load(file)
        return benchMarkDesigns[0]
    elif design =="CNOTRU":
        with open("designs/BenchMarkDesigns", 'rb') as file:
            benchMarkDesigns = pickle.load(file)
        return benchMarkDesigns[1]
    elif design == "CZ":
        with open("designs/BenchMarkDesigns", 'rb') as file:
            benchMarkDesigns = pickle.load(file)
        return benchMarkDesigns[2]
    elif design== "CZRU":
        with open("designs/BenchMarkDesigns", 'rb') as file:
            benchMarkDesigns = pickle.load(file)
        return benchMarkDesigns[3]
    elif "random" in design:
        for i in range(1000):
            if design == "RandomDesign{}".format(i):
                with open("designs/Top1000DesignsRandomSearch", 'rb') as file:
                    RandomDesigns = pickle.load(file)
                return RandomDesigns[i]
    elif "selected" in design:
        for i in range(6):
            if design=="selected{}".format(i):
                with open("designs/SelectedDesigns", 'rb') as file:
                    SelectedDesigns = pickle.load(file)
                return SelectedDesigns[i]
    else:
        raise Exception("Design should be a dictionary of the form {'000': True, '001': 'y', '002': 'CNot',... }")



def runner(args, train_loader, test_loader, design_indicator=None):
    design = design_set(design_indicator)
    print("design: ", design)
    torch.manual_seed(args.seed)
    model = QNet(args).to(args.device)
    optimizer = optim.Adam(model.QuantumLayer.parameters(), lr=args.lr)
    if (design_indicator is None or "selected" in design_indicator) and not args.from_scratch:
        if "selected" in design_indicator:
            for i in range(6):
                if design_indicator == "selected{}".format(i):
                    Checkpoint = torch.load("quantumWeights/{}QuantumWeights".format(i))
        else:
            Checkpoint = torch.load("quantumWeights/CNOT")
        model.QuantumLayer.load_state_dict(Checkpoint['model_state_dict'])
        optimizer.load_state_dict(Checkpoint['optimizer_state_dict'])
        print("Saved Quantum Weights are loaded")
    Best_epoch = 0
    best_test = 10000
    test_lossList = []
    train_lossList = []
    train_accuracy_list = []
    test_accuracy_list = []
    for epoch in range(1, args.Qepochs + 1):
        train(model, train_loader, optimizer, design, args)
        train_loss, train_accuracy = test(model, train_loader, design, args)
        test_loss, test_accuracy = test(model, test_loader, design, args)
        print("train acc: ", train_accuracy, "test acc: ", test_accuracy)
        test_lossList.append(test_loss)
        train_lossList.append(train_loss)
        train_accuracy_list.append(train_accuracy)
        test_accuracy_list.append(test_accuracy)
        if test_loss < best_test:
            best_test = test_loss
            Best_epoch = epoch
        torch.save({
            'epoch': epoch, 'model_state_dict': model.QuantumLayer.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), 'train loss list': train_lossList,
            'test loss list': test_lossList}, args.QuantumPATH)
    print("Best_epoch is {} with {} test loss".format(Best_epoch, best_test))
    return best_test, train_lossList, test_lossList, train_accuracy_list, test_accuracy_list





if __name__ == '__main__':
    args = get_args()
    train_loader, test_loader = GlassDataLoaders(args)
    runner(args, train_loader, test_loader, args.design_identifier)


