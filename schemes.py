import torch
import torch.nn.functional as F
from Arguments import get_args
from QuantumNetwork import QNet
import torch.optim as optim


def train(q_model, data_loader, optimizer, design, args):
    q_model.train()
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(args.device), target.to(args.device)
        optimizer.zero_grad()
        output = q_model(data, design)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

def test(q_model, data_loader, design, args):
    q_model.eval()
    epoch_loss = 0
    accuracy = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(args.device), target.to(args.device)
            output = q_model(data, design)
            instant_loss = F.nll_loss(output, target, reduction='sum').item()
            epoch_loss += instant_loss
            prediction = output.argmax(dim=1, keepdim=True)
            accuracy += prediction.eq(target.view_as(prediction)).sum().item()
    epoch_loss /= len(data_loader.dataset)
    accuracy /= len(data_loader.dataset)
    return epoch_loss, accuracy

def controller_train(q_model, controller, data_loader, controller_optimizer, design, log_prob, entropy, args):
    epoch_loss = 0
    controller.train()
    q_model.eval()
    accuracy = 0
    for data, target in data_loader:
        data, target = data.to(args.device), target.to(args.device)
        with torch.no_grad():
            q_output = q_model(data, design)
            if args.policy == "loss":
                q_loss = F.nll_loss(q_output, target, reduction='sum').item()
            else:
                prediction = q_output.argmax(dim=1, keepdim=True)
                accuracy += prediction.eq(target.view_as(prediction)).sum().item()
        if args.policy == "loss":
            policy_loss = log_prob * q_loss
        else:
            policy_loss = log_prob * (1 - accuracy / len(target))
        entropy_loss = args.entropy_weight * entropy
        instant_loss = policy_loss + entropy_loss
        instant_loss.backward()
        controller_optimizer.step()
        epoch_loss += instant_loss.item()
    if args.policy == "loss":
        epoch_loss /= len(data_loader.dataset)
    return epoch_loss


def Scheme(controller, train_loader, val_loader, test_loader, controller_optimizer, args):
    train_loss_list, val_loss_list, test_loss_list = [], [], []
    best_train_loss, best_val_loss, best_test_loss = 10000, 10000, 10000
    best_train_epoch, best_val_epoch, best_test_epoch = 0, 0, 0
    train_accuracy_list, test_accuracy_list = [], []
    best_design = None
    for epoch in range(1, args.Cepochs + 1):
        controller.eval()
        design, log_prob, entropy = controller()
        print("design: ", design)

        q_model = QNet(args).to(args.device)
        optimizer = optim.Adam(q_model.QuantumLayer.parameters(), lr=args.lr)
        for q_epoch in range(1, args.Qepochs + 1):
            train(q_model, train_loader, optimizer, design, args)
            epoch_train_loss, epoch_train_accuracy = test(q_model, train_loader, design, args)
            epoch_test_loss, epoch_test_accuracy = test(q_model, test_loader, design, args)
            train_loss_list.append(epoch_train_loss)
            train_accuracy_list.append(epoch_train_accuracy)
            test_loss_list.append(epoch_test_loss)
            test_accuracy_list.append(epoch_test_accuracy)
            if epoch_train_loss < best_train_loss:
                best_train_loss = epoch_train_loss
                best_train_epoch = epoch
            if epoch_test_loss < best_test_loss:
                best_test_loss = epoch_test_loss
                best_test_epoch = epoch
        epoch_val_loss = controller_train(q_model, controller, val_loader, controller_optimizer, design, log_prob,
                                          entropy, args)
        val_loss_list.append(epoch_val_loss)
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_val_epoch = epoch
            best_design = design
            torch.save({
                'epoch': epoch, 'q_model_state_dict': q_model.QuantumLayer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'controller_optimizer_state_dict': controller_optimizer.state_dict(),
                'controller_state_dict': controller.state_dict(),
                "test_loss_list": test_loss_list, "val_loss_list": val_loss_list, "train_loss_list": train_loss_list,
                "best_val_epoch": best_val_epoch, "best_val_loss": best_val_loss,
                "best_train_epoch": best_train_epoch, "best_train_loss": best_train_loss,
                "best_test_epoch": best_test_epoch, "best_test_loss": best_test_loss,
                "best_design": best_design}, args.QuantumPATH)
    return {"test_loss_list": test_loss_list, "val_loss_list": val_loss_list, "train_loss_list": train_loss_list,
            "best_val_epoch": best_val_epoch, "best_val_loss": best_val_loss,
            "best_train_epoch": best_train_epoch, "best_train_loss": best_train_loss,
            "best_test_epoch": best_test_epoch, "best_test_loss": best_test_loss,
            "best_design": best_design}

