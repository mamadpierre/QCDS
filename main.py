from __future__ import print_function
from ControllerNetwrok import Controller, ControllerSmall
from datasets import IRISDataLoaders
from plot import loss_plotting
from schemes import *
import torch
import os
import pickle
from Arguments import get_args


def main(args):
    print("args:", args)
    train_loader, val_loader, test_loader = IRISDataLoaders(args)
    if args.small_design:
        ControllerModel = ControllerSmall(args).to(args.device)
    else:
        ControllerModel = Controller(args).to(args.device)
    controller_optimizer = torch.optim.Adam(ControllerModel.parameters(), lr=args.Clr, eps=1e-3)
    report = Scheme(ControllerModel, train_loader, val_loader, test_loader, controller_optimizer, args)

    with open(os.path.join(args.path, 'IrisResults{}{}'.format(str(args.Clr), str(args.entropy_weight))), 'wb') as file:
        pickle.dump(report, file)
    print("report: ", report)
    loss_plotting({"train loss": report['train_loss_list'],
                   "validation loss": report['val_loss_list'], "test loss": report['test_loss_list']})





if __name__ == '__main__':
    args = get_args()
    main(args)















