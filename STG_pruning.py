import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
import math
from copy import deepcopy
import numpy as np
import pandas as pd
import pickle

from torchmetrics.classification import BinaryAccuracy

#Code adapted from "https://github.com/runopti/stg" (Yamada et al., 2020)
class STGLayer(nn.Module):
    def __init__(self, input_dim, sigma, hard_selection=False):
        super(STGLayer, self).__init__()
        self.mu = torch.nn.Parameter(0.01 * torch.randn(input_dim), requires_grad=True)
        self.noise = torch.randn_like(self.mu)
        self.sigma = sigma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gates = torch.zeros_like(self.mu)
        self.prob_selection = torch.zeros_like(self.mu)
        self.hard_selection = hard_selection

    def forward(self, prev):
        z = self.mu + self.sigma * self.noise.normal_() * self.training
        self.gates = self.hard_sigmoid(z)
        self.prob_selection = torch.max(torch.Tensor([0]).to(self.device),
                                        torch.min(torch.Tensor([1]).to(self.device), z))
        if self.hard_selection:
            new = prev * self.prob_selection
        else:
            new = prev * self.gates

        return new

    def hard_sigmoid(self, x):
        return torch.clamp(x + 0.5, 0.0, 1.0)

    def regularizer(self, x):
        ''' Gaussian CDF. '''
        return 0.5 * (1 + torch.erf(x / math.sqrt(2)))

    def _apply(self, fn):
        super(STGLayer, self)._apply(fn)
        self.noise = fn(self.noise)
        return self


class STGPruningNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, h_dim, n_layers=3, dropout=0.2, activation=nn.ReLU(), sigma=0.5,
                 binary_outcome=False):
        super(STGPruningNetwork, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.h_dim_big = h_dim
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.sigma = sigma
        self.binary_outcome = binary_outcome
        self.pruned = torch.zeros(self.input_dim).to(self.device)
        self.selected = torch.zeros(self.input_dim).to(self.device)
        self.prob_selection = torch.zeros(self.input_dim).to(self.device)
        self.feature_selector = STGLayer(self.input_dim, self.sigma, hard_selection=False).to(self.device)
        self.reg = self.feature_selector.regularizer
        self.mu = self.feature_selector.mu

        layers = [nn.Linear(self.input_dim, self.h_dim),
                  nn.BatchNorm1d(self.h_dim), self.activation, self.dropout]
        for i in range(self.n_layers - 2):
            layers += [nn.Linear(self.h_dim, self.h_dim), nn.BatchNorm1d(self.h_dim),
                       self.activation, self.dropout]
        layers += [nn.Linear(self.h_dim, self.output_dim)]
        if self.binary_outcome:
            layers += [nn.Sigmoid()]
        self.layers = nn.Sequential(*layers)

    def forward(self, x, y):
        y_hat = self.layers(self.feature_selector(x))
        self.pruned = torch.where(self.feature_selector.gates < 0.5)[0]
        self.selected = torch.where(self.feature_selector.gates >= 0.5)[0]
        self.prob_selection = self.feature_selector.prob_selection[torch.nonzero(self.feature_selector.prob_selection)].squeeze()
        if self.binary_outcome:
            CL = F.binary_cross_entropy(y_hat, y)
        else:
            CL = F.mse_loss(y_hat, y)
        REG = torch.mean(self.reg((self.mu + 0.5)/self.sigma))/self.input_dim

        pred = {'y_hat': y_hat}
        loss = {'CL': CL, 'REG': REG}

        return pred, loss


def train_STG_single(model, train_loader, val_loader, optimizer, scheduler, n_epochs, coef,
                     binary_outcome=False, early_stopping=True, verbose=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if binary_outcome:
        metric = BinaryAccuracy().to(device)
    else:
        metric = nn.MSELoss().to(device)

    if early_stopping:
        best_loss = float("inf")
        early_stopping_wait = 20
        best_model_state_epoch = 100

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        train_loss = []
        train_CL = []
        train_REG = []
        train_metric = 0

        val_loss = []
        val_CL = []
        val_REG = []
        val_metric = 0

        for i, (x, y) in enumerate(train_loader):
            model.train()
            x, y = x.to(device), y.to(device).unsqueeze(dim=1).float()
            pred, loss = model(x, y)
            total_loss = (loss['CL'] + coef * loss['REG'])
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += [total_loss.item()]
            train_CL += [loss['CL'].item()]
            train_REG += [loss['REG'].item()]
            train_metric += metric(pred['y_hat'], y)
        train_metric /= len(train_loader)
        metrics = {"train": train_metric}

        for i, (x, y) in enumerate(val_loader):
            model.eval()
            x, y = x.to(device), y.to(device).unsqueeze(dim=1).float()
            pred, loss = model(x, y)
            total_loss = (loss['CL'] + coef * loss['REG'])

            val_loss += [total_loss.item()]
            val_CL += [loss['CL'].item()]
            val_REG += [loss['REG'].item()]
            val_metric += metric(pred['y_hat'], y)
        val_metric /= len(val_loader)
        metrics["val"] = val_metric

        if verbose:
            print("Epoch {}/{} Done, Train Loss: {:.4f}, Validation Loss: {:.4f}".format(epoch + 1, n_epochs,
                                                                                         sum(train_loss) / len(train_loss),
                                                                                         sum(val_loss) / len(val_loss)))
            if (epoch + 1) % 10 == 0:
                print("-------------------------Training-------------------------")
                print("BCE: {:.4f}, REG: {:.4f}, Metric: {:.4f}".format(sum(train_CL) / len(train_CL),
                                                                        sum(train_REG) / len(train_REG),
                                                                        train_metric))

                print("------------------------Validation------------------------")
                print("BCE: {:.4f}, REG: {:.4f}, Metric: {:.4f}".format(sum(val_CL) / len(val_CL),
                                                                        sum(val_REG) / len(val_REG),
                                                                        val_metric))
        if early_stopping:
            if epoch > best_model_state_epoch:
                total_val_loss = sum(val_loss) / len(val_loss)
                if best_loss > total_val_loss:
                    best_loss = total_val_loss
                    best_model_state = deepcopy(model.state_dict())
                    best_model_state_epoch = epoch

            if epoch - best_model_state_epoch > early_stopping_wait:  # Early stopping
                break

    if early_stopping:
        print("Early stoping at epoch {}".format(epoch))
        print("Best model at epoch {} with loss {}".format(best_model_state_epoch, best_loss))
        model.load_state_dict(best_model_state)

    return model

def exponential_increase(start=1, end=2, num_points=100):
    # Calculate the decay rate
    rate = np.log(end / start) / (num_points - 1)

    # Generate the decay values
    values = start * np.exp(rate * np.arange(num_points))
    return values


class STGPruner(object):
    def __init__(self, n_nodes, A_init, order, params, binary_outcome=True):
        super(STGPruner, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.n_nodes = n_nodes
        self.A_init = A_init
        self.order = order
        self.params = params
        self.binary_outcome = binary_outcome
        self.A_pruned = deepcopy(self.A_init)
        self.prob_selection = {}

    def prune_all(self, data):
        for j, node in enumerate(self.order):
            parents = torch.where(self.A_init[:, node] == 1)[0]
            if len(parents) < 2:
                continue
            pruner = SinglePruner(self.A_init, node, self.params, self.binary_outcome, j)
            print("---------------------------- Start pruning node {} ----------------------------".format(node))
            pruner.prune(data)
            self.A_pruned[:, node][pruner.pruned] = 0
            prob_dict = {}
            if len(pruner.selected) == 1:
                prob_dict[pruner.selected.item()] = pruner.prob_selection.detach().cpu().item()
            else:
                for i in range(len(pruner.selected)):
                    prob_dict[pruner.selected[i].item()] = pruner.prob_selection.detach().cpu()[i].item()
            self.prob_selection[node] = prob_dict
        return


class SinglePruner(object):
    def __init__(self, A_init, leaf, params, binary_outcome, j):
        super(SinglePruner, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.A_init = A_init
        self.leaf = leaf
        self.parents = torch.where(self.A_init[:, self.leaf] == 1)[0]
        self.h_dim = params['h_dim']
        self.n_layers = params['n_layers']
        self.batch_size = params['batch_size']
        self.lr = params['lr']
        self.step_size = params['step_size']
        self.gamma = params['gamma']
        self.n_epochs = params['n_epochs']
        increase = exponential_increase(params['coef'], 2*params['coef'], A_init.shape[0])
        self.coef = increase[j]
        self.sigma = params['sigma']
        self.binary_outcome = binary_outcome
        self.predictor = STGPruningNetwork(len(self.parents), 1, self.h_dim, self.n_layers,
                                           sigma=self.sigma, binary_outcome=self.binary_outcome)
        self.pruned = torch.empty(0)
        self.selected = torch.empty(0)
        self.prob_selection = torch.empty(0)

    def prune(self, data):
        x = data[:, self.parents]
        if self.binary_outcome:
            y = (data[:, self.leaf] > 0).long()
        else:
            y = data[:, self.leaf]
        train_data, val_data = random_split(TensorDataset(x, y), [0.8, 0.2])
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=True, drop_last=True)
        optimizer = torch.optim.Adam(self.predictor.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
        self.predictor = train_STG_single(self.predictor, train_loader, val_loader, optimizer, scheduler,
                                          self.n_epochs, self.coef, binary_outcome=self.binary_outcome,
                                          early_stopping=True, verbose=False)
        self.pruned = self.parents[self.predictor.pruned.cpu()]
        self.selected = self.parents[self.predictor.selected.cpu()]
        self.prob_selection = self.predictor.prob_selection

        return
