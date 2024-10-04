import numpy as np
import pandas as pd
import torch
from diffan_multistage import DiffAN_multistage
from utils import full_DAG
from simulation import generate_layeredDAG, gp_randomData
from sklearn.preprocessing import StandardScaler

def multistage_order_search(data, stage_set):
    scaler = StandardScaler()
    data.loc[:, :] = scaler.fit_transform(data)
    data = torch.Tensor(data.values)
    num_samples, n_nodes = data.shape
    model = DiffAN_multistage(n_nodes, len(stage_set))
    order = model.fit(data, stage_set)
    dag_oc = full_DAG(order)
    dag_oc_pd = pd.DataFrame(data=dag_oc.astype(int), index=data.columns, columns=data.columns)

    return order, dag_oc_pd

if __name__ == '__main__':
    #Random layered DAG simulation
    n_layers = 10
    stage_degree = 10
    p_intra = 0.2
    p_inter = 0.2
    n_sample = 10000

    dag_gt, stage_set = generate_layeredDAG(n_layers, stage_degree, p_intra, p_inter)
    data = gp_randomData(dag_gt, n_sample)

    dag_gt = pd.DataFrame(data=dag_gt, index=np.arange(dag_gt.shape[0]), columns=np.arange(dag_gt.shape[0]))
    data = pd.DataFrame(data=data, columns=np.arange(dag_gt.shape[0]))

    #Causal order search
    order, dag_oc = multistage_order_search(data, stage_set)

    #STG-pruning
    params = {}
    params['h_dim'] = 128
    params['n_layers'] = 3
    params['batch_size'] = 256
    params['lr'] = 0.0001
    params['step_size'] = 50
    params['gamma'] = 0.99
    params['n_epochs'] = 300
    params['coef'] = 1
    params['sigma'] = 1

    pruner = STGPruner(dag_oc.shape[0], dag_oc, order, params, binary_outcome=False)
    pruner.prune_all(torch.Tensor(data.values))

