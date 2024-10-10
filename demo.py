import pickle
import numpy as np
import pandas as pd
import torch
from diffan_multistage import DiffAN_multistage
from STG_pruning import *
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
    data = pd.read_csv('./data_demo/data.csv', index_col=0)
    with open('./data_demo/stage.pkl', 'rb') as f:
        stage_set = pickle.load(f)
    #Causal order search
    order, dag_oc = multistage_order_search(data, stage_set)

    #STG-pruning
    pruner = STGPruner(len(order), torch.Tensor(dag_oc), order, params, binary_outcome=False)
    pruner.prune_all(torch.Tensor(data.values))

