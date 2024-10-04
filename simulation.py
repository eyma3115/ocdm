import os
import numpy as np
import pandas as pd
import pickle
import igraph as ig
import networkx as nx
import random

from sklearn.gaussian_process.kernels import RBF

def sample_onehot(shape, p=0.2):
    return (np.random.rand(shape[0], shape[1]) < p).astype(int)


def connect_layers(graphs, p=0.2):
    n_per_stage = [0]
    n_per_stage += [graph.shape[0] for graph in graphs]
    n_upto_stage = [sum(n_per_stage[:i + 1]) for i in range(len(n_per_stage))]
    n_nodes = sum(n_per_stage)
    full_dag = np.zeros([n_nodes, n_nodes])
    for i, graph in enumerate(graphs):
        full_dag[n_upto_stage[i]:n_upto_stage[i + 1], n_upto_stage[i]:n_upto_stage[i + 1]] = graph
        if i == len(graphs):
            break
        n_desc = n_upto_stage[-1] - n_upto_stage[i + 1]
        connection = sample_onehot((n_per_stage[i + 1], n_desc), p)
        full_dag[n_upto_stage[i]:n_upto_stage[i + 1], n_upto_stage[i + 1]:] = connection

    return full_dag


def generate_layeredDAG(n_layers, stage_degree, p_intra, p_inter):
    graphs = []
    stage_set = []
    if type(stage_degree) == int:
        last_node = 0
        for i in range(n_layers):
            graph = ig.Graph.Erdos_Renyi(n=stage_degree, p=p_intra, directed=False, loops=False)
            graph.to_directed(mode='acyclic')
            graph = nx.adjacency_matrix(graph.to_networkx()).todense()
            graphs.append(graph)
            stage_set.append(list(range(last_node, last_node + stage_degree)))
            last_node = stage_degree * (i + 1)
    else:
        last_node = 0
        for i, degree in enumerate(stage_degree):
            graph = ig.Graph.Erdos_Renyi(n=stage_degree[i], p=p_intra, directed=False, loops=False)
            graph.to_directed(mode='acyclic')
            graph = nx.adjacency_matrix(graph.to_networkx()).todense()
            graphs.append(graph)
            stage_set.append(list(range(last_node, sum(stage_degree[:i + 1]))))
            last_node = sum(stage_degree[:i + 1])
    full_dag = connect_layers(graphs, p_inter)

    return full_dag, stage_set


def gp_sampling(x):
    kernel = RBF()
    K = kernel(x, x)
    f = np.random.multivariate_normal(np.zeros(x.shape[0]), K, 1)
    return f


def gp_randomData(dag, n=5000):
    data = np.zeros([n, dag.shape[0]])
    for i in range(dag.shape[0]):
        print("Generating data of the {}-th dimension".format(i))
        if np.count_nonzero(dag[:, i]) == 0:
            noise = np.random.normal(loc=0, scale=np.random.uniform(0.4, 0.8), size=n)
            x = np.random.normal(loc=0, scale=np.random.uniform(1, 2), size=n) + noise
            data[:, i] = x
        else:
            parents = data[:, np.nonzero(dag[:, i])[0]]
            x = gp_sampling(parents) + np.random.normal(loc=0, scale=np.random.uniform(0.4, 0.8), size=n)
            data[:, i] = x
    return data

if __name__ == "__main__":
    dag, stage = generate_layeredDAG(n_layers, stage_degree, p_intra, p_inter)
    data = gp_randomData(dag, n_sample)