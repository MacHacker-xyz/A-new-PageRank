"""Utilities for data manipulation."""
import numpy as np
import pandas as pd

def graph_reader(path):
    """
    Function to read the graph from the path.
    :param path: Path to the edge list.
    :return graph: NetworkX object returned.
    """
    file = open(path)
    index = []
    for line in file:
        values = line.split(",")
        index.append([int(values[0]),int(values[1])])
    
    graph = np.zeros((2708,2708), dtype = np.float64)
    for [i,j] in index:
        graph[i,j] = 1.0
    return graph

def target_reader(path):
    """
    Reading the target vector from disk.
    :param path: Path to the target.
    :return target: Target vector.
    """
    target = np.array(pd.read_csv(path)["target"])
    return target

def normalize_adjacency_matrix(P):
    """
    Creating a normalized adjacency matrix with self loops.
    :param A: Sparse adjacency matrix.
    :param I: Identity matrix.
    :return A_tile_hat: Normalized adjacency matrix.
    """
    
    for i in range(P.shape[0]):
        if np.sum(P[i,:])==0:
            P[i,:]=1/2708
            continue
        P[i,:] = P[i,:] / np.sum(P[i,:])
    return P