"""
Main C1C code 

Several functions taken or modified from:
https://github.com/ssarfraz/FINCH-Clustering/blob/master/finch/finch.py

Reference:
1. Vicky Kalogeiton, and Andrew Zisserman
“Constrained Video Face Clustering using 1NN Relations”, In BMVC, 2020

2. 
M. Saquib Sarfraz and Vivek Sharma and Rainer Stiefelhagen}, 
“Efficient Parameter-free Clustering Using First Neighbor Relations”, In CVPR, 2019

(c) Vicky Kalogeiton 2021
"""

__author__ = "Vicky Kalogeiton"


import time
import argparse
import numpy as np
from sklearn import metrics
import scipy.sparse as sp
import warnings
import networkx as nx
from copy import deepcopy

try:
    from pyflann import *
    pyflann_available = True
except Exception as e:
    warnings.warn('pyflann not installed: {}'.format(e))
    pyflann_available = False
    pass

RUN_FLANN = 70000


def clust_rank(mat, initial_rank=None, distance='cosine', cannot_link={}):
    s = mat.shape[0]
    if initial_rank is not None:
        orig_dist = []
    elif s <= RUN_FLANN:
        orig_dist = metrics.pairwise.pairwise_distances(mat, mat, metric=distance)
        np.fill_diagonal(orig_dist, 1000.0)
        
        for (i, j) in cannot_link:
            orig_dist[i, j] = 1000.0
            orig_dist[j, i] = 1000.0 # to ensure symetry over diagonal 
        
        initial_rank = np.argmin(orig_dist, axis=1)
        #import pdb; pdb.set_trace()
    else:
        if not pyflann_available:
            raise MemoryError("You should use pyflann for inputs larger than {} samples.".format(RUN_FLANN))
        print('Using flann to compute 1st-neighbours at this step ...')
        flann = FLANN()
        result, dists = flann.nn(mat, mat, num_neighbors=2, algorithm="kdtree", trees=8, checks=128)
        initial_rank = result[:, 1]
        orig_dist = []
        print('Step flann done ...')

    # The Clustering Equation
    A = sp.csr_matrix((np.ones_like(initial_rank, dtype=np.float32), (np.arange(0, s), initial_rank)), shape=(s, s))
    A = A + sp.eye(s, dtype=np.float32, format='csr')
    A = A @ A.T

    A = A.tolil()
    A.setdiag(0)
    return A, orig_dist


def get_clust(a, orig_dist, min_sim=None):
    if min_sim is not None:
        a[np.where((orig_dist * a.toarray()) > min_sim)] = 0

    num_clust, u = sp.csgraph.connected_components(csgraph=a, directed=True, connection='weak', return_labels=True)
    return u, num_clust


def cool_mean(M, u):
    _, nf = np.unique(u, return_counts=True)
    idx = np.argsort(u)
    M = M[idx, :]
    M = np.vstack((np.zeros((1, M.shape[1])), M))

    np.cumsum(M, axis=0, out=M)
    cnf = np.cumsum(nf)
    nf1 = np.insert(cnf, 0, 0)
    nf1 = nf1[:-1]

    M = M[cnf, :] - M[nf1, :]
    M = M / nf[:, None]
    return M


def get_merge(c, u, data):
    if len(c) != 0:
        _, ig = np.unique(c, return_inverse=True)
        c = u[ig]
    else:
        c = u

    mat = cool_mean(data, c)
    return c, mat


def update_adj(adj, d):
    # Update adj, keep one merge at a time
    idx = adj.nonzero()
    v = np.argsort(d[idx])
    v = v[:2]
    x = [idx[0][v[0]], idx[0][v[1]]]
    y = [idx[1][v[0]], idx[1][v[1]]]
    a = sp.lil_matrix(adj.get_shape())
    a[x, y] = 1
    return a


def req_numclust(c, data, req_clust, distance, cannot_link):
    iter_ = len(np.unique(c)) - req_clust
    c_, mat = get_merge([], c, data)
    for i in range(iter_):
        adj, orig_dist = clust_rank(mat, initial_rank=None, distance=distance, cannot_link=cannot_link)
        
        idx = adj.todense().nonzero()
        if np.all(orig_dist[idx] == 1000.):
            # We cannot merge anything without violating a cannot-link constraint
            # Relaxing all cannot-link constraints for the following iterations
            cannot_link = {}
            adj, orig_dist = clust_rank(mat, initial_rank=None, distance=distance)
        
        adj = update_adj(adj, orig_dist)
        u, _ = get_clust(adj, [], min_sim=None)
        c_, mat = get_merge(c_, u, data)
        
        cannot_link = {(u[i], u[j]) for i, j in cannot_link if u[i] != u[j]}
        
    return c_


def enforce_cannot_constraints(adj, orig_dist, u_old, cannot_link, num_clust_curr, k, distance):
    """Enforce cannot constraints when they are not working because of chaining NN
    - Inputs:
    adj: the adj matrix NxN
    orig_dist: the adj matrix with the distances instead of 1 and 0, shape: NxN
    u_old: current cluster assignements: shape: Nx1 with K different values, K being the new cluster values
    num_clust_curr: K+1 (since indecing starts from 0) - the actual number of current different clusters 
    cannot_link: the tuples of cannot link constraints
    k: the partition number
    - Outputs:
    u_old: the cluster assignments
    num_clust_curr: the number of clusters  
    """
    for i, j in cannot_link:
        if u_old[i] == u_old[j]:
            idx = np.where(u_old == u_old[i])[0]

            # Adding 1% to the max value in order to have no edges with zero capacity,
            # hence having a non-zero cost in the min-cut max-flow algorithm
            max_value = max([orig_dist[a, b] for a in idx for b in idx if adj[a,b]]) * 1.01
            
            G = nx.Graph()
            for a in idx:
                for b in idx:
                    if adj[a, b]:
                        G.add_edge(a, b, capacity=max_value - orig_dist[a, b])
            
            _, (side_i, side_j) = nx.algorithms.flow.minimum_cut(G, i, j)
            
            u_new = deepcopy(u_old)
            
            for jj in side_j:
                u_new[jj] = num_clust_curr
                
            return enforce_cannot_constraints(adj, orig_dist, u_new, cannot_link, num_clust_curr + 1, k, distance)
            
    return u_old, num_clust_curr


def c1c(data, initial_rank=None, req_num_cls=None, distance='euclidean', verbose=True, must_link={}, cannot_link={}):
    """ C1C clustering algorithm.
    :param data: Input matrix with features in rows.
    :param initial_rank: Nx1 first integer neighbor indices (optional).
    :param req_clreq_num_clsust: Set output number of clusters (optional). Not recommended.
    :param distance: One of ['cosine', 'euclidean', 'l2'] Recommended 'euclidean'.
    :param verbose: Print verbose output.
    :return:
            c: NxP matrix where P is the partition. Cluster label for every partition.
            num_clust: Number of clusters.
            req_cls: Labels of required clusters (Nx1). Only set if `req_num_cls` is not None.
    
    """
    # Cast input data to float32
    data = data.astype(np.float32)
    cannot_links = {'initial': cannot_link}

    min_sim = None
    adj, orig_dist = clust_rank(data, initial_rank, distance, cannot_links['initial'])
    
    if must_link:
        adj = sp.coo_matrix(([1] * len(must_link), ([i for i, _ in must_link], [j for _, j in must_link])), shape=adj.shape)
        adj += adj.T

    initial_rank = None
    group, num_clust = get_clust(adj, [], min_sim)
    c, mat = get_merge([], group, data)
    
    cannot_links[0] = {(group[i], group[j]) for i, j in cannot_links['initial'] if group[i] != group[j]}

    if verbose:
        print('Partition 0: {} clusters'.format(num_clust))
    if len(orig_dist) != 0:
        min_sim = np.max(orig_dist * adj.toarray())

    exit_clust = 2
    c_ = c
    k = 1
    num_clust = [num_clust]
    #import pdb; pdb.set_trace()
    while exit_clust > 1:
        adj, orig_dist = clust_rank(mat, initial_rank, distance, cannot_links[k-1])
        u_tmp, num_clust_curr_tmp = get_clust(adj, orig_dist, min_sim)
        u, num_clust_curr = enforce_cannot_constraints(adj, orig_dist, u_tmp, cannot_links[k-1], num_clust_curr_tmp, k, distance)
        
        c_, mat = get_merge(c_, u, data)

        num_clust.append(num_clust_curr)
        c = np.column_stack((c, c_))
        exit_clust = num_clust[-2] - num_clust_curr

        if num_clust_curr == 1 or exit_clust < 1:
            num_clust = num_clust[:-1]
            c = c[:, :-1]
            break

        if verbose:
            print('Partition {}: {} clusters'.format(k, num_clust[k]))
            
        cannot_links[k] = {(u[i], u[j]) for i, j in cannot_links[k-1] if u[i] != u[j]}
        
        k += 1

    if req_num_cls is not None:
        # Corner case: the initial number of clusters is larger than the actual req number of clusters 
        if req_num_cls not in num_clust:
            ind = [i for i, v in enumerate(num_clust) if v >= req_num_cls]            
            req_cls = None
            if cannot_link:
                req_cls = req_numclust(c[:, ind[-1]], data, req_num_cls, distance, cannot_links[ind[-1]])
            else:
                req_cls = req_numclust(c[:, ind[-1]], data, req_num_cls, distance, {})
        else:
            req_cls = c[:, num_clust.index(req_num_cls)]
    else:
        req_cls = None
       

    return c, num_clust, req_cls


def main():
    print('Not implemented..')


if __name__ == '__main__':
    main()
