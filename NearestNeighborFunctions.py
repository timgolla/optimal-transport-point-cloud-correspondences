# @author Tim Golla <tim.golla.official@gmail.com>

import time
import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

def getKNNGraphsklearn(m1,m2, n_neighbors, algorithm='auto',n_jobs=-1, metric='euclidean'):
    st = time.time()
    n_neighbors=min(n_neighbors,len(m2))
    nbrs12 = NearestNeighbors(n_neighbors, algorithm='auto',n_jobs=-1, metric='euclidean').fit(m2)
    dists,nninds=  nbrs12.kneighbors(m1)
    dists += 1e-7 # wiggle the dists a bit, so the sparse matrix representation is the way we need it
    i = np.arange(dists.shape[0])
    i = np.repeat(i,n_neighbors)
    costmatrix1 = sp.coo_matrix((dists.flatten('C'),(i,nninds.flatten('C'))), shape = (len(m1), len(m2)))
    et = time.time()
    totaltime = et - st
    return costmatrix1, totaltime

def getKNNGraph(m1,m2, n_neighbors):
    return getKNNGraphsklearn(m1,m2, n_neighbors, algorithm='auto',n_jobs=-1, metric='euclidean')
