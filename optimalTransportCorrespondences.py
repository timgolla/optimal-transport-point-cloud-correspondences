# @author Tim Golla <tim.golla.official@gmail.com>

import sys
import argparse
import numpy as np
import time
import ot
import open3d
import scipy.sparse as sp

import NearestNeighborFunctions
import OptimalTransport

def parseArgs():
    parser = argparse.ArgumentParser(description="Generate point cloud correspondences using optimal transport. Author: Tim Golla <tim.golla.official@gmail.com>")
    parser.add_argument('pcfile1')
    parser.add_argument('pcfile2')
    parser.add_argument('outfilename')
    parser.add_argument('--labelfile1', default="")
    parser.add_argument('--labelfile2',default="")
    parser.add_argument("--nn",type=int, default = 10)
    parser.add_argument("--lam",type=float, default = 0.001)
    parser.add_argument("--solver",default="")
    args=parser.parse_args(sys.argv[1:])
    return args

def readFiles(args):
    pc = open3d.read_point_cloud(args.pcfile1)
    v1 = np.asarray(pc.points)
    pc = open3d.read_point_cloud(args.pcfile2)
    v2 = np.asarray(pc.points)
    vertices = [v1,v2]

    if args.labelfile1 != "" and args.labelfile2 != "":
        l1 = np.loadtxt(args.labelfile1,dtype=int)
        l2 = np.loadtxt(args.labelfile2,dtype=int)
    else:
        l1 = np.zeros(v1.shape[0])
        l2 = np.zeros(v2.shape[0])

    # shift labels:
    l1 -= l1.min()
    l2 -= l2.min()

    labels = [l1,l2]

    return vertices,labels

def solveAssignmentPredefinedClusters(vertices,labels,args, normals = None):
    assignedcenters = []
    correspondenceIndices = []
    assignedclusterlabels = []
    assignednormals = []

    nn = args.nn
    lam = args.lam

    timesum=0

    assert(np.max(labels[0]) == np.max(labels[1]))

    # start cluster loop
    for clusternr1 in range(np.max(labels[0]) + 1):
        print("Cluster %u of %u"%(clusternr1, np.max(labels[0])))
        indices1 = np.where(labels[0] == clusternr1)[0]
        indices2 = np.where(labels[1] == clusternr1)[0]
        m1 = vertices[0][labels[0] == clusternr1]
        m2 = vertices[1][labels[1] == clusternr1]
        centers = [m1,m2]
        print("Computing nn graph")
        timesum = 0
        costmatrix1, knntime = NearestNeighborFunctions.getKNNGraph(m1,m2,nn)
        print("nn graph computed")
        timesum += knntime
        localst = time.time()
        P1 = costmatrix1.copy()
        P1.data = P1.data/costmatrix1.max()*lam # stabilization
        P1.data = np.exp(-1* P1.data / lam)
        # assert(np.all(P1.max(axis=1).toarray() > 0))
        print("Computing nn graph")
        localet = time.time()
        localtime = localet - localst
        timesum += localtime
        costmatrix2, knntime = NearestNeighborFunctions.getKNNGraph(m2,m1,nn)
        print("nn graph computed")
        timesum += knntime
        localst = time.time()
        assert(np.all(costmatrix2.max(axis=1).toarray() > 0))
        P2 = costmatrix2.copy()
        P2.data = P2.data/costmatrix2.max()*lam # stabilization
        P2.data = np.exp(-1* P2.data / lam)
        print("Computing final cost and Sinkhorn matrix")
        costmatrix = costmatrix1.maximum(costmatrix2.T)
        P = P1.maximum(P2.T)
        override = False
        try:
            assert(np.all(P.max(axis=0).toarray() > 0))
            assert(np.all(P.max(axis=1).toarray() > 0))
        except AssertionError:
            P.data = np.ones_like(P.data)
            override = True
            print("P matrix has zero maxes. Overriding")
        if override:
            G0=P
        else:
            P = P.tocsr()
            setupendtime = time.time()
            print("Starting optimization")
            stt = time.time()
            localet = time.time()
            localtime = localet - localst
            timesum += localtime
            mass1 = np.ones(len(m1))
            mass2 = np.ones(len(m2))
            if args.solver == "ot.bregman.greenkhorn":
                otstarttime = time.time()
                G0 = ot.bregman.greenkhorn(mass1,mass2,costmatrix.T,reg =lam)
                sinkhorntime = time.time() - otstarttime
            else:
                G0, sinkhorntime = OptimalTransport.sinkhorn_sparse(P,mass1,mass2)
            timesum += sinkhorntime
            ett = time.time()
            print("optimization took: " + str(ett - stt))
            print("G0.shape: " + str(G0.shape))
            print("G0 total entries: " + str(G0.shape[0]*G0.shape[1]))
            print("G0.nnz: " + str(G0.nnz))
            print("G0 occupancy: " + str(G0.nnz/(G0.shape[0]*G0.shape[1])))
            
            localst = time.time()
            removealmostzero = False
            if removealmostzero:
                print("Removing entries < " + str(eps))
                G0.data[np.isclose(G0.data,0,eps,eps)] = 0
                G0.eliminate_zeros()

            userowmaxcolmax = True
            if userowmaxcolmax:
                print("Using only maximum row and col entries")
                rowargmaxes = np.array(G0.argmax(axis=1))[:,0]
                colargmaxes = np.array(G0.argmax(axis=0))[0,:]
                rowinds = np.arange(G0.shape[0])
                colinds = np.arange(G0.shape[1])
                rowmaxdata = G0.max(axis=1)
                colmaxdata = G0.max(axis=0)
                G0_rowmax = sp.coo_matrix((rowmaxdata.toarray()[:,0],(rowinds,rowargmaxes)), shape = G0.shape)
                assert(G0_rowmax.shape == G0.shape)
                G0_colmax = sp.coo_matrix((colmaxdata.toarray()[0],(colargmaxes,colinds)), shape = G0.shape)
                assert(G0_colmax.shape == G0.shape)
                G0_maxes = G0_rowmax.maximum(G0_colmax)
                assert(G0_maxes.shape == G0.shape)
                G0 = G0_maxes
            localet = time.time()
            localtime = localet - localst
            timesum += localtime
        print("G0.nnz: " + str(G0.nnz))
        print("G0 occupancy: " + str(G0.nnz/(G0.shape[0]*G0.shape[1])))

        costmatrix.tocsr()
        wassersteindistance = (costmatrix.multiply(G0)).sum()
        print("Wasserstein distance: " + str(wassersteindistance))
        localst = time.time()

        usemax = False
        
        if usemax:
            for i in range(G0.shape[0]):
                k = G0[i].data.argmax()
                j = G0[i].indices[k]
                assignedcenter = np.hstack([centers[0][i],centers[1][j]])
                assignedcenters.append(assignedcenter)
        else:
            G0 = G0.tocsr()
            I,J,V = sp.find(G0)
            for k in range(len(I)):
                i = I[k]
                j = J[k]
                correspondenceIndices.append([indices1[i],indices2[j]])
                outputClusterColors = False
                if outputClusterColors:
                    assignedcenter = np.hstack([m1_assign[i],m2_assign[j]])
                else:
                    assignedcenter = np.hstack([centers[0][i][:3],centers[1][j][:3]])
                assignedcenters.append(assignedcenter)
                if normals is not None and len(normals[0] > 0) and len(normals[1] > 0):
                    assignednormal = np.hstack([normals[0][i],normals[1][j]])
                    assignednormals.append(assignednormal)
                assignedclusterlabels.append([clusternr1,clusternr1])
    # end cluster loop

    assignedcenters = np.array(assignedcenters)
    if normals is not None and len(normals[0]) > 0 and len(normals[1]) > 0:
        assignednormals = np.vstack(assignednormals)
    localet = time.time()
    localtime = localet - localst
    timesum += localtime
    print("done with generating new centers etc")
    assignedclusterlabels = np.array(assignedclusterlabels)
    correspondenceIndices = np.array(correspondenceIndices)
    return correspondenceIndices, assignedcenters, assignednormals, assignedclusterlabels




def main():
    args = parseArgs()
    vertices, labels = readFiles(args)
    correspondenceIndices = solveAssignmentPredefinedClusters(vertices,labels,args)[0]
    np.savetxt(args.outfilename,correspondenceIndices,fmt="%u")

if __name__ == "__main__":
    main()