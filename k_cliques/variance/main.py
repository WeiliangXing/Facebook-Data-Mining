#!/usr/bin/env python


import random
import time
import math

max_clique = 9

def rate(k):
    return k * (k-1) / 2

def is_clique2(mat, k):
    l = 0
    for i in xrange(len(k)):
        for j in xrange(i+1, len(k)):
            if mat[k[i]][k[j]] != 0:
                l += 1
    t = rate(len(k))
    return l >= t / 4

def is_clique(mat, k):
    for i in xrange(len(k)):
        for j in xrange(i+1, len(k)):
            # print k[i], k[j]
            if mat[k[i]][k[j]] == 0:
                return False
    return True

def get_clique2(mat):
    size_mat = len(mat)
    result = {}
    for i in xrange(2, max_clique):
        result[i] = 0
        k = [0]

        while len(k) != 0:
            if k[-1] == size_mat:
                k = k[:-1]
                if len(k) != 0:
                    k[-1] += 1
                continue

            if not is_clique(mat, k):
                k[-1] += 1
                continue

            if len(k) < i:
                k.append(k[-1]+1)
            else:
                result[len(k)] += 1
                k[-1] += 1
        # print i, ':', result[i]

    return result

def all_connect(mat, prev, j):
    for p in prev:
        if mat[p][j] == 0:
            return False
    return True

def get_clique(mat):
    size_mat = len(mat)
    result = {2:0}
    result_cache = []
    costs = {}

    start_time = time.time()

    next_cache = []
    for i in xrange(size_mat):
        for j in xrange(i, size_mat):
            if mat[i][j] != 0:
                result[2] += 1
                result_cache.append([i, j])
    end_time = time.time()
    # print '2 %d %f' % (result[2], end_time - start_time)
    costs[2] = end_time - start_time

    for i in xrange(3, max_clique):
        result[i] = 0
        next_cache  = []

        start_time = time.time()
        for prev in result_cache:
            start = max(prev)+1
            for j in xrange(start, size_mat):
                if all_connect(mat, prev, j):
                    next_cache.append(prev + [j])
                    result[i] += 1

        end_time = time.time()
        result_cache = next_cache
        # print i, ':', result[i]
        print '%d %d %f' % (i, result[i], end_time - start_time)
        costs[i] = end_time - start_time

    return result, costs

def read_data():
    data_filename = 'data/raw_data.txt'
    with open(data_filename, 'r') as f:
        num_nodes = int(f.readline())
        num_edges = int(f.readline())
        adj_matrix = [[0] * num_nodes for i in xrange(num_nodes)]
        print len(adj_matrix)

        for l in f:
            (a, b) = eval(l)
            a = int(a)-1
            b = int(b)-1
            adj_matrix[a][b] = 1
            adj_matrix[b][a] = 1

    return adj_matrix

def get_doulion_sparse(adj_matrix, p):
    size_matrix = len(adj_matrix)
    sparse_mat = [[0] * size_matrix for i in xrange(size_matrix)]
    edges = 0
    orig_edges = 0
    for i in xrange(size_matrix):
        for j in xrange(i+1, size_matrix):
            if adj_matrix[i][j] != 0:
                orig_edges += 1
                if random.random() < p:
                    edges += 1
                    sparse_mat[i][j] = 1
                    sparse_mat[j][i] = 1
    # print "edge: %d vs %d with p %.2f" % (orig_edges, edges, p)
    return sparse_mat

def get_complete():
    size = 100

    adj_matrix = [[1] * size for i in xrange(size)]
    for i in xrange(size):
        adj_matrix[i][i] = 0

    return adj_matrix

if __name__ == "__main__":
    method = 'complete'
    if method == 'complete':
        adj_matrix = get_complete()
    elif method == 'read':
        adj_matrix = read_data()
    # print adj_matrix

    compare_res = {}
    compare_res, compare_costs = get_clique(adj_matrix)
    print compare_res
    print compare_costs

    average_res = {i:0 for i in xrange(2, max_clique)}
    average_mid = {i:0 for i in xrange(2, max_clique)}
    all_data = {i:[] for i in xrange(2, max_clique)}

    num_rounds = 100
    costs = {i:0 for i in xrange(2, max_clique)}
    for i in xrange(num_rounds):
        print "round:", i
        p = 0.7
        sparse_mat = get_doulion_sparse(adj_matrix, p)
        sparse_mid, sub_costs = get_clique(sparse_mat)
        sparse_res = {k:int(v * ((1/p) ** rate(k))) for k, v in sparse_mid.items()}

        for k, v in sub_costs.items():
            costs[k] += v

        for j in average_res.keys():
            if sparse_res.has_key(j):
                average_res[j] += sparse_res[j]
                average_mid[j] += sparse_mid[j]
                all_data[j].append(sparse_res[j])

    average_res = {k:(v / num_rounds) for k, v in average_res.items()}
    costs = {k:(v / num_rounds) for k, v in costs.items()}

    vars = {}
    for i in xrange(2, max_clique):
        assert average_res.has_key(i)
        assert all_data.has_key(i)
        vars[i] = sum((v - average_res[i])*(v - average_res[i]) for v in all_data[i]) / num_rounds

    print average_res.keys()
    print 'average_mid = ', average_mid.values()
    print 'compare_res = ', compare_res.values()
    print 'average_res = ', average_res.values()
    print 'costs = ', costs.values()
    print 'vars = ', vars.values()
