#!/usr/bin/env python

import numpy as ny
import networkx as nx
import matplotlib.pyplot as plt
from scipy.cluster.vq import vq, kmeans, whiten
from sklearn.cluster import KMeans
import random

size = 500

p = 0.2

order_map = {}
group_num = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}
limits_num = {0:-1, 1:-1, 2:-1, 3:0, 4:0, 5:0, 6:-1, 7:-1, 8:-1, 9:-1}
colors = {0:'r', 1:'k', 2:'b', 3:'g', 4:'y', 5:'m', 6:'c', 7:'#808000', 8:'#808000', 9:'#808080', 10:'r'}
nodes_added = set()

skip = 0

class People:
    def __init__(self, strr):
        s = strr.strip().split(' ')
        self.order = int(s[0])
        self.id = int(s[1])
        self.name = s[2]
        self.tie = int(s[3])
        self.group = int(s[4])

def valid(idx):
    if idx in nodes_added:
        return True
    return True

    assert group < 10 and group >= 0, "group %d" % (group)
    if limits_num[group] != -1 and group_num[group] >= limits_num[group] :
        return False

    group_num[group] += 1
    nodes_added.add(idx)
    return True

def read_data():
    for l in open('data.txt', 'r'):
        ppl = People(l)
        order_map[ppl.order] = ppl

    trips = []
    for l in open('tie.txt', 'r'):
        i, j, k = [int(x) for x in l.strip().split(' ')]
        if valid(i) and valid(j):
            if True:
            # if i == 1 or j == 1 or order_map[i].group == order_map[j].group:
                trips.append((i, j, k))
    return trips

def build_matrix(data):
    base = ny.zeros((size, size))
    for tri in data:
        i, j, k = tri
        if base[i-1][j-1] == 0:
            base[i-1][j-1] = -1
            base[j-1][i-1] = -1
            base[i-1][i-1] += 1
            base[j-1][j-1] += 1
    return base

def calculate_radix(base):
    print base
    eig_d, eig_v = ny.linalg.eig(base)
    eig_d.sort()
    print eig_d
    return eig_v[1], eig_v[2]

def draw_data(labels, g_orig, g_sparse, pos1, graph_pos):
    # g_new = nx.Graph()

    all_label = set(labels)
    print len(all_label)
    print 'all_labels:', labels

    node_sets = {i:[] for i in all_label}
    edge_sets = {i:[] for i in all_label}

    for i in xrange(len(labels)):
        node_sets[labels[i]].append(i+1)

    for i in all_label:
        for v1 in node_sets[i]:
            for v2 in node_sets[i]:
                if g_sparse.has_edge(v1, v2):
                    edge_sets[i].append((v1, v2))

    outline_edges = []
    for v1 in xrange(len(labels)):
        for v2 in xrange(v1+1, len(labels)):
            if g_sparse.has_edge(v1+1, v2+2) and labels[v1] != labels[v2]:
                print v1, v2, labels[v1], labels[v2]
                outline_edges.append((v1+1, v2+1))
    print 'len of outline:', len(outline_edges)

    # pos = nx.spring_layout(g)
    # print pos

    pos = pos1
    print 'pos[1] = ', pos[1]

    if True:
        plt.close()
        # plt.subplot(graph_pos)
        for i in all_label:
            nx.draw_networkx_nodes(g_orig, pos, nodelist = node_sets[i], node_color=colors[i], node_size = 10, alpha=0.5)
            nx.draw_networkx_edges(g_orig, pos, edgelist = edge_sets[i], edge_color=colors[i], width = 1, alpha=0.5)

        nx.draw_networkx_edges(g_orig, pos, edgelist = outline_edges, edge_color=colors[len(all_label)], width = 1, alpha=0.5)
        plt.axis('off')
        plt.savefig("%d.png" % (graph_pos))
        plt.close()
        # plt.savefig("a.png")

def build_graph_from_data(data):
    g = nx.Graph()
    nodes_group = {}
    lines_group = {}

    for i in xrange(10):
        nodes_group[i] = set()
        lines_group[i] = set()
    lines_group[10] = set()

    for tri in data:
        i, j, k = tri
        g.add_edge(i, j, color = 'r')

        nodes_group[order_map[i].group].add(i)
        nodes_group[order_map[j].group].add(j)

        if order_map[i].group == order_map[j].group:
            lines_group[order_map[i].group].add((i, j))
        elif i == 1:
            lines_group[order_map[j].group].add((i, j))
        elif j == 1:
            lines_group[order_map[i].group].add((i, j))
        else:
            lines_group[10].add((i, j))

    pos = nx.spring_layout(g)

    if False:
        for i in xrange(10):
            if len(nodes_group[i]) != 0:
                nx.draw_networkx_nodes(g, pos, nodelist = list(nodes_group[i]), node_color=colors[i], node_size = 10, alpha=0.5)
        for i in xrange(10):
            if len(lines_group[i]) != 0:
                nx.draw_networkx_edges(g, pos, edgelist = list(lines_group[i]), edge_color=colors[i], width = 1, alpha=0.5)
    return g

def random_graph():
    k_size = 4
    node_size = 100
    group_edge_size = 300
    cross_size = 20

    adj_matrix = [[0] * node_size for i in xrange(node_size)]

    groups = [[], [], [], []]
    group_size = node_size / k_size

    for i in xrange(k_size):
        groups[i] = [1 + j + i * group_size for j in xrange(group_size)]

    g = nx.Graph()
    for i in xrange(node_size):
        g.add_node(i+1)

    group_edges = {i:[] for i in xrange(k_size)}
    outlier_edges = []

    for i in xrange(k_size):
        for j in xrange(group_edge_size):
            v1 = random.randint(0, len(groups[i])-1)
            v2 = random.randint(0, len(groups[i])-1)
            if v1 != v2:
                g.add_edge(groups[i][v1], groups[i][v2])
                group_edges[i].append((groups[i][v1], groups[i][v2]))

    for i in xrange(k_size):
        for j in xrange(i+1, k_size):
            print i, j
            for t in xrange(cross_size):
                v1 = random.randint(0, len(groups[i])-1)
                v2 = random.randint(0, len(groups[j])-1)
                # print 'connect:', groups[i][v1], groups[j][v2]
                g.add_edge(groups[i][v1], groups[j][v2])
                outlier_edges.append((groups[i][v1], groups[j][v2]))

    pos = nx.spring_layout(g)

    # plt.subplot(221)
    if True:
        plt.close()
        print 'pos[1]2 =', pos[1]
        for i in xrange(k_size):
            nx.draw_networkx_nodes(g, pos, nodelist = groups[i], node_color=colors[i], node_size = 10, alpha=0.5)
            nx.draw_networkx_edges(g, pos, edgelist = group_edges[i], edge_color=colors[i], width = 1, alpha=0.5)

        nx.draw_networkx_edges(g, pos, edgelist = outlier_edges, edge_color=colors[k_size], width = 1, alpha=0.5)
        plt.axis('off')
    # plt.savefig("a.png")
        plt.savefig("%d.png" % (221))
        plt.close()

    return g, pos, groups

def sparse_graph(g):
    g2 = nx.Graph()

    for n in g.nodes_iter():
        g2.add_node(n)

    for e in g.edges_iter():
        if random.random() < p:
            # print e
            g2.add_edge(e[0], e[1])
    return g2

def sparse_run(g, pos1):

    g2 = sparse_graph(g)

    # pos1 = nx.spring_layout(g)
    pos2 = nx.spring_layout(g2)

    features = []
    for u in g2.nodes_iter():
        # print type(u)
        # print u
        # print pos[u]
        features.append(pos2[u])
    print "featurs:", len(features)
    features = ny.array(features)

    method = 2
    if method == 1:
        whitened = whiten(features)
        book = ny.array((whitened[0],whitened[2]))
        km = kmeans(whitened, book)

        print km
    elif method == 2:
        n_digits = 4
        km = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
        res = km.fit(features)
        print len(km.labels_), km.labels_
        print res
    return km.labels_, g2

def group_similar(group1, group2):
    rate = 0
    for g2 in group2:
        if g2 in group1:
            rate += 1
    if (len(group2) > 10 and rate > len(group2) / 3) or rate > len(group2) / 2:
        return True
    return False

def relabel(node_groups, sparse_labels):
    k = len(node_groups)
    label_map = {}

    all_pos_labels = set(sparse_labels)

    old_map = {i:[] for i in all_pos_labels}
    for i in xrange(len(sparse_labels)):
        old_map[sparse_labels[i]].append(i)

    for i in all_pos_labels:
        for j in xrange(k):
            if group_similar(node_groups[j], list(old_map[i])):
                assert not label_map.has_key(i)
                label_map[i] = j
    if len(label_map) == k-1:
        set1 = set(label_map.keys())
        set2 = set(label_map.values())
        set_all = set([i for i in xrange(k)])
        p1 = list(set_all - set1)[0]
        p2 = list(set_all - set2)[0]
        label_map[p1] = label_map[p2]

    assert len(label_map) == k, '%d %d' % (len(label_map), k)
    res_labels = []
    for l in sparse_labels:
        res_labels.append(label_map[l])
    return res_labels

def recover(sparse_labels, g, node_groups):
    recover_labels_map = {}
    to_cover = [i+1 for i in xrange(len(sparse_labels))]

    all_labels = set(sparse_labels)

    iterations = 1000000
    iter_times = 0

    while len(to_cover) != 0 and iter_times < iterations:
        iter_times += 1
        if iter_times % 10000 == 0:
            print 'iter: ', iter_times

        node_a = to_cover[0]
        to_cover = to_cover[1:]

        counts = [0 for i in xrange(4)]

        for node_neibo in g.nodes_iter():
            if g.has_edge(node_neibo, node_a):
                assert node_neibo-1 < len(sparse_labels), '%d %d' % (len(sparse_labels), node_neibo)
                counts[sparse_labels[node_neibo-1]] += 1
                if recover_labels_map.has_key(node_neibo-1):
                    counts[recover_labels_map[node_neibo-1]] += 1

        m = max(counts)
        l = sum(counts)
        if m * 1.8 >= l:
            for i in xrange(len(counts)):
                if counts[i] == m:
                    recover_labels_map[node_a-1] = i
                    break
        else:
            to_cover.append(node_a)
    return_labels = []
    for i in xrange(len(sparse_labels)):
        return_labels.append(recover_labels_map[i] if recover_labels_map.has_key(i) else 0)
    return return_labels



def build_graph():
    gen_graph = 'random'
    if gen_graph == 'data':
        data = read_data()
        g = build_graph_from_data(data)
    elif gen_graph == 'random':
        g, pos1, node_groups = random_graph()

    sparse_labels, g2 = sparse_run(g, pos1)
    sparse_labels = relabel(node_groups, sparse_labels)
    draw_data(sparse_labels, g, g2, pos1, graph_pos = 222)

    recover_labels = recover(sparse_labels, g, node_groups)
    draw_data(recover_labels, g, g2, pos1, graph_pos = 223)

    error_num = 0
    for i in xrange(g.number_of_nodes()):
        l = recover_labels[i]
        if i not in node_groups[l]:
            error_num += 1
    print 'error rate: %.2f' % (error_num * 1.0 / g.number_of_nodes())

    return g

def run():
    plt.clf()

    g = build_graph()
    plt.axis('off')

    plt.savefig("a.png")
    # plt.show()


if __name__ == "__main__":
    run()
