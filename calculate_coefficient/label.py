__author__ = 'weiliangxing'

# Data is extracted and cleaned from Give Me My Data for Facebook ego map
# Nodes: 144
# Edges: 802

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def friends_dic(dic):
    """
    build the dictionary: name of friend to the label.
    :param dic: input dictionary
    :return: None
    """
    f = open('friends_label_raw.txt')
    for line in f.readlines():
        case = line.split(",")
        dic[case[0]] = int(case[1])
    f.close()


def relation_tup(dic, tup):
    """
    build the relation-tuple list: label of friend i to label of friend j
    :param dic: input dictionary
    :param tup: input list
    :return: None
    """
    f = open('real_raw_data4_modified.txt')
    for line in f.readlines():
        case = line.split(",")
        if case[0] in dic and case[1] in dic:
            tup.append((dic[case[0]], dic[case[1]]))
    f.close()


def generate_raw_data(dic, tup):
    """
    generate raw_data.txt in the format as the homework requires
    :param dic: input vertices dictionary
    :param tup: input edges pair list
    :return: None
    """
    f = open('raw_data.txt', 'r+')
    f.write("{vertices}\n".format(vertices=len(dic)))
    f.write("{edges}\n".format(edges=len(tup)))
    for i in range(len(tup)):
        f.write("{edge_pair}\n".format(edge_pair=tup[i]))
    f.close()


def generate_degree(graph, x, y):
    """
    build dictionary of dictionary for each x-y node pair
    :param graph: the target graph
    :param x: one node
    :param y: another node
    :return: None
    """
    if x not in graph:
        # build dictionary for node x
        graph[x] = {}
    # in dictionary of dictionary, add link
    graph[x][y] = 1
    if y not in graph:
        graph[y] = {}
    graph[y][x] = 1


def local_coefficient(graph, vertex):
    """
    calculate the local clustering coefficient
    :param graph: the target graph
    :param vertex: the target local vertex
    :return: the local clustering coefficient
    """
    neighbor_nodes = graph[vertex].keys()
    if len(neighbor_nodes) is 1:
        return -1.0
    l = 0
    for n in neighbor_nodes:
        for m in neighbor_nodes:
            if m in graph[n]:
                l += 0.5
    coff = 2.0 * l / (len(neighbor_nodes) * (len(neighbor_nodes) - 1))
    return coff


def clustering_coefficient(graph, tup):
    """
    generate all links in target graph in dictionary of dictionary, and use it to do
    the accumulation and average of each local clustering coefficient to get the
    global clustering coefficient.
    :param graph: The target graph
    :param tup: The raw-data input
    :return: the global clustering coefficient
    """
    sum_coeff = 0
    for (x, y) in tup:
        generate_degree(graph, x, y)
    for vertex in graph:
        sum_coeff += local_coefficient(graph, vertex)
    sum_coeff /= len(graph)
    return sum_coeff


def degree_matrix(graph):
    """
    generate degree matrix
    :param graph: the target graph
    :return:generated degree matrix
    """
    matrix_degree = [[0 for x in range(len(graph))] for x in range(len(graph))]
    for v in graph:
        matrix_degree[v - 1][v - 1] = len(graph[v])
    return matrix_degree


def adjacency_matrix(graph):
    """
    generate adjacency matrix
    :param graph: the target graph
    :return: generated adjacency matrix
    """
    matrix_adjacency = [[0 for x in range(len(graph))] for x in range(len(graph))]
    for v in graph:
        for u in graph[v]:
            matrix_adjacency[v - 1][u - 1] = 1
            matrix_adjacency[u - 1][v - 1] = 1
    return matrix_adjacency


def laplacian_matrix(graph, degree, adjacency):
    matrix = [[0 for x in range(len(graph))] for x in range(len(graph))]
    for i in range(len(adjacency)):
        for j in range(len(adjacency)):
            matrix[i][j] = degree[i][j] - adjacency[i][j]
    return matrix


def plot_matrix(x_coord, y_coord):
    """
    plot the clustering figure in 2D
    :param x_coord: the second smallest vector as x coordinates
    :param y_coord: the third smallest vector as y coordinates
    :return: None
    """
    plt.scatter(x_coord, y_coord)
    for i in range(len(x_coord)):
        plt.text(x_coord[i], y_coord[i], i)
    plt.show()


def plot_3d(x_coord, y_coord, z_coord):
    """
    plot the clustering figure in 3D
    :param x_coord:
    :param y_coord:
    :param z_coord:
    :return:
    """
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x_coord, y_coord, z_coord)
    plt.show()


def main():
    dic = {}
    tup = []
    graph = {}
    friends_dic(dic)
    relation_tup(dic, tup)
    generate_raw_data(dic, tup)
    clustering_coeff = clustering_coefficient(graph, tup)
    matrix_degree = degree_matrix(graph)
    matrix_adjacency = adjacency_matrix(graph)
    matrix_laplacian = laplacian_matrix(graph, matrix_degree, matrix_adjacency)
    eigenvalues, eigenvectors = np.linalg.eig(matrix_laplacian)
    x = eigenvectors[1]
    y = eigenvectors[2]
    z = eigenvectors[3]
    plot_matrix(x, y)
    plot_3d(x, y, z)
    print(clustering_coeff)

if __name__ == "__main__":
    main()