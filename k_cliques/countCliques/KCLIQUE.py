__author__ = 'weiliangxing'

import random
import math
import networkx as nx
import datetime
import resource
from matplotlib import pylab as pl
import itertools


class SubList:
    """
    sublist for the algorithm
    """
    def __init__(self):
        # base_nodes are current nodes
        # common_neighbors are defined as the nodes which is outside the cliques
        # and connected to the clique
        self.base_nodes = []
        self.common_neighbors = []

class KCliques_Yun:
    """
    the class to calculate KCliques
    It could generate all kinds of cliques to maximum cliques
    """
    def __init__(self, graph, k_threshold=float('inf')):
        """
        :param graph: the GenerateGraph object
        :param k_range: a list that indicates k's range. To get statistical results for particular
        k, you should set first and last item of k_range same
        """
        self.G = graph.G
        self.prob = graph.prob

        self.num_kcliques = 0
        self.maximum_clique = 0

        #core algorithm: the sequential method to do maximal cliques generation
        self.maximal_cliques_list = self.maximal_cliques_sequential(k_threshold)
        self.maximum_clique = self.get_max_clique()
        self.res_list = []
        self.num_kcliques = 0

    def get_max_clique(self):
        """
        function to get the maximum number of cliques list
        :return:maximum clique's value
        """
        max_clique = []
        for l in self.maximal_cliques_list:
            max_clique.append(len(l))
        if not max_clique:
            max_clique = [0]
        num = max(max_clique)
        return num

    def clique_counting(self, k_range):
        """
        function to generate cliques list
        :param k_range: input k range; by rules, k's low and high value should be same
        :return:None
        """
        # self.res_list is a list with cliques list by particular k range
        self.res_list = self.res_by_clique(k_range)
        # count the number of cliques in particular k and probability
        self.num_kcliques = self.count_kcliques(self.prob, k_range)

    def test_simple_graph(self):
        """
        the function to generate simple graph, cited from figure 4 of this algorithm's paper
        :return: edges list with edge tuples, marked by numbered node
        """
        # a:1,b:2,c:3,d:4,e:5,f:6,g:7
        edges = [(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5), (3, 4),
                 (3, 5), (4, 5), (6, 2), (6, 3), (6, 7), (7, 3), (7, 4), (7, 5)]
        return edges

    def count_kcliques(self, prob, k_range):
        """
        function to count kcliques
        :param prob: input probability
        :param k_range: input k_range
        :return: the number of k cliques
        """

        if k_range[0] != k_range[1]:
            print("In order to count at k , please set k_range with only one number!")
            return len(self.res_list)
        else:
            if prob < 1.0:
                num = self.get_clique_num(k_range[0])
                count = len(self.res_list)
                return round(math.pow(1/prob, num) * count)
            else:
                return len(self.res_list)

    def get_clique_num(self,k):
        """
        function to sum up the cliques
        :param k: input k values
        :return: sum of cliques
        """
        sum = 0
        for i in range(1, k):
            sum += i
        return sum


    def clique_neighbors(self, cur_node, sorted_nodes):
        """
        function to find neighbors of chosen clique
        :param cur_node: current node
        :param sorted_nodes: nodes that sorted as ascending order for better analysis
        :return:all neighbors whose index is greater than cur_node's.
        """
        neighbors = []

        for i in range(len(sorted_nodes)):
            node = sorted_nodes[i]
            if sorted_nodes.index(cur_node) < i:
                if node in self.G.neighbors(cur_node):
                    neighbors.append(node)
        return neighbors

    def init_sublists(self, sorted_nodes):
        """
        function to initialize the sublists
        :param sorted_nodes: sorted list
        :return: initiaized sublists list, contains sublist objects
        """
        sublists = []
        for i in range(len(sorted_nodes)):
            node = sorted_nodes[i]
            temp = []
            sublist = SubList()
            temp.append(node)
            sublist.base_nodes = temp
            sublist.common_neighbors = self.clique_neighbors(node, sorted_nodes)
            sublists.append(sublist)
        return sublists

    def generate_sublists(self, result, sublist, sublists, nodes_sorted, k_threshold):
        """
        function to generate k-sublists for each k+1 clique
        :param result: the result of all k-cliques
        :param sublist: the sublist for particular step
        :param sublists: the sublists from initialized sublists
        :param nodes_sorted: sorted nodes
        :return: result of current loop and sublist for next loop
        """
        threshold = k_threshold
        max_clique_val = 0
        for neighbor in sublist.common_neighbors:
            if max_clique_val >= threshold:
                break
            cur_neighbors = self.clique_neighbors(neighbor, nodes_sorted)

            temp = []
            temp.append(neighbor)
            cur_base_nodes = sublist.base_nodes + temp
            cur_intersected_nodes = set(cur_neighbors).intersection(sublist.common_neighbors)
            cur_common_neighbors = sorted(cur_intersected_nodes)

            if len(cur_base_nodes) > max_clique_val:
                max_clique_val = len(cur_base_nodes)
            if max_clique_val >= threshold:
                break

            result.append(cur_base_nodes)
            # print(str(cur_base_nodes))

            for node in cur_common_neighbors:
                temp = []
                temp.append(node)
                new_base_nodes = cur_base_nodes + temp
                new_neighbors = self.clique_neighbors(node, nodes_sorted)
                new_intersected_nodes = set(cur_common_neighbors).intersection(new_neighbors)
                new_common_neighbors = sorted(new_intersected_nodes)

                if len(new_common_neighbors) < 2:
                    if len(new_base_nodes) > max_clique_val:
                        max_clique_val = len(new_base_nodes)
                    if max_clique_val >= threshold:
                        break
                    result.append(new_base_nodes)
                    # print(str(new_base_nodes))

                    if len(new_common_neighbors) == 1:
                        if len(new_base_nodes + new_common_neighbors) > max_clique_val:
                            max_clique_val = len(new_base_nodes + new_common_neighbors)
                        if max_clique_val >= threshold:
                            break

                        result.append(new_base_nodes + new_common_neighbors)
                        # print(str(new_base_nodes + new_common_neighbors))
                else:
                    # print("candidate: " + str([new_base_nodes, new_common_neighbors]))
                    new_sublist = SubList()
                    new_sublist.base_nodes = new_base_nodes
                    new_sublist.common_neighbors = new_common_neighbors
                    sublists.append(new_sublist)
        return result, sublist

    def maximal_cliques_sequential(self, k_threshold):
        """
        the function to do sequential calculation for maximal cliques
        :return: the final list for all maximal cliques
        """

        nodes_sorted = sorted(self.G.nodes())
        sublists = self.init_sublists(nodes_sorted)
        result = []

        while sublists:
            # here maybe do parallel computing for larger graph
            sublist = sublists.pop(0)
            (result, sublist) = self.generate_sublists(result, sublist, sublists, nodes_sorted, k_threshold)
        return result

    def res_by_clique(self, k_range):
        """
        function to choose cliques wih k in k_range
        :param k_range: input k_range
        :return: the cliques list with k in k_range
        """
        res = []
        max_clique = []
        for l in self.maximal_cliques_list:
            max_clique.append(len(l))
            if k_range[0] <= len(l) <= k_range[1]:
                res.append(l)
        self.maximum_clique = max(max_clique)
        return res

    def draw_origin(self):
        """
        function to draw original graph
        :return:None
        """

        pos = nx.spring_layout(self.G)
        pl.figure(1)
        pl.title("original figure")
        nx.draw(self.G, pos=pos)
        nx.draw_networkx_labels(self.G, pos, font_size=10, font_family='sans-serif')
        pl.show()

    def draw_kcliques(self):
        """
        function to draw graph have k cliques
        :return: None
        """
        new_nodes = []
        for n in self.G.nodes():
            if n in self.res_list:
                new_nodes.append(n)

        subgraph = self.G.subgraph(new_nodes)
        pos = nx.spring_layout(self.G)
        pl.figure(2)
        nx.draw(subgraph, pos=pos)
        nx.draw_networkx_labels(subgraph, pos, font_size=10, font_family='sans-serif')
        pl.show()

class KCliques_Bron:
    def __init__(self, graph, k_threshold=float('inf')):
        self.G = graph.G
        self.prob = graph.prob

        self.num_kcliques = 0
        self.maximum_clique = 0

        self.cliques = nx.find_cliques(self.G)

        self.maximum_clique = self.get_max_clique()

        self.res_list = []
        self.num_kcliques = 0


    def get_max_clique(self):
        max_clique = 0
        for clq in self.cliques:
            if len(clq) > max_clique:
                max_clique = len(clq)
        return max_clique

    def clique_counting(self, k_range):
        """
        function to generate cliques list
        :param k_range: input k range; by rules, k's low and high value should be same
        :return:None
        """
        # self.res_list is a list with cliques list by particular k range
        self.res_list = self.maximal_cliques(k_range)
        # count the number of cliques in particular k and probability
        self.num_kcliques = self.count_kcliques(self.prob, k_range)

    def count_kcliques(self, prob, k_range):
        """
        function to count kcliques
        :param prob: input probability
        :param k_range: input k_range
        :return: the number of k cliques
        """
        if k_range[0] != k_range[1]:
            print("In order to count at k , please set k_range with only one number!")
            return len(self.res_list)
        else:
            if prob < 1.0:
                num = self.get_clique_num(k_range[0])
                count = len(self.res_list)
                return round(math.pow(1/prob, num) * count)
            else:
                return len(self.res_list)

    def get_clique_num(self,k):
        """
        function to sum up the cliques
        :param k: input k values
        :return: sum of cliques
        """
        sum = 0
        for i in range(1, k):
            sum += i
        return sum


    def maximal_cliques(self, k_range):
        lo = k_range[0]
        hi = k_range[1]
        cliques = nx.find_cliques(self.G)
        # cliques_lists = [clq for clq in cliques if lo <= len(clq) <= hi]
        cliquesk = []
        # for clq in cliques:
        #     if len(clq) >= lo:
        #         cliquesk += itertools.combinations(clq, lo)
        # cliquesk = [list(itertools.combinations(clq, 3)) for clq in cliques if len(clq) >= 3]
        cliquesk = set(sum([list(itertools.combinations(set(clq), lo)) for clq in cliques if len(clq)>=lo],[]))

        # print(lo)
        # print(cliquesk)
        return cliquesk