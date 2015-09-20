__author__ = 'weiliangxing'

import random
import math
import networkx as nx
import datetime
import resource
from matplotlib import pylab as pl
from KCLIQUE import *
import sys
from operator import add
import copy


class Result:
    """
    The class to represent the result with detailed lists
    """
    def __init__(self, time_list, nodes_list, edges_list, density_list, cluster_co_list,
                 trans_list, maximum_clique_list, num_cliques_by_prob, prob_list, speed_list=[], accuracy_list=[], total_time_list=[]):
        """
        here initial details list about one result object

        """
        self.time_list = time_list
        self.nodes_list = nodes_list
        self.edges_list = edges_list
        self.density_list = density_list
        self.cluster_co_list = cluster_co_list
        self.trans_list = trans_list
        self.maximum_clique_list = maximum_clique_list
        self.num_cliques_by_prob = num_cliques_by_prob
        self.prob_list = prob_list
        self.speed_list = speed_list
        self.accuracy_list = accuracy_list
        self.total_time_list = total_time_list

class GenerateGraph:
    """
    class to generate graph
    """

    def __init__(self, input_file=None, prob=1.0, sample_type=None, nodes=None, edges=None, degree=None, shrink_ratio=1.0):
        """
        the function is used to generate graph according different kinds of input requirements
        :param input_file: input file as string "foo.txt"
        :param prob: probability value for sampling process
        :param sample_type: two types: "karate_club" and "random_graph"
        :param nodes: if sample_type is "random_graph", here number of nodes must be declared
        :param edges: if sample_type is "random_graph", here number of edges can be declared
        :param degree: if sample_type is "random_graph", here degree of each node can be declared
        :param shrink_ratio: the ratio used to be randomly shrink the input file for large input file
        :return: None
        """
        self.G = nx.Graph()

        if input_file:
            self.edges = self.import_file(input_file, shrink_ratio)
            self.G.add_edges_from(self.edges)
            self.edges = self.G.edges()
        if sample_type:
            if sample_type == "karate_club":
                self.G = nx.karate_club_graph()
                self.edges = self.G.edges()
            elif sample_type == "random_graph":
                if nodes and edges:
                    self.G = nx.gnm_random_graph(nodes, edges)
                    self.edges = self.G.edges()
                elif nodes and degree:
                    w = [nodes*degree for i in range(nodes)] # w = p*n for all nodes
                    self.G = nx.expected_degree_graph(w)  # configuration model
                    self.edges = self.G.edges()
                else:
                    raise Exception('Wrong: ' + "nodes/edges/degree is None!")
            else:
                raise Exception("Wrong: please give input file name!")


        self.random_select_edge_at(prob)
        self.prob = prob

    def random_select_edge_at(self, prob):
        """
        function to do biased coin tossing
        :param prob: input probability
        :return: set self.G with edges which
        are only chosen when random variable is under the prob
        """
        random_var = prob
        remove_list = []
        for edge in self.edges:
            if random.uniform(0, 1) > random_var:
                remove_list.append(edge)
        self.G.remove_edges_from(remove_list)

    def import_file(self, input_file, s):
        """
        for raw_data.txt
        function to parse input txt file into lists
        :param input_file: input txt file
        :return: a list with tuples for each edge
        """
        output_list = []
        f = open(input_file)
        for line in f.readlines():
            case = line.split(", ")
            output_list.append((int(float((case[0]))), int(float((case[1])))))
        f.close()
        # assume no duplicates in input
        # output_list = self.remove_duplicates(output_list)
        # shrink data
        output_list = self.shrink_data(output_list, s)
        return output_list

    def shrink_data(self, output_list, s):
        """
        function the shrink the input file
        :param output_list: the list represent edges in input file
        :param s: shrink ratio
        :return: shrunk list
        """
        prob = s
        output = []
        for edge in output_list:
            if random.uniform(0,1) <= prob:
                output.append(edge)
        return output


    def remove_duplicates(self, output_list):
        """
        function to do preprocessing step before coin tossing, used to eliminate
        duplication and equivalent edges
        :param output_list: the list of edges before elimination
        :return: the list of edges after elimination
        """
        dic = {}
        garbage = []
        for item in output_list:
            if (item[0], item[1]) not in dic.keys()\
                    and (item[0], item[1]) not in dic.values():
                dic[(item[0], item[1])] = (item[1], item[0])
            else:
                garbage.append((item[0], item[1]))
        output = []
        for item in dic:
            output.append(item)
        return output



def k_range(k):
    """
    function to set range of k
    :param k: k number
    :return: a list
    """
    return [k, k]

def runExp(input_file=None, draw=0, sample_input=None, nodes=None, edges=None,degree=None, shrink=1.0, threshold=float('inf'), algorithm='yun'):
    """
    the function to run single experiment with all defined probability range
    :param input_file: input file
    :param draw: decide whether draw & save the figure
    :param sample_input: sample type
    :param nodes: sample nodes for random graph
    :param edges: sample edges for random graph
    :param degree: sample degree for each node in random graph
    :param shrink: shrink ratio
    :param threshold: threshold to stop execution of the algorithm when max clique value is over it
    :return:None
    """
    # prob_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    prob_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    # prob_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # prob_list = [0.9, 1.0]
    # prob_list = [1.0]

    problem_collection = []
    time_list = []
    for prob in prob_list:
        start = datetime.datetime.now()
        graph = GenerateGraph(input_file, prob, sample_input, nodes, edges, degree, shrink)
        print("Finish Generating Graph at probability={p}...".format(p=prob))
        if algorithm == "yun":
            problem = KCliques_Yun(graph,threshold)
        if algorithm == "bron":
            problem = KCliques_Bron(graph,threshold)
        print("Finish Generating Cliques at probability={p}...".format(p=prob))
        end = datetime.datetime.now()
        time_list.append(round((end - start).total_seconds() * 1000, 3))
        problem_collection.append(problem)


    # node number for each probability:
    nodes_list = []
    # edges number for each probability:
    edges_list = []
    # density for each probability:
    density_list = []
    cluster_co_list = []
    trans_list = []
    maximum_clique_list =[]
    cliques_with_prob_list = []
    num_cliques_by_prob = []

    for p in range(len(problem_collection)):
        clique_collection = []
        num_cliques = []
        prob_cliq_dic={}
        prob = prob_list[p]
        problem = problem_collection[p]
        print("=============probability={prob}=====================".format(prob=(p+1.0)/10))
        nodes_list.append(nx.number_of_nodes(problem.G))
        edges_list.append(nx.number_of_edges(problem.G))
        density_list.append(round(nx.density(problem.G),3))
        cluster_co_list.append(round(nx.average_clustering(problem.G),3))
        trans_list.append(round(nx.transitivity(problem.G),3))
        maximum_clique_list.append(problem.maximum_clique)

        for k in range(3, problem.maximum_clique + 1):
            problem.clique_counting(k_range(k))
            num_cliques.append(problem.num_kcliques)
            # print("cliques at {k_num} with length ".format(k_num=k) + str(len(problem.res_list)) + ": ")
            # print(str(problem.res_list))
            clique_collection.append(problem.res_list)
        prob_cliq_dic[prob] = clique_collection
        print("=============probability={prob} Done=====================".format(prob=(p+1.0)/10))

        cliques_with_prob_list.append(prob_cliq_dic)
        if draw != 0:
            draw_figures(problem, clique_collection, prob)

        num_cliques_by_prob.append(num_cliques)

    print("=====Results listed with prob: {prob_li}=====".format(prob_li=prob_list))
    print("Node Number: " + str(nodes_list))
    print("Edge Number: " + str(edges_list))
    print("Graph Density: " + str(density_list))
    print("Cluster Coefficient: " + str(cluster_co_list))
    print("Transitivity: " + str(trans_list))
    print("Maximum Clique Number: " + str(maximum_clique_list))
    print("Time Cost(Milliseconds): " + str(time_list))

    speedup_list = []
    for i in time_list:
        sp = round(time_list[-1]/i, 3)
        speedup_list.append(sp)
    print("Speed Up(times): " + str(speedup_list))

    accuracy_list = []
    print("Number Of Cliques Based On k From 3 To max:")
    for i in range(len(num_cliques_by_prob)):
        accu_li = []
        prob = prob_list[i]
        for j in range(len(num_cliques_by_prob[-1])):
            if j > len(num_cliques_by_prob[i]) - 1:
                num_cliques_by_prob[i].append(0)
            accu = round(1 - abs(num_cliques_by_prob[i][j] - num_cliques_by_prob[-1][j])/ num_cliques_by_prob[-1][j], 3)
            accu_li.append(accu)
        accuracy_list.append(accu_li)
        print("# in prob={p}: ".format(p=prob) + str(num_cliques_by_prob[i]) +
        "   Accuracy in prob={p}: ".format(p=prob) + str(accu_li))

    res = Result(time_list, nodes_list, edges_list, density_list, cluster_co_list,
                 trans_list, maximum_clique_list, num_cliques_by_prob, prob_list,time_list)

    return res



def draw_figures(problem, clique_collection, probability):
    """
    function to draw figure
    :param problem: input problem
    :param clique_collection: all collection lists of cliques
    :param probability: input probability
    :return: show and save the out figure from origin to k-cliques graph with probability
    """

    pos = nx.spring_layout(problem.G)
    pl.figure(1)
    pl.title("original figure with probability {p}".format(p=probability))
    nx.draw(problem.G, pos=pos)
    nx.draw_networkx_labels(problem.G, pos, font_size=10, font_family='sans-serif')
    pl.savefig("origin_with_prob_{p}.png".format(p=probability))
    # pl.show()

    for i in range(len(clique_collection)):
        new_nodes = []
        for n in problem.G.nodes():
            for li in clique_collection[i]:
                if n in li:
                    new_nodes.append(n)

        subgraph = problem.G.subgraph(new_nodes)
        pos = nx.spring_layout(problem.G)
        pl.figure()
        pl.title('{k} cliques with probability {p}'.format(k=i+3, p=probability))
        nx.draw(subgraph, pos=pos)
        nx.draw_networkx_labels(subgraph, pos, font_size=10, font_family='sans-serif')
        pl.savefig("{k}_cliq_with_prob_{p}.png".format(k=i+3, p=probability))
        # pl.show()

def readCommand(argv):
    """
    function to send option into commands
    :param argv: system args from command line
    :return: parsed args
    """
    from optparse import OptionParser
    usageStr = """
    USAGE:      python3.4 exp.py <options>
    EXAMPLES:   (1) python exp.py -f <file>
                  - starts an experiment on a file
    """
    parser = OptionParser(usageStr)
    parser.add_option('-f', '--file', dest='input_file', type='string')
    parser.add_option('-d', '--draw', dest='draw', type='int', default=0)
    parser.add_option('-s', '--sample', dest='sample_input', type='string')
    parser.add_option('-n', '--nodes', dest='nodes', type='int')
    parser.add_option('-m', '--edges', dest='edges', type='int')
    parser.add_option('-g', '--degree', dest='degree', type='float')
    parser.add_option('-r', '--shrink', dest='shrink', type='float', default=1.0)
    parser.add_option('-t', '--times', dest='times', type='int', default=1)
    parser.add_option('-e','--threshold', dest='threshold',type='float',default=float('inf'))
    parser.add_option('-a', '--algorithm', dest='algorithm', type='string', default='bron')
    options, otherjunk = parser.parse_args(argv)
    if len(otherjunk) != 0:
        raise Exception('Command line input not understood: ' + str(otherjunk))

    args = {}
    args['input_file'] = options.input_file
    args['draw'] = options.draw
    args['sample_input'] = options.sample_input
    args['nodes'] = options.nodes
    args['edges'] = options.edges
    args['degree'] = options.degree
    args['shrink'] = options.shrink
    args['times'] = options.times
    args['threshold'] = options.threshold
    args['algorithm'] = options.algorithm
    return args

def runExps(input_file=None, draw=0, sample_input=None, nodes=None,
            edges=None,degree=None, shrink=1.0, times=1, threshold=float('inf'), algorithm='yun'):
    """
    function to run many experiments
    :param input_file: input file
    :param draw: whether draw
    :param sample_input: sample input
    :param nodes: nodes in random graph
    :param edges: edges in random graph
    :param degree: degree in random graph
    :param shrink: shrink ratio in graph
    :param times: how many repeated experiments for same setting
    :param threshold: threshold to limit maximum clique number; used to experiment to
            decide times
    :return: result object with all statistical details
    """
    run_times = times
    res_list = []
    for t in range(run_times):
        res = runExp(input_file, draw, sample_input, nodes, edges, degree, shrink,threshold, algorithm)
        res_list.append(res)

    print("=====statistical results=======")
    n = len(res_list[0].time_list)
    m = len(res_list[0].num_cliques_by_prob[-1])
    time_list = [0] * n
    nodes_list = [0] * n
    edges_list = [0] * n
    density_list = [0] * n
    cluster_co_list = [0] *n
    trans_list = [0] * n
    maximum_clique_list =[0] * n
    cliques = []
    for i in range(n):
        cliques.append([0] * m)


    for res in res_list:
        time_list = list(map(add, time_list, res.time_list))
        nodes_list = list(map(add, nodes_list, res.nodes_list))
        edges_list = list(map(add, edges_list, res.edges_list))
        density_list = list(map(add, density_list, res.density_list))
        cluster_co_list = list(map(add, cluster_co_list, res.cluster_co_list))
        trans_list = list(map(add, trans_list, res.trans_list))
        maximum_clique_list = list(map(add, maximum_clique_list, res.maximum_clique_list))

        for i in range(len(res.num_cliques_by_prob)):
            cliques[i] = list(map(add, cliques[i], res.num_cliques_by_prob[i]))

    total_time_list = copy.deepcopy(time_list)

    total_time_list = [round(x,3) for x in total_time_list]
    time_list = [round(x / run_times, 3) for x in time_list]
    nodes_list = [round(x / run_times, 3) for x in nodes_list]
    edges_list = [round(x / run_times, 3) for x in edges_list]
    density_list = [round(x / run_times, 3) for x in density_list]
    cluster_co_list = [round(x / run_times, 3) for x in cluster_co_list]
    trans_list = [round(x / run_times, 3) for x in trans_list]
    maximum_clique_list = [round(x / run_times, 3) for x in maximum_clique_list]

    for i in range(len(cliques)):
        cliques[i] = [round(x / run_times, 3) for x in cliques[i]]

    print("=====Statistical Results listed with prob: {prob_li}=====".format(prob_li=res_list[0].prob_list))
    print("Average Node Number: " + str(nodes_list))
    print("Average Edge Number: " + str(edges_list))
    print("Average Graph Density: " + str(density_list))
    print("Average Cluster Coefficient: " + str(cluster_co_list))
    print("Average Transitivity: " + str(trans_list))
    print("Average Maximum Clique Number: " + str(maximum_clique_list))
    print("Average Time Cost(Milliseconds): " + str(time_list))
    speedup_list = []
    for i in time_list:
        sp = round(time_list[-1]/i, 3)
        speedup_list.append(sp)
    print("Average Speed up(times): " + str(speedup_list))
    print("Total Time cost(Milliseconds): " + str(total_time_list))

    accuracy_list = []
    print("Average Number Of Cliques Based On k From 3 To Max:")
    for i in range(len(cliques)):
        accu_li = []
        prob = res_list[0].prob_list[i]
        for j in range(len(cliques[-1])):
            accu = round(1 - abs(cliques[i][j] - cliques[-1][j])/cliques[-1][j], 3)
            accu_li.append(accu)
        accuracy_list.append(accu_li)
        print("# in prob={p}: ".format(p=prob) + str(cliques[i]) +
        "   Accuracy in prob={p}: ".format(p=prob) + str(accu_li))

    print("arrange accuracy by cliques")

    accuracy_rearrange = []
    for i in range(len(accuracy_list[0])):
        temp_accu = []
        for row in accuracy_list:
            temp_accu.append(row[i])
        accuracy_rearrange.append(temp_accu)
    print(accuracy_rearrange)


    labels = []

    pl.figure()
    pl.title("Accuracy in round {r}".format(r=times))
    for i in range(len(accuracy_rearrange)):
        pl.plot(res_list[0].prob_list, accuracy_rearrange[i])
        labels.append("{k}cliques".format(k=i+3))
    pl.legend(labels, ncol=3, loc='lower left')
    pl.ylim([-2.5, 1.2])

    # pl.savefig("accuracy_round{r}.png".format(r=times))
    # pl.show()

    labels = []

    pl.figure()
    pl.title("Accuracy in round {r}".format(r=times))
    for i in range(len(accuracy_list)):
        pl.plot([i for i in range(3, int(float(maximum_clique_list[-1] + 1)))], accuracy_list[i])
        labels.append("prob={k}".format(k=(i+1)/10))
    pl.legend(labels, ncol=3, loc='lower left')
    pl.ylim([-2.5, 1.2])

    # pl.savefig("accuracy_round{r}.png".format(r=times))
    # pl.show()


    result = Result(time_list, nodes_list, edges_list, density_list,
                cluster_co_list,trans_list, maximum_clique_list, [],
                res_list[0].prob_list, speedup_list, accuracy_list, total_time_list)

    return result

if __name__ == "__main__":
    args = readCommand(sys.argv[1:])
    result = runExps(**args)


