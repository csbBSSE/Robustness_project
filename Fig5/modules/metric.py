import os.path
import networkx as nx
import numpy as np

def networkx_graph(network_name):
    topofile_data = open(os.path.dirname(__file__) + "/../input/{}.topo".format(network_name), 'r').read().split("\n")[1:]

    empty_break = 1
    if "" in topofile_data:
        topofile_data.remove("")
    else:
        empty_break = 0

    graph = nx.DiGraph()
    for i in range(len(topofile_data)):
        topofile_data[i] = topofile_data[i].split(" ")
        topofile_data[i][2] = -1 if int(topofile_data[i][2]) == 2 else 1

    for i in topofile_data:
        graph.add_edge(i[0], i[1], weight = i[2])

    return graph

def cycle_info(network_name, graph):

    cycles = list(nx.simple_cycles(graph))
    weight = []
    sign = []
    num_pos_cycles = 0
    num_neg_cycles = 0
    weigh_num_pos = 0
    weigh_num_neg = 0

    for i in cycles:
        if len(i) == 1:
            sign.append(graph[i[0]][i[0]]['weight'])
        else:
            sign_temp = 1
            for j in range(len(i)):
                truej1 = (j+1) % len(i)
                sign_temp *= graph[i[j]][i[truej1]]['weight']
            sign.append(sign_temp)
        weight.append(1/len(i))

    for i in range(len(sign)):
        if sign[i] == 1:
            num_pos_cycles += 1
            weigh_num_pos += weight[i]
        else:
            num_neg_cycles += 1
            weigh_num_neg += weight[i]

    total_cycles = len(sign)
    total_weight = weigh_num_pos + weigh_num_neg
    
    return np.array([num_pos_cycles, num_neg_cycles, num_pos_cycles/max(total_cycles,1), (weigh_num_pos ,  weigh_num_neg) ])

def matrix(network_name):
    network = open(os.path.dirname(__file__) + "/../input/{}.topo".format(network_name)).read().split("\n")[1:]
    ids = open(os.path.dirname(__file__) + "/../input/{}.ids".format(network_name)).read().split("\n")[1:]

    empty_break = 1
    if "" in network:
        network.remove("")
    else:
        empty_break = 0

    empty_break = 1
    if "" in ids:
        ids.remove("")
    else:
        empty_break = 0

    nodes_dict = {}
    for i in ids:
        temp = i.split(" ")
        nodes_dict[temp[0]] = int(temp[1])

    nodes = len(ids)
    link_matrix = np.zeros((nodes, nodes))

    for i in network:
        temp = i.split(" ")
        source = nodes_dict[temp[0]]
        target = nodes_dict[temp[1]]

        link_matrix[source][target] = -1 if temp[2] == '2' else 1

    return link_matrix
  
if __name__ == '__main__':
    import sys
    sys.path.append('../')
    import initialise.initialise as initialise
    import initialise.parser as parser

    network_name = "GRHL2"
    graph = networkx_graph(network_name)
    cycle_stuff = cycle_info(network_name, graph)
    link_matrix = matrix(network_name)

