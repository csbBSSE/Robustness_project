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
    
    return np.array([num_pos_cycles, num_neg_cycles, num_neg_cycles/max(total_cycles,1),  weigh_num_pos ,  weigh_num_neg ])

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

def activation_measure(curstate, link_matrix, graph):
    activation = np.matmul(curstate, link_matrix)
    sum_act = np.dot(curstate, activation)
    bad_nodes = []
    all_nodes = list(graph.nodes)
    for i in range(len(graph.nodes)):
        if activation[i] * curstate[i] < 0:
            bad_nodes.append(all_nodes[i])

    # for i in bad_nodes:
    #     print(graph.in_degree(i))
    a = []
    for i in bad_nodes:
        if graph.in_degree(i) == 0:
            a.append(10)
        else:
            a.append(1/graph.in_degree(i))
    bad_indegree_sum = sum(a)
    indegree_sum = sum([graph.in_degree(i) for i in graph.nodes])

    return (sum_act/(1 + bad_indegree_sum))/indegree_sum


def act_eig(network_name, graph, lm):
    lm_sym = (lm + lm.T) / 2

    val, vec = np.linalg.eig(lm_sym)
    n = len(val)

    arg = np.flip(np.argsort(val))

    val_sort = np.zeros(val.shape)
    vec_sort = np.zeros(vec.shape)

    for i in range(n):
        val_sort[i] = val[arg[i]]
        vec_sort[i] = vec[arg[i]]

    v = vec_sort[0]
    for i in range(n):
        v[i] = np.real(v[i])

    v_bin = np.zeros(n)
    
    for i in range(n):
        if(v[i] >0):
            v_bin[i] = 1
        elif(v[i]<0):
            v_bin[i] = -1
        else:
            v_bin[i] = 0

    for run in range(10):
        act = np.matmul(v_bin, lm)
       
        for i in range(n):
            v_bin[i] = -v_bin[i] if act[i]*v_bin[i] < 0 else v_bin[i]
            

    act_measure = activation_measure(v_bin, lm, graph)

    val_pos = [i for i in val_sort if i>0]
    eigsum = sum(val_pos)

    return [act_measure, val_sort[0], eigsum, val_sort[0]/np.sqrt(n), eigsum/(np.sqrt(n) * len(val_pos))]

