import os
import time
from os import listdir
from os.path import isfile, join
import networkx as nx

topofiles= [os.path.splitext(f)[0] for f in listdir("topofiles/") if isfile(join("topofiles/", f))]

"abheepsa was weird"
"Anish - 2020"

#topofiles = ["GRHL2","GRHL2wa","OCT4","OVOL2","OVOLsi","NRF2","Jia_1","Jia_2","Jia_CBS","Nfatc","silveira","TGFB","Wnt","tian","EMT_EACIPE","EMT_RACIPE2","dsrgn","asbpsa"]

#topofiles=["TGFB"]
#
# print(topofiles)
# ####
# import initialise.initialise as initialise
# import initialise.parser as parser
# in_file = 'init.txt'
# max_initlines = 14
# begin=1
# process_count=1
# params = initialise.initialise(in_file, max_initlines)
# params['file_reqs'] = initialise.set_file_reqs(params)
# id_to_node=[]
#
# length=len(topofiles)
# nodes=[0]*len(topofiles)
# for i in params['file_reqs']:
#         for j in range (len(topofiles)):
#                 repj=topofiles[j]
#                 random_seed = int(begin) + process_count
#                 weighted_tick = 1 if "_weigh" in i else 0
#                 async_tick = 1 if "_async" in i else 0
#                 print(topofiles[j])
#                 link_matrix, id_to_node = parser.parse_topo(params,repj,weighted_tick, random_seed)
#                 nodes[j]=len(id_to_node)
#  #####
#
#
#
#
#
#
# num_simulations = 10000
# num_threads = 80
#
# inittext = """input_folder_name input
# output_folder_name output
# input_filenames {}
# num_threads {}
# num_runs 1
# num_simulations {}
# maxtime 2000
# asynchronous_run 1
# synchronous_run 0
# weighted_run 0
# unweighted_run 1
# selective_edge_weights 0
# randomise_edges_file randomise.txt
# constant_node_count 0
# """
#
# tottime = time.time()

final_data = open("networkCycles.txt", "w")
final_csv = open("networkCycles.csv", "w")
final_string = "name num_pos num_neg frac_num_neg total weight_pos weight_neg frac_weighted_num_neg total_weight\n"
final_string_csv = "name,num_pos,num_neg,frac_num_neg,total,weight_pos,weight_neg,frac_weighted_num_neg ,total_weight\n"

for topo in range(len(topofiles)):
    # looptime = time.time()
    # initfile = open("init.txt", "w")
    #print( inittext.format(topofiles[i], num_threads, num_simulations[i]*nodes[i]**2)  )
    # initfile.write(inittext.format(topofiles[topo], num_threads, num_simulations *nodes[topo]))
    # initfile.close()

    topofile_data = open("topofiles/{}.topo".format(topofiles[topo]), 'r').read().split("\n")[1:]

    empty_break = 1
    while empty_break:
        if "" in topofile_data:
            topofile_data.remove("")
        else:
            empty_break = 0
    print("hello", flush = "true")
    print(topofiles[topo], len(topofile_data), end = " ", flush = "true")

    graph = nx.DiGraph()
    for i in range(len(topofile_data)):
        # print(i)
        topofile_data[i] = topofile_data[i].split(" ")
        topofile_data[i][2] = -1 if int(topofile_data[i][2]) == 2 else 1

    for i in topofile_data:
        # print(i[0], i[1], i[2])
        graph.add_edge(i[0], i[1], weight = i[2])
    # print(graph.edges())

    cycles = list(nx.simple_cycles(graph))
    weight = []
    sign = []
    num_pos_cycles = 0
    num_neg_cycles = 0
    weigh_num_pos = 0
    weigh_num_neg = 0



    print(len(cycles), flush = "true")
    # print(cycles)
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
    # print(sign)
    total_cycles = len(sign)
    total_weight = weigh_num_pos + weigh_num_neg
    if topo == len(topofiles) - 1:
        final_string += "{} {} {:.2f} {} {} {:.2f} {:.2f} {:.2f} {:.2f}".format(topofiles[topo], num_pos_cycles, num_neg_cycles, num_neg_cycles/max(total_cycles,1), total_cycles, weigh_num_pos, weigh_num_neg,weigh_num_neg/max(total_weight,1), total_weight)
        final_string_csv += "{},{},{:.2f},{},{},{:.2f},{:.2f},{:.2f},{:.2f}".format(topofiles[topo], num_pos_cycles, num_neg_cycles, num_neg_cycles/max(total_cycles,1), total_cycles, weigh_num_pos, weigh_num_neg,weigh_num_neg/max(total_weight,1), total_weight)
    else:
        final_string += "{} {} {:.2f} {} {} {:.2f} {:.2f} {:.2f} {:.2f}\n".format(topofiles[topo], num_pos_cycles, num_neg_cycles, num_neg_cycles/max(total_cycles,1), total_cycles, weigh_num_pos, weigh_num_neg,weigh_num_neg/max(total_weight,1), total_weight)
        final_string_csv += "{},{},{:.2f},{},{},{:.2f},{:.2f},{:.2f},{:.2f}\n".format(topofiles[topo], num_pos_cycles, num_neg_cycles, num_neg_cycles/max(total_cycles,1), total_cycles, weigh_num_pos, weigh_num_neg,weigh_num_neg/max(total_weight,1), total_weight)

final_data.write(final_string)
final_csv.write(final_string_csv)
final_csv.close()
final_data.close()


    # print("RACIPE_{}".format(topofiles[i]) )
    # os.system("./RACIPE {}.topo {}.ids -num_paras {} -num_stability 4 -threads {}".format(topofiles[i],topofiles[i],num_simulations *nodes[i],num_threads  ))
    #
    # print("classifier")
    # os.system("python3 classifyracipe.py") #NOTE: just python for windows machines (if you don't have python2 installed that is)

    #print("activation")
    #os.system("python3 activationplotter.py") #same issue as above

    # print("Run {}/{}: time = {:.2f}, totaltime = {:.2f}".format(i+1, len(topofiles), time.time() - looptime, time.time() - tottime))
