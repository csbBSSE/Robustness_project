import os
import time
from os import listdir
from os.path import isfile, join
topofiles= [os.path.splitext(f)[0] for f in listdir("topofiles/") if isfile(join("topofiles/", f))]
from multiprocessing import Pool, freeze_support
from time import sleep
import numpy as np
import sys
import initialise.initialise as initialise
import initialise.parser as parser
import modules.metric as metric
from scipy.spatial.distance import jensenshannon
from matplotlib.pyplot import plt

def perturb_gen(x):
    global num_threads
    global num_simulations

    inittext = """input_folder_name input
    output_folder_name output
    input_filenames {}
    num_threads {}
    num_runs 1
    num_simulations {}
    maxtime 2000
    asynchronous_run 1
    synchronous_run 0
    weighted_run 0
    unweighted_run 1
    selective_edge_weights 0
    randomise_edges_file randomise.txt
    constant_node_count 0
    """

    topofile_list = []

    in_file = 'init_{}.txt'.format(x)
    max_initlines = 14
    begin=1
    process_count=1
    params = initialise.initialise(in_file, max_initlines)
    params['file_reqs'] = initialise.set_file_reqs(params)
    id_to_node=[]
    link_matrix = 0
    copy_linkmatrix = 0
    id_to_node = 0
    length = len(topofiles)
    nodes = 0
    for i in params['file_reqs']:
        repj=x
        random_seed = int(begin) + process_count
        weighted_tick = 1 if "_weigh" in i else 0
        async_tick = 1 if "_async" in i else 0
        print(x)
        graph_matrix, id_to_node = parser.parse_topo(params,repj,weighted_tick, random_seed)
        copy_graph_matrix = np.matrix(graph_matrix)
        nodes = len(id_to_node)

    outdegarr=[0]*nodes
    indegarr=[0]*nodes

    for j in range(nodes):
        for k in range(nodes):
            if(copy_graph_matrix[j][k] !=0):
                outdegarr[j]+=1

            if(copy_graph_matrix[k][j] !=0):
                indegarr[j]+=1
    cnt = 1

    #print(copy_graph_matrix[0][0])
    for j in range(nodes):
        for k in range(nodes):
            for l in range(3):
                if l-1 == graph_matrix[j][k]:
                    continue
                else:
                    if(graph_matrix[j][k]!=0 and (indegarr[j]+outdegarr[j] ==1 or  indegarr[k]+outdegarr[k] ==1)):
                        if(l-1 == 0):
                            continue
                    copy_graph_matrix[j][k] = l - 1
                    tempfile = "{}_{}_{}_{}".format(x, j, k, l)
                    topofile_list.append(tempfile)
                    initfile = open(initfilename, "w")
                    #print( inittext.format(x, num_threads, num_simulations[i]*nodes**2)  )
                    initfile.write(inittext.format(tempfile, num_threads,nodes*num_simulations))
                    initfile.close()
                    temptopofile = open("input/" + tempfile + ".topo", 'w')

                    tempidsfile = open("input/" + tempfile + ".ids", 'w')
                    tempidsfile.write(open("input/" + x+".ids", 'r').read())
                    tempidsfile.close()

                    strtemp = "Source Target Type\n"
                    #print(nodes)
                    #print(id_to_node[i])
                    for p1 in range(nodes):
                        for p2 in range(nodes):
                            #print(p1,p2)
                            if graph_matrix[p1][p2] == 0:
                                continue
                            else:
                                strtemp += id_to_node[i][p1] + " " + id_to_node[i][p2] + " " + str(1 if graph_matrix[p1][p2] == 1 else 2) + "\n"

                    strtemp = strtemp[:-1]
                    #strtemp = strtemp.split("\n")[:-1]
                    #strfinal = ""
                    #for p in range(len(strtemp)):
                    #    if p!= len(strtemp)-1:
                    #        strfinal += strtemp[p] + "\n"
                    #    else:
                    #        strfinal += strtemp[p]
                    temptopofile.write(strtemp)
                    temptopofile.close()

                    # print("bool_{}".format(tempfile) )
                    # os.system("./main {}".format(initfilename))
                    # os.system("python3 plotternew.py {}".format(initfilename))
                    # print("classifier")

                    # os.system("python3 classifybool.py {} {}".format(initfilename, x))
                    #
                    # os.remove("output/{}_async_unweigh_fss_run1.txt".format(tempfile))
                    # os.remove("output/{}_async_unweigh_nss_run1.txt".format(tempfile))
                    # os.remove("output/{}_async_unweigh_ss_run1.txt".format(tempfile))
                    # os.remove("output/{}_async_unweigh_init_run1.txt".format(tempfile))

                    copy_graph_matrix[j][k] = graph_matrix[j][k]
    return topofile_list
    # print("Run {}/{}: time = {:.2f}, totaltime = {:.2f}".format(i+1, len(topofiles), time.time() - looptime, time.time() - tottime))

def runner_code(x, original_name):
    global num_threads
    global num_simulations

    looptime = time.time()
    initfilename = "init_{}.txt".format(x)
    initfile = open(initfilename, "w")
    #print( inittext.format(x, num_threads, num_simulations[i]*nodes[i]**2)  )
    initfile.write(inittext.format(x, num_threads, num_simulations * nodes[i]))
    initfile.close()

    print("Bool_{}".format(x))
    os.system("./main {}".format(initfilename))

    os.system("python3 plotternew.py {}".format(initfilename))
    if x == original_name:
        open("JSD/{}_jsd.txt".format(original_name), 'w')

    print("classify")
    os.system("python3 classifybool.py {} {}".format(initfilename, x))

    os.remove("output/{}_async_unweigh_fss_run1.txt".format(x))
    os.remove("output/{}_async_unweigh_ss_run1.txt".format(x))
    os.remove("output/{}_async_unweigh_nss_run1.txt".format(x))
    os.remove("output/{}_async_unweigh_init_run1.txt".format(x))


def robustness_calc(x):
    global num_threads

    #runs base topofile
    runner_code(x,x)

    #runs perturbations using pool
    topofile_list = perturb_gen(x)

    ####POOL FOR TOPOFILE LIST
    n = len(topofile_list)
    argarr = []
    for i in range(n):
        argarr.append([topofile_list[i], x])


    pool = Pool(num_threads)
    pool.starmap(runner_code, argarr)


    #finds average jsd
    tot_jsd = 0
    jsdfile = np.loadtxt("JSD/{}_jsd.txt".format(x))
    m,n = jsdfile.shape
    for i in range(1,m):
        jsd = jensenshannon(jsdfile[0],jsdfile[i],2)
        tot_jsd += jsdfile
    tot_jsd /= m - 1
    return tot_jsd

def causal_perturb_up(x):
    topofile_list = perturb_gen(x)
    frac_pos_cycles = []
    for i in topofile_list:
        graph = metric.networkx_graph(i)
        frac = metric.cycle_info(i, graph)
        frac_pos_cycles.append([frac[3], i])
    frac_pos_cycles.sort()
    return frac_pos_cycles[-1][1]

def causal_perturb_dn(x):
    topofile_list = perturb_gen(x)
    frac_pos_cycles = []
    for i in topofile_list:
        graph = metric.networkx_graph(i)
        frac = metric.cycle_info(i, graph)
        frac_pos_cycles.append([frac[3], i])
    frac_pos_cycles.sort()
    return frac_pos_cycles[0][1]

def graph_generate(x):
    n = 5
    name = x
    orig_jsd = robustness_calc(x)
    jsdarr_up = [orig_jsd]
    jsdarr_dn = [orig_jsd]
    for i in range(n):
        name = causal_perturb_up(name)
        tot_jsd = robustness_calc(name)
        jsdarr_up.append(tot_jsd)
    name = x
    for i in range(n):
        name = causal_perturb_dn(name)
        tot_jsd = robustness_calc(name)
        jsdarr_dn.append(tot_jsd)

    plt.plot([i for i in range(n + 1)], jsdarr_up)
    plt.savefig("{}_jsd_up.png".format(x), transparent = True)
    plot.clf()

    plt.plot([i for i in range(n + 1)], jsdarr_dn)
    plt.savefig("{}_jsd_dn.png".format(x), transparent = True)
    plot.clf()



#topofiles = ["GRHL2","GRHL2wa","OCT4","OVOL2","OVOLsi","NRF2","Jia_1","Jia_2","Jia_CBS","Nfatc","silveira","TGFB","Wnt","tian","EMT_EACIPE","EMT_RACIPE2","dsrgn","asbpsa"]

#topofiles=["TGFB"]

#print(topofiles)

matplotlib.rcParams.update({'font.weight':'bold', 'xtick.color':'0.3', 'ytick.color':'0.3', 'axes.labelweight':'bold', 'axes.titleweight':'bold', 'figure.titleweight':'bold', 'text.color':'0.3', 'axes.labelcolor':'0.3', 'axes.titlecolor':'0.3', 'font.size': '25', 'axes.titlesize':'50', 'axes.labelsize':'45', 'xtick.labelsize':'33', 'ytick.labelsize':'30', 'legend.fontsize':'33'})

num_simulations = 10000
num_threads = 50

#tottime = time.time()

x = sys.argv[2]
graph_generate(x)
