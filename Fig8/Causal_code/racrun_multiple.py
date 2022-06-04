import os
import time
from os import listdir
from os.path import isfile, join
topofiles= [os.path.splitext(f)[0] for f in listdir("topofiles/") if isfile(join("topofiles/", f))]



#topofiles = ["GRHL2","GRHL2wa","OCT4","OVOL2","OVOLsi","NRF2","Jia_1","Jia_2","Jia_CBS","Nfatc","silveira","TGFB","Wnt","tian","EMT_EACIPE","EMT_RACIPE2","dsrgn","asbpsa"]

#topofiles=["TGFB"]

print(topofiles)
####
import initialise.initialise as initialise
import initialise.parser as parser
in_file = 'init.txt'
max_initlines = 14
begin=1
process_count=1
params = initialise.initialise(in_file, max_initlines)
params['file_reqs'] = initialise.set_file_reqs(params)
id_to_node=[]

length=len(topofiles)
nodes=[0]*len(topofiles)
for i in params['file_reqs']:
        for j in range (len(topofiles)):
                repj=topofiles[j]
                random_seed = int(begin) + process_count
                weighted_tick = 1 if "_weigh" in i else 0
                async_tick = 1 if "_async" in i else 0
                print(topofiles[j])
                link_matrix, id_to_node = parser.parse_topo(params,repj,weighted_tick, random_seed)
                nodes[j]=len(id_to_node)
 #####          






num_simulations = 10000
num_threads = 80

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

tottime = time.time()


for i in range(len(topofiles)):
    looptime = time.time()
    initfile = open("init.txt", "w")
    #print( inittext.format(topofiles[i], num_threads, num_simulations[i]*nodes[i]**2)  )
    initfile.write(inittext.format(topofiles[i], num_threads, num_simulations *nodes[i]))
    initfile.close()
    
    print("RACIPE_{}".format(topofiles[i]) )
    os.system("./RACIPE {}.topo {}.ids -num_paras {} -num_stability 4 -threads {}".format(topofiles[i],topofiles[i],num_simulations *nodes[i],num_threads  ))

    print("classifier")
    os.system("python3 classifyracipe.py") #NOTE: just python for windows machines (if you don't have python2 installed that is)

    #print("activation")
    #os.system("python3 activationplotter.py") #same issue as above

    print("Run {}/{}: time = {:.2f}, totaltime = {:.2f}".format(i+1, len(topofiles), time.time() - looptime, time.time() - tottime))
