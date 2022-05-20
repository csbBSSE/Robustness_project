import os
import time
from os import listdir
from os.path import isfile, join
import random
topofiles= [os.path.splitext(f)[0] for f in listdir("topofiles/") if isfile(join("topofiles/", f))]

import numpy as np

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
link_matrix = [0]*len(topofiles)
copy_linkmatrix = [0]*len(topofiles)
id_to_node = [0]*len(topofiles)
length = len(topofiles)
nodes = [0]*len(topofiles)
for i in params['file_reqs']:
        for j in range (len(topofiles)):
                repj=topofiles[j]
                random_seed = int(begin) + process_count
                weighted_tick = 1 if "_weigh" in i else 0
                async_tick = 1 if "_async" in i else 0
                print(topofiles[j])
                link_matrix[j], id_to_node[j] = parser.parse_topo(params,repj,weighted_tick, random_seed)
                copy_linkmatrix[j], id_to_node[j] = parser.parse_topo(params,repj,weighted_tick, random_seed)
                nodes[j] = len(id_to_node[j])
 #####




num_simulations = 10000
num_threads = 5

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
    curnetwork = open("curnetwork.txt", 'w')
    looptime = time.time()
    initfile = open("init.txt", "w")
    #print( inittext.format(topofiles[i], num_threads, num_simulations[i]*nodes[i]**2)  )
    initfile.write(inittext.format(topofiles[i], num_threads, num_simulations * nodes[i]))
    initfile.close()

    print("Bool_{}".format(topofiles[i]) )
    os.system("./main")
   
    os.system("python3 plotternew.py")
    
    
    curnetwork.write(topofiles[i])
    curnetwork.close()
    
    open("JSD/{}_jsd.txt".format(topofiles[i]), 'w')
    
    print("classify")
    os.system("python3 classifybool.py")
    
        
    
    
    cnt = 1
    #print(copy_linkmatrix[i][0][0])
    
    
    tmpcount=0
    for j in range(nodes[i]):
        for k in range(nodes[i]):
            if(copy_linkmatrix[i][j][k] !=0):
                tmpcount+=1

    
    arr= random.sample(range(0, tmpcount ),10)
    arr=np.array(arr)
    arr=np.sort(arr)
    
    outdegarr=[0]*nodes[i]
    indegarr=[0]*nodes[i]
    
    
    for j in range(nodes[i]):
        for k in range(nodes[i]):
            if(copy_linkmatrix[i][j][k] !=0):
                outdegarr[j]+=1
                
            if(copy_linkmatrix[i][k][j] !=0):
                indegarr[j]+=1
             
            
            
            
    tmpcount=0
    tmpindex=0
    #print(arr)
    #print(indegarr)
    #print(outdegarr)
    print(link_matrix[i])
    print(arr)
    for j in range(nodes[i]):
        if(tmpindex==10):
                break
        for k in range(nodes[i]):
            f=0
            if(tmpindex==10):
                break
            if(copy_linkmatrix[i][j][k] !=0):
                print(j,k,tmpindex)
                if(tmpcount==arr[tmpindex]):
                    f=1
                    tmpindex+=1
                
                tmpcount+=1
            if(f!=1):
             continue
           
           
           
          
           
           
           


            val = randint(-1 , 1)
            #print(val)
            
            
            
            if(  indegarr[j]+outdegarr[j] == 1 or indegarr[k]+outdegarr[k] ==1):
                val=  (-1) *copy_linkmatrix[i][j][k] 
            
            
            
            for l in range(3):
                if (l-1 == copy_linkmatrix[i][j][k] or copy_linkmatrix[i][j][k]==0 ):
                    continue
                    
                else:
                    if( (l-1) != val):
                        continue

                    link_matrix[i][j][k] = l - 1
                    tempfile = "{}_{}_{}_{}".format(topofiles[i], j, k, l)
                    initfile = open("init.txt", "w")
                    #print( inittext.format(topofiles[i], num_threads, num_simulations[i]*nodes[i]**2)  )
                    initfile.write(inittext.format(tempfile, num_threads,nodes[i]*num_simulations))
                    initfile.close()
                    temptopofile = open(tempfile + ".topo", 'w')
                    
                    tempidsfile = open(tempfile + ".ids", 'w')
                    tempidsfile.write(open(topofiles[i]+".ids", 'r').read())
                    tempidsfile.close()

                    strtemp = "Source Target Type\n"
                    #print(nodes[i])
                    #print(id_to_node[i])
                    for p1 in range(nodes[i]):
                        for p2 in range(nodes[i]):
                            #print(p1,p2)
                            if link_matrix[i][p1][p2] == 0:
                                continue
                            else:
                                strtemp += id_to_node[i][p1] + " " + id_to_node[i][p2] + " " + str(1 if link_matrix[i][p1][p2] == 1 else 2) + "\n"
                    
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
                    
                    temptopofile = open("input/{}".format(tempfile + ".topo") , 'w')
                    temptopofile.write(open(tempfile+".topo", 'r').read())
                    temptopofile.close()
                    
                    
                    tempidsfile = open("input/{}".format(tempfile + ".ids") , 'w')
                    tempidsfile.write(open(tempfile+".ids", 'r').read())
                    tempidsfile.close()
                    
                    
                    print("bool_{}".format(tempfile) )
                    os.system("./main")
                    os.system("python3 plotternew.py")
                    print("classifier")
                    
                    os.system("python3 classifybool.py")
                    
                    link_matrix[i][j][k] = copy_linkmatrix[i][j][k]
                    print("Run {}/{}: time = {:.2f}, totaltime = {:.2f}".format(i+1, len(topofiles), time.time() - looptime, time.time() - tottime))
    #print("activation")
    #os.system("python3 activationplotter.py") #same issue as above
