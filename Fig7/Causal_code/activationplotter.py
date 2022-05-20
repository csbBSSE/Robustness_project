import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import math
import initialise.initialise as initialise
import initialise.parser as parser
print("modules imported")

in_file = 'init.txt'
max_initlines = 14
begin=1
process_count=1
params = initialise.initialise(in_file, max_initlines)
params['file_reqs'] = initialise.set_file_reqs(params)
for i in params['file_reqs']:
    for j in params['input_filenames']:
                random_seed = int(begin) + process_count
                weighted_tick = 1 if "_weigh" in i else 0
                async_tick = 1 if "_async" in i else 0
                link_matrix, id_to_node = parser.parse_topo(params,j,weighted_tick, random_seed)
                # print(params)
                # plot_bar(j+i,id_to_node,params)
                probfile = open("{}/{}_ssprob_all.txt".format(params['output_folder_name'], j+i), "r")
                probdata = probfile.read().split("\n")[1:]
                if "" in probdata:
                    probdata.remove("")
                probfile.close()

                activation_arr = []
                label_arr = []
                prob_arr = []
                filename_index = 0
                act_file = open("{}_activation.txt".format(j+i), "w")
                act_file.write("State ActivationSum Probability\n")
                for p in range(len(params['input_filenames'])):
                    if j == params['input_filenames'][p]:
                        filename_index = p
                        break
                cnt=0        
                for k in probdata:
                    temp = k.split(" ")
                    statevec = [0]*params['constant_node_count'][filename_index]
                    for p in temp[0]:
                        statevec.append(int(p))
                    # print(statevec)
                    for p in range(len(statevec)):
                        statevec[p] = -1 if statevec[p] == 0 else 1
                    statevec = np.array(statevec)
                    
                    deg=[0]*len(statevec)
                    deg=np.array(deg)
                    for y in range(len(statevec)):
                        for h in range(len(statevec)):
                            deg[y]+=abs( link_matrix[h][y])
                    # print(len(statevec), statevec)
                    activationarr= np.matmul(statevec, link_matrix)
                    statevec=np.array(statevec)
                    activationarr=np.array(activationarr)
                    activationconv=[0.0]*len(statevec)
                    activationconv=np.array(activationconv)
                    #activationconv=activationconv.astype(int)
                    #activationconv=[0]*len(statevec)
                    sum1=0
                    
                    
                    invdeg=np.array(deg)
                    for q in range(len(deg)):
                        if(deg[q]==0):
                            invdeg[q]=1000
                        else:    
                            invdeg[q]=1/deg[q]
                    
                    for t in range(len(statevec)):
                        activationconv[t]=statevec[t]*activationarr[t]
                        sum1+=activationconv[t]
                        
                    neg=1
                    
                    
                    
                    for q in range(len(statevec)):
                        if(activationconv[q]<=0 and deg[q]!=0):
                            neg+= (-activationconv[q]+1)/deg[q]    
                    sum1=sum1/neg        
                    activation_sum = np.dot(statevec, activationarr)/neg    
                    
                    
                    #if(neg>1):     
                    #    activation_sum = 0
                    
                    
                            
                    #print(activation_sum,sum1)
                    #print(statevec)
                    #print(np.matmul(statevec, link_matrix))
                    #print(activation_sum)                 
                    #print("_____________________")
                    # print(statevec, np.matmul(statevec, link_matrix), activation_sum)
                    
                    
                    #activation_arr.append(activation_sum)
                    activation_arr.append(sum1)
                    
                    
                    label_arr.append(temp[0])
                    prob_arr.append(float(temp[1]))
                    
                    if(sum1>17):
                        print(label_arr[cnt])
                        print(statevec)
                        print(activationconv)
                        print(deg)
                        print("____________________")
                    statevec =statevec.astype(int)
                    activationarr=activationarr.astype(int)
                    activationconv=activationconv.astype(int)
                    deg=deg.astype(int)
                    
                    #act_file.write("{} {} {}\n".format(label_arr[cnt], activation_arr[cnt], prob_arr[cnt]))
                    act_file.write("{} {} {}\n".format(label_arr[cnt],sum1, prob_arr[cnt]))
                    
                    
                    act_file.write(np.array_str( statevec ) )
                    act_file.write("\n")
                    act_file.write(np.array_str( activationarr ) )
                    act_file.write("\n")
                    act_file.write(np.array_str( activationconv ) )
                    act_file.write("\n")
                    act_file.write(np.array_str( deg ) )
                    act_file.write("\n")
                    act_file.flush()
                    
                    cnt+=1
                #np.fromstring(ts,dtype=int) 
                
                #s1=statevec.tostring()
                #s2=activationarr.tostring()
                #s3=activationconv.tostring()
                
      
                    
                plt.figure(figsize = (20,10))
                plt.scatter(prob_arr, activation_arr)
                plt.savefig("{}_activationplot.png".format(j+i))
                plt.clf()
