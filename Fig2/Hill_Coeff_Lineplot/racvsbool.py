import numpy as np
import matplotlib.pyplot as plt
from math import pow
import os
from scipy.stats import norm, zscore
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from scipy.spatial.distance import jensenshannon
from matplotlib import rcParams
import sys
import initialise.initialise as initialise
import initialise.parser as parser

version='cont'

in_file = 'init.txt'
begin=1
process_count=1
params = initialise.initialise(in_file)
initialise.create_folders(params)
id_to_node=[]

final_file_name = sys.argv

for j in params['input_filenames']:
        random_seed = int(begin) + process_count
        link_matrix, id_to_node = parser.parse_topo(params,j,random_seed)

network_name =  params['input_filenames'][0] # name_solution.dat and name_ssprob_all.txt files
plot_plotterdata = 1 # if boolean plot values are included or not

left= params['constant_node_count'][0]
right=len(id_to_node)-1

#give column range of genes you want to classify, 0 indexed
length=right-left+1
binlabelformat = "{0:0" + str(length) +  "b}"
rac_probfile=open("Datafiles/{}_racipe_probfull_{}_{}.txt".format(network_name, final_file_name[1], final_file_name[2]))

rac_probdata = rac_probfile.read().split("\n")[0:]

if "" in rac_probdata:
    rac_probdata.remove("")
rac_probfile.close()

xaxislabel= []
final_xaxislabel=[]
yaxis = []
final_yaxis =[]
xaxis=[]
final_xaxis=[]
final_index=[]
racipeclassify = {}

for k in rac_probdata:
    temp = k.split(" ")
    xaxislabel.append(temp[0])
    yaxis.append(float(temp[1]))
    if(float(temp[1]) >=0.01):
        final_xaxislabel.append(temp[0])
        final_index.append(int(temp[0],2))

    racipeclassify[ int(temp[0],2) ]  = float(temp[1])


final_yaxis = [racipeclassify[i] for i in final_index]
final_xaxislabel = [binlabelformat.format(i) for i in final_index]
argarr = np.argsort(np.array(final_yaxis))[::-1]

final_yaxis = [final_yaxis[i] for i in argarr]
final_xaxislabel = [final_xaxislabel[i] for i in argarr]
final_xaxis = np.linspace(0, len(final_yaxis), len(final_yaxis))
plt.figure(figsize=(20,12))

if(not(plot_plotterdata)):
    rcParams.update({'figure.autolayout':True})
    plt.bar(final_xaxis,final_yaxis, width = 0.3, color = 'r')
    plt.xticks(final_xaxis, final_xaxislabel, rotation = 'vertical')
    plt.savefig("{}_racipeVbool.png".format(network_name))



if plot_plotterdata:
    dataplotter = open("Datafiles/{}_racipe_probfull_wt_0.txt".format(network_name),'r').read().split("\n")[1:-1]
    x_labels = []
    y_data = []
    for temp in dataplotter:
        i = temp.split(" ")
        x_labels.append(i[0])
        y_data.append(float(i[1]))
    final_yaxislist=list(final_yaxis)
    list_labels=[]
    dict_yaxis={}
    for i in range(len(final_xaxis)):
        list_labels.append(final_xaxislabel[i])
        dict_yaxis.update( {final_xaxislabel[i] : i })
    cnt=0
    for i in range(len(y_data)):
        if (y_data[i] >= 0.01 and (str(x_labels[i]) not in final_xaxislabel) ):
            list_labels.append(x_labels[i])
            dict_yaxis.update( {x_labels[i] : len(final_xaxis)+cnt} )
            final_yaxislist.append(0)
            cnt+=1

    final_ydata = np.zeros(len(list_labels))
    for i in range(len(y_data)):
        if(x_labels[i] in dict_yaxis):
            index=dict_yaxis[x_labels[i]]

            final_ydata[index]=y_data[i]


    weight = 0.35
    final_xaxis=[i for i in range(len(list_labels)) ]
    rcParams.update({'figure.autolayout':True})
    final_yaxislist=np.array(final_yaxislist)
    rac_sum=0
    for q in range(len( final_yaxislist )):
        rac_sum+=final_yaxislist[q]

    final_yaxislist=  final_yaxislist / rac_sum

    bool_sum=0
    for q in range(len( final_ydata )):
        bool_sum+=final_ydata[q]

    final_ydata=  final_ydata / bool_sum
    fig, ax = plt.subplots(1,1)
    plt.bar(final_xaxis,np.array(final_yaxislist) , width = 0.3, color = 'r')
    plt.bar(np.array(final_xaxis)+weight, final_ydata, width = 0.3, color = 'b')
    plt.xticks(final_xaxis,list_labels, rotation = 'vertical')
    jsd=jensenshannon(final_ydata,np.array(final_yaxislist),2 )
    plt.title( "JSD = {}".format(jsd))
    plt.tight_layout()
    plt.savefig("Boolvsracipegraphs/{}_{}_{}_racipeV{}.png".format(network_name,final_file_name[1], final_file_name[2],version))
    jsdfile=open("JSD/{}_{}_jsd.txt".format(network_name, final_file_name[1]),"a")
    jsdfile.write("{:.6f}\n".format(jsd))
    jsdfile.close()