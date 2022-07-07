import numpy as np
import matplotlib.pyplot as plt
from math import pow
import os
from scipy.stats import norm, zscore
import statsmodels.api as sm
from scipy.spatial.distance import jensenshannon
from matplotlib import rcParams
import matplotlib.lines as mlines
#from scipy.stats import kde
# from scipy import stats

from os import listdir
from os.path import isfile, join

from matplotlib import rcParams
#matplotlib.use('ps')
#from matplotlib import rc

#rc('text',usetex=True)
#rc('text.latex', preamble='\usepackage{color}')
topofiles= os.listdir()

rcParams.update({'font.weight':'bold', 'xtick.color':'0.3', 'ytick.color':'0.3', 'axes.labelweight':'bold', 'axes.titleweight':'bold', 'figure.titleweight':'bold', 'text.color':'0.3', 'axes.labelcolor':'0.3', 'axes.titlecolor':'0.3', })
#print(topofiles)
version='cont'


if "maincurrent.c" in topofiles:
    version='ising'
else:
    version = 'cont'

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

for i in params['file_reqs']:
        for j in params['input_filenames']:
                random_seed = int(begin) + process_count
                weighted_tick = 1 if "_weigh" in i else 0
                async_tick = 1 if "_async" in i else 0
                link_matrix, id_to_node = parser.parse_topo(params,j,weighted_tick, random_seed)
 #####





network_name =  params['input_filenames'][0] # name_solution.dat and name_async_unweigh_ssprob_all.txt files
plot_plotterdata = 0 # if boolean plot values are included or not

left= params['constant_node_count'][0]
right=len(id_to_node)-1

#give column range of genes you want to classify, 0 indexed
# print(data)
# exit(0)
length=right-left+1



binlabelformat = "{0:0" + str(length) +  "b}"


rac_probfile=open("Datafiles_error/{}_racipe_probfull_error.txt".format(network_name))

rac_probdata = rac_probfile.read().split("\n")[0:]

if "" in rac_probdata:
    rac_probdata.remove("")
rac_probfile.close()

xaxislabel= []
final_xaxislabel=[]
yaxis = []
final_yaxis =[]
error = []
final_error = []
xaxis=[]
final_xaxis=[]
final_index=[]
racipeclassify = {}
dict_encounter={}

encount=0
probval=0
for k in rac_probdata:
    temp = k.split(" ")

    if(temp[0][left:] not in xaxislabel):
            xaxislabel.append(temp[0][left:])
            encount+=1
            dict_encounter.update( {temp[0][left:] : encount-1 })
            yaxis.append(float(temp[1]))
            error.append(float(temp[2]))

            probval=float(temp[1])
    else:
            yaxis[  dict_encounter[ temp[0][left:] ] ] += float(temp[1])
            error[dict_encounter[temp[0][left:]]] += float(temp[2])
            probval = yaxis[  dict_encounter[ temp[0][left:] ] ]


for k in range ( len(yaxis) ):

    if(yaxis[k] >=0.01):
        final_xaxislabel.append(xaxislabel[k])
        final_index.append(int(xaxislabel[k],2))

    racipeclassify[ int(xaxislabel[k],2) ]  = [float(yaxis[k]), float(error[k])]


final_yaxis = [racipeclassify[i][0] for i in final_index]
final_xaxislabel = [binlabelformat.format(i) for i in final_index]
final_error = [racipeclassify[i][1] for i in final_index]

argarr = np.argsort(np.array(final_yaxis))[::-1]

final_yaxis = [final_yaxis[i] for i in argarr]
final_error = [final_error[i] for i in argarr]
final_xaxislabel = [final_xaxislabel[i] for i in argarr]
final_xaxis = np.linspace(0, len(final_yaxis), len(final_yaxis))
plt.figure(figsize=(20,12))

#print(final_yaxis)
#print(final_xaxislabel)

if(not(plot_plotterdata)):
    rcParams.update({'figure.autolayout':True})
    plt.bar(final_xaxis,final_yaxis,  width = 0.3, color = 'r')
    plt.errorbar(final_xaxis,final_yaxis, yerr = final_error , color = 'k', fmt = 'o', markersize = 0, barsabove = True, capsize = 3)
    plt.xticks(final_xaxis, final_xaxislabel, rotation = 'vertical')
    # plt.show()
    plt.savefig("distrplot/{}_racipeVbool.png".format(network_name))

# print(argarr)
# print(yaxis)


if plot_plotterdata:
    dataplotter = open("output/{}_ssprob_all.txt".format("{}_async_unweigh".format(network_name)),'r').read().split("\n")[1:-1]
    x_labels = []
    y_data = []
    error_data = []
    # print(dataplotter)
    for temp in dataplotter:
        # print(i)
        i = temp.split(" ")
        x_labels.append(i[0])
        y_data.append(float(i[1]))
        error_data.append(float(i[2]))

    final_yaxislist = list(final_yaxis)
    final_errorlist = list(final_error)

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
            try:
                final_yaxislist.append(racipeclassify[int(x_labels[i],2)][0])
                final_errorlist.append(racipeclassify[int(x_labels[i],2)][1])
            except:
                final_yaxislist.append(0)
                final_errorlist.append(0)
            cnt+=1

    final_ydata = np.zeros(len(list_labels))
    final_errordata = np.zeros(len(list_labels))
    for i in range(len(y_data)):
        if(x_labels[i] in dict_yaxis):
            index=dict_yaxis[x_labels[i]]
            final_ydata[index]=y_data[i]
            final_errordata[index] = error_data[i]


    weight = 0.35
    final_xaxis=[i for i in range(len(list_labels)) ]
    #rcParams.update({'figure.autolayout':True})
    final_yaxislist=np.array(final_yaxislist)
    final_errorlist = np.array(final_errorlist)
    rac_sum=0
    for q in range(len( final_yaxislist )):
        rac_sum+=final_yaxislist[q]

    final_yaxislist=  final_yaxislist / rac_sum
    final_errorlist=  final_errorlist / rac_sum

    bool_sum=0
    for q in range(len( final_ydata )):
        bool_sum += final_ydata[q]

    final_ydata=  final_ydata / bool_sum
    final_errordata = final_errordata / bool_sum
    #plt.figure(figsize = (20,10))
    fig, ax = plt.subplots(1,1)
    plt.bar(final_xaxis,np.array(final_yaxislist) , width = 0.3, color = 'r')
    plt.errorbar(final_xaxis,np.array(final_yaxislist) , yerr = final_errorlist, fmt = 'o', markersize = 0, barsabove = True, capsize = 3, color = 'k')
    # bars =  plt.bar(final_xaxis,np.array(final_yaxislist) , width = 0.3, color = 'r')
    plt.bar(np.array(final_xaxis)+weight, final_ydata , width = 0.3, color = 'b')
    plt.errorbar(np.array(final_xaxis)+weight, final_ydata, yerr = final_errordata, fmt = 'o', markersize = 0, barsabove = True, capsize = 3, color = 'k')
    plt.xticks(final_xaxis,list_labels, rotation = 'vertical')
    jsd=jensenshannon(final_ydata,np.array(final_yaxislist),2 )
    plt.title("{}".format(network_name) +" "+ version + "     " + "JSD = {}".format( "{:.4f}".format(jsd)))

    #titles= ["{}".format(network_name) +" "+ version , "JSD = {}".format( "{:.4f}".format(jsd)) ]
    #plt.title(r'\textcolor{red}{ "{}".format(titles[0])}  +

    #        r'\textcolor{blue}{cloudy.}')
    #plt.title( titles, color=['r','b'] )
    ax.set_xlabel("Stable States", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)

    # degarr=[0]*len(id_to_node)
    # activationsum=[0]*len(list_labels)
    # for w in range ( len(list_labels ) ):
    #     curstate=[0] * len(id_to_node)
    #     for k in range(len(id_to_node)):
    #         if(list_labels[w][k]=='0'):
    #             curstate[k]=-1
    #         else:
    #             curstate[k]=1
    #     activation= np.matmul(curstate,link_matrix)
    #     for u in range(len(id_to_node)):
    #         activationsum[w] += activation[u] * curstate[u]
    #
    # for u in range(len(id_to_node)):
    #     for k in range(len(id_to_node)):
    #         degarr[u]+=abs ( link_matrix[k][u] )


    #print(activationsum)
    #for i, v in enumerate(activationsum):
    #    print(v)
    #    plt.text(final_xaxis[i] , max(np.array(final_yaxislist)[i], final_ydata[i]) + 0.005, str(v), fontsize=12,color='g')
    #qq=0
    #for bar in bars:
    #    yval = activationsum[qq]
    #    qq+=1
    #
    #    ax.text(bar.get_x(), yval + .005, yval, ha='center', va='bottom')

    plt.tight_layout()
    #plt.figtext(0.5, 0.01, "{}".format(network_name) +" "+ version, ha="center", fontsize=18, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})

    leg1 = mlines.Line2D([], [], color='red', ls='', marker = 's', label='RACIPE')
    leg2 = mlines.Line2D([], [], color='blue',  ls='', marker = 's',  label='Boolean')
    plt.legend(handles=[leg1, leg2])

    plt.savefig("Boolvsracipegraphs/{}_racipeV{}test_error.png".format(network_name,version))
    #try:
    #    jsdfile=open("cont_jsd.txt","a")
    #except:
    #    jsdfile=open("cont_jsd.txt","w+")
    jsdfile=open("{}_jsd.txt".format(version),"a")
    jsdfile.write("{} {}\n".format(network_name,jsd))
    jsdfile.close()
    # print(final_ydata)

# print(xaxislabel)
