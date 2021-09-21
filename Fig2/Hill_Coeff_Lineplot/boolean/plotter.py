import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import math
import initialise.initialise as initialise
import initialise.parser as parser

print("modules")


def plot_bar(filename, id_to_node, params):
    filename_index = 0
    for i in range(len(params['input_filenames'])):
        if "{}_".format(params['input_filenames'][i]) in filename:
            filename_index = i
            break

    prob_matrix = []
    for i in range(params['num_runs']):
        prob_matrix.append({})

    string_setbin = "{0:0" + str(len(id_to_node) - params['constant_node_count'][filename_index]) + "b}"
    for cur_run in range(params['num_runs']):
        with open("{}/{}_ss_run{}.txt".format(params['output_folder_name'], filename, cur_run + 1)) as f:
            data_lines = f.readline().split(' ')
            ncols = len(data_lines)
            first_row = data_lines[0][1:]

        data = np.loadtxt("{}/{}_ss_run{}.txt".format(params['output_folder_name'], filename, cur_run + 1),
                          delimiter=' ', skiprows=1, usecols=range(1, ncols))

        gene_id = list(first_row)

        for i in range(params['constant_node_count'][filename_index]):
            gene_id.remove(id_to_node[i])
        data = data[:, params['constant_node_count'][filename_index]:]

        for i in data:
            tempstr = ""
            for j in i:
                tempstr += '0' if j < 0 else '1'
            nodeval = int(tempstr, 2)

            try:
                prob_matrix[cur_run][nodeval] += 1
            except:
                prob_matrix[cur_run][nodeval] = 0

        for i in prob_matrix[cur_run].keys():
            prob_matrix[cur_run][i] /= params['num_simulations']

        prob_file = open("{}/{}_ssprob_run{}.txt".format(params['output_folder_name'], filename, cur_run + 1), 'w')
        for i in prob_matrix[cur_run].keys():
            prob_file.write("{} {}\n".format(string_setbin.format(i), prob_matrix[cur_run][i]))

        prob_file.close()

    keys_arr = {}
    keys_list = []
    probmatrix_final = []
    counter_index = 0

    for i in range(params['num_runs']):
        for j in prob_matrix[i].keys():
            if j not in keys_arr.keys():
                keys_list.append(j)
                keys_arr[j] = counter_index
                counter_index += 1

    for i in range(params['num_runs']):
        temparr = [0] * len(keys_arr.keys())
        for j in prob_matrix[i].keys():
            if j in keys_arr.keys():
                temparr[keys_arr[j]] = prob_matrix[i][j]
        probmatrix_final.append(temparr)

    prob_matrix = np.matrix(probmatrix_final)

    error = [0 for i in range(len(keys_arr.keys()))]
    final = [0 for i in range(len(keys_arr.keys()))]
    finalin = [i for i in range(len(keys_arr.keys()))]
    yterr = [[0] * len(final), [0] * len(final)]

    for i in range(len(keys_arr.keys())):

        error[i] = np.std(prob_matrix[:, i])

        final[i] = np.mean(prob_matrix[:, i])

        dev1 = 0
        dev2 = 0
        cnt1 = 0
        cnt2 = 0
        arr1 = prob_matrix[:, i]
        for v in range(params['num_runs']):
            if (arr1[v] >= final[i]):
                dev1 += (arr1[v] - final[i]) ** 2
                cnt1 += 1
            else:
                dev2 += (arr1[v] - final[i]) ** 2
                cnt2 += 1
        if (cnt1 != 0):
            yterr[1][i] = math.sqrt(dev1 / cnt1)
        if (cnt2 != 0):
            yterr[0][i] = math.sqrt(dev2 / cnt2)

    x_label = "States: "
    for i in gene_id:
        x_label += i + " "
    remove_index = []
    for i in range(len(final)):
        if final[i] == 0:
            remove_index.append(i)
    num_cycles = 0
    for temp in remove_index:
        i = temp - num_cycles
        final.remove(final[i])
        finalin.remove(finalin[i])
        error.remove(error[i])
        keys_list.remove(keys_list[i])

        yterr[0].remove(yterr[0][i])
        yterr[1].remove(yterr[1][i])
        num_cycles += 1

    prob_file = open("{}/{}_ssprob_all.txt".format(params['output_folder_name'], filename), 'w')
    prob_file.write("Node_Config Probability Error\n")
    for i in range(len(final)):
        prob_file.write("{} {:.6f} {:.6f}\n".format(string_setbin.format(keys_list[i]), final[i], error[i]))
    prob_file.close()

    notfinal_index = []
    notfinal = []
    notset_bin_fin = []
    notyterr0 = []
    notyterr1 = []

    for i in range(len(final)):
        if final[i] < 0.01:
            notfinal_index.append(i)
            notfinal.append(final[i])
            notset_bin_fin.append(string_setbin.format(keys_list[i]))
            notyterr0.append(yterr[0][i])
            notyterr1.append(yterr[1][i])

    for i in range(len(notfinal_index)):
        final.remove(notfinal[i])

        yterr[0].remove(notyterr0[i])
        yterr[1].remove(notyterr1[i])

    argarr = np.argsort(final)[::-1]

    set_bin_fin = [string_setbin.format(keys_list[i]) for i in argarr]

    final = [final[i] for i in argarr]

    yterr[0] = [yterr[0][i] for i in argarr]
    yterr[1] = [yterr[1][i] for i in argarr]

    rcParams.update({'figure.autolayout': True})  # NOTE!!!!!!!!! properly resizes things :D
    plt.figure(figsize=(20, 10))
    plt.subplots_adjust(0.1, 0.5, 0.9, 0.9)
    plt.title("{}_steady_states".format(filename))
    plt.xlabel(x_label)

    plt.xticks(rotation='vertical')

    plt.bar(set_bin_fin, final, yerr=yterr, capsize=5)

    plt.savefig("{}/{}/{}_ss_barplot.png".format(params['output_folder_name'], 'graphs', filename))


in_file = 'init.txt'
begin = 1
process_count = 1
params = initialise.initialise(in_file)
initialise.create_folders(params)
for j in params['input_filenames']:
    print(j)
    random_seed = int(begin) + process_count
    link_matrix, id_to_node = parser.parse_topo(params, j, random_seed)
    plot_bar(j, id_to_node, params)
