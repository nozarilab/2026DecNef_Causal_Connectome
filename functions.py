
import numpy as np
import datetime
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import matplotlib.colors as colors
import scipy


def apply_mask(fmri_data,parcellation_mask):
    mask_idx = np.where(parcellation_mask > 0)
#     print('mask:',mask_idx[1],mask_idx[1].shape)
    data = fmri_data[np.array(mask_idx[1]),:]
    data_updated = data[np.std(data, axis = 1)> 0,:]
   
    parcellation_mask_updated = parcellation_mask[parcellation_mask > 0]
    parcellation_mask_updated = parcellation_mask_updated[np.std(data, axis = 1)> 0]
    
    return data_updated,parcellation_mask_updated

# Detrening function
def detrend(data_sample,parcellation_mask_updated):
    num_voxels = data_sample.shape[0]
    num_time_points = data_sample.shape[1]
    t = np.reshape(np.linspace(0,num_time_points,num_time_points),(num_time_points,1))    
    data_sample_detrend = np.zeros((num_voxels,num_time_points))
    for v in range(0, num_voxels):
        sig = np.reshape(data_sample[v,:],(num_time_points,1))
        reg = LinearRegression().fit(t, sig)
        reg.score(t,sig)
        a = reg.coef_
        b = reg.intercept_
        y = a*t+b
        sig_linear_trend_removed = sig - y
        data_sample_detrend[v,:] = np.squeeze(sig_linear_trend_removed)/np.std(np.squeeze(sig_linear_trend_removed))
    return data_sample_detrend

# Helper functions
def parcelate(fmri_data, parcellation_mask, num_parcels):

    num_samples = fmri_data.shape[1]
    parcelated_data = np.zeros((num_samples, num_parcels))

    for p in range(0, num_parcels):
        voxel_idx = np.where(parcellation_mask == (p+1))[0]
        parcelated_data[:,p] = np.mean(fmri_data[voxel_idx,:], axis = 0)


    return parcelated_data

def regressout_confounders(fmri_data,confounders_9):
    
    y = fmri_data.T
    A = confounders_9.T
    x = np.linalg.pinv(A)@y

    print(np.max(np.max(A@x)), np.min(np.min(A@x)),np.max(np.max(y)), np.min(np.min(y)) )
    y_err = y - A@x

    fmri_data_confounders_removed = y_err.T

    return fmri_data_confounders_removed

def compute_cohens_d(x,y):

    n1 = x.shape[0]
    n2 = y.shape[0]

    s1 = np.std(x, ddof = 1)
    s2 = np.std(y, ddof = 1)
    s = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2)/(n1+n2-2))
    cohens_d = (np.mean(x)-np.mean(y))/s
    return cohens_d

# functions needed for causal discovery analysis

# This function plot a directional graph from a binary matrix
def plot_graph(matrix, nodes, title):

    node_color = 'lightblue'
    node_size = 2500
    font_size = 28
    edge_width = 3

    G = nx.from_numpy_matrix(matrix, create_using = nx.MultiDiGraph())  

    pos=nx.circular_layout(G) 

    nx.draw_networkx(G, node_size = node_size, node_color=node_color, font_size = font_size, pos = pos, width = edge_width)

    # labels = nx.draw_networkx_labels(G, pos= pos)
    plt.title(title, fontsize = 40)


def is_strongly_connected(graph):

    n = graph.shape[0]
    S = graph
    k = 1

    Full = np.all((S != 0))
    
    S_binarized = np.zeros_like(S)
    while (k <= n):

        S = S + np.linalg.matrix_power(graph, k+1)
        k = k + 1

        Full = np.all((S != 0))

    disconnected = np.where (S == 0)
    
    S_binarized[S != 0] = 1

    return Full, disconnected, S_binarized,S

def import_npz(npz_file):
    Data = np.load(npz_file, allow_pickle= True)
    print(Data)
    for varName in Data:
        globals()[varName] = Data[varName]  

        

def network_map(graph,net_labels,network_dic,parcel_labels, num_cortical_parcels):


    num_networks = len(net_labels)
    num_parcels = len(parcel_labels)

    network_parcel_idx= eval(network_dic)

    for i in range (0 , num_parcels):

        if i < num_cortical_parcels:
            temp = parcel_labels[i]
            temp_first = temp[0:6]
            if temp_first in network_parcel_idx:
                network_parcel_idx[temp_first].append(i)
        elif i >= num_cortical_parcels  and i<num_cortical_parcels +8:
            network_parcel_idx['RH_Sub'].append(i)
        
        else:
            network_parcel_idx['LH_Sub'].append(i)

    all_network_normalized_num_edges_from_to = np.zeros((num_networks,num_networks))

    for i in range(0, num_networks):
        for j in range(0,num_networks):

            from_net = net_labels[i]
            to_net = net_labels[j]

            from_net_parcel_idx =  np.array(network_parcel_idx[from_net])
            to_net_parcel_idx =  np.array(network_parcel_idx[to_net])

            if len(to_net_parcel_idx) != 0 and len(from_net_parcel_idx) != 0:

                temp = graph[:,to_net_parcel_idx]
                temp2 = temp[from_net_parcel_idx,:]

                if from_net == to_net:
                    np.fill_diagonal(temp2, 0)

                num_edges_from_to = np.sum(temp2)
                all_network_normalized_num_edges_from_to[i,j] = num_edges_from_to/(len(from_net_parcel_idx)*len(to_net_parcel_idx))
            
    return all_network_normalized_num_edges_from_to



    
        
def network_map_v2(graph,net_labels,network_dic,parcel_labels):


    num_parcels = len(parcel_labels)

    num_networks = len(net_labels)

    network_parcel_idx = eval(network_dic)
    decoder_parcel_labels = ['LH_DorsAttn_Post_1', 'RH_Default_Temp_1','LH_Limbic_TempPole_1','LH_Limbic_TempPole_2', 'RH_Limbic_TempPole_1', 'LH_Vis_2' , 'LH_Vis_7', 'RH_Vis_2','RH_Vis_3']

    for i in range (0 , num_parcels):

        if i < 8:
            network_parcel_idx['RH_Sub'].append(i)

        elif i>=8 and i<16:
            network_parcel_idx['LH_Sub'].append(i)

        elif i>=16:
            temp = parcel_labels[i]
            temp_first = temp[0:6]

            if temp in decoder_parcel_labels:
                network_parcel_idx['Decoder'].append(i)
            elif temp_first in network_parcel_idx:
                network_parcel_idx[temp_first].append(i)

    all_network_normalized_num_edges_from_to = np.zeros((num_networks,num_networks))
    for i in range(0, num_networks):
        for j in range(0,num_networks):

            from_net = net_labels[i]
            to_net = net_labels[j]

            from_net_parcel_idx =  np.array(network_parcel_idx[from_net])
            to_net_parcel_idx =  np.array(network_parcel_idx[to_net])

            if len(to_net_parcel_idx) != 0 and len(from_net_parcel_idx) != 0:

                temp = graph[:,to_net_parcel_idx]
                temp2 = temp[from_net_parcel_idx,:]

                if from_net == to_net:
                    np.fill_diagonal(temp2, 0)

                num_edges_from_to = np.sum(temp2)
                all_network_normalized_num_edges_from_to[i,j] = num_edges_from_to/(len(from_net_parcel_idx)*len(to_net_parcel_idx))
            
    return all_network_normalized_num_edges_from_to


def network_map_v2_without_decoder(graph,net_labels,network_dic,parcel_labels):


    num_parcels = len(parcel_labels)
    num_networks = len(net_labels)

    network_parcel_idx = eval(network_dic)
    network_parcel_labels = eval(network_dic)

    for i in range (0 , num_parcels):

        if i < 8:
            network_parcel_idx['RH_Sub'].append(i)
            network_parcel_labels['RH_Sub'].append(parcel_labels[i])

        elif i>=8 and i<16:
            network_parcel_idx['LH_Sub'].append(i)
            network_parcel_labels['LH_Sub'].append(parcel_labels[i])

        elif i>=16:
            temp = parcel_labels[i]
            temp_first = temp[0:6]
            if temp_first in network_parcel_idx:
                network_parcel_idx[temp_first].append(i)
                network_parcel_labels[temp_first].append(parcel_labels[i])


    all_network_normalized_num_edges_from_to = np.zeros((num_networks,num_networks))
    for i in range(0, num_networks):
        for j in range(0,num_networks):

            from_net = net_labels[i]
            to_net = net_labels[j]

            from_net_parcel_idx =  np.array(network_parcel_idx[from_net])
            to_net_parcel_idx =  np.array(network_parcel_idx[to_net])

            if len(to_net_parcel_idx) != 0 and len(from_net_parcel_idx) != 0:

                temp = graph[:,to_net_parcel_idx]
                temp2 = temp[from_net_parcel_idx,:]

                if from_net == to_net:
                    np.fill_diagonal(temp2, 0)

                num_edges_from_to = np.sum(temp2)
                all_network_normalized_num_edges_from_to[i,j] = num_edges_from_to/(len(from_net_parcel_idx)*len(to_net_parcel_idx))
            
    return all_network_normalized_num_edges_from_to


def network_map_v3(graph,net_labels,network_dic,parcel_labels):


    num_networks = len(net_labels)
    num_parcels = len(parcel_labels)

    network_parcel_idx= eval(network_dic)

    for i in range (0 , num_parcels):

            temp = parcel_labels[i]
            temp_first = temp[3:6]
            if temp_first in network_parcel_idx:
                network_parcel_idx[temp_first].append(i)

            if i<16:
                network_parcel_idx['Sub'].append(i)  
                

    all_network_normalized_num_edges_from_to = np.zeros((num_networks,num_networks))

    for i in range(0, num_networks):
        for j in range(0,num_networks):

            from_net = net_labels[i]
            to_net = net_labels[j]

            from_net_parcel_idx =  np.array(network_parcel_idx[from_net])
            to_net_parcel_idx =  np.array(network_parcel_idx[to_net])

            if len(to_net_parcel_idx) != 0 and len(from_net_parcel_idx) != 0:

                temp = graph[:,to_net_parcel_idx]
                temp2 = temp[from_net_parcel_idx,:]

                if from_net == to_net:
                    np.fill_diagonal(temp2, 0)

                num_edges_from_to = np.sum(temp2)
                all_network_normalized_num_edges_from_to[i,j] = num_edges_from_to/(len(from_net_parcel_idx)*len(to_net_parcel_idx))
            
    return all_network_normalized_num_edges_from_to

# def network_map_v4(graph,net_labels,network_dic,parcel_labels, decoder_parcel_labels):
def network_map_v4(graph,net_labels,network_dic,parcel_labels):


    num_parcels = len(parcel_labels)
    network_parcel_idx = eval(network_dic)
    num_networks = len(net_labels)

    for i in range (0 , num_parcels):

        if  i>=16:
            temp = parcel_labels[i]
            temp_first = temp[3:6]
            network_parcel_idx[temp_first].append(i)

    sub_cortical_idx = list(range(8,16))+ list(range(0,8))  ## we do this because in subcortical parcels, right comes first and we need to make it consistent with others where left comes first

    network_parcel_idx['Sub'] = sub_cortical_idx


    all_network_normalized_num_edges_from_to = np.zeros((num_networks,num_networks))
    for i in range(0, num_networks):
        for j in range(0,num_networks):

            from_net = net_labels[i]
            to_net = net_labels[j]

            from_net_parcel_idx =  np.array(network_parcel_idx[from_net])
            to_net_parcel_idx =  np.array(network_parcel_idx[to_net])

            if len(to_net_parcel_idx) != 0 and len(from_net_parcel_idx) != 0:

                temp = graph[:,to_net_parcel_idx]
                temp2 = temp[from_net_parcel_idx,:]

                if from_net == to_net:
                    np.fill_diagonal(temp2, 0)

                num_edges_from_to = np.sum(temp2)
                all_network_normalized_num_edges_from_to[i,j] = num_edges_from_to/(len(from_net_parcel_idx)*len(to_net_parcel_idx))
            
    return all_network_normalized_num_edges_from_to


def rearange_matrix(old_matrix,network_parcel_idx):

    new_idx_stop = 0
    new_matrix = np.zeros_like(old_matrix)
    for n in network_parcel_idx:

        new_idx_start = new_idx_stop
        new_idx_stop  = len(network_parcel_idx[n])+new_idx_start

        old_idx = network_parcel_idx[n]
        
        new_matrix[new_idx_start:new_idx_stop,:] = old_matrix[old_idx,:]

    out_matrix = np.zeros_like(old_matrix)
    new_idx_stop = 0
    for n in network_parcel_idx:

        new_idx_start = new_idx_stop
        new_idx_stop  = len(network_parcel_idx[n])+new_idx_start

        old_idx = network_parcel_idx[n]
        
        out_matrix[:,new_idx_start:new_idx_stop] = new_matrix[:,old_idx]

    return out_matrix


def rearange_matrix_v2(old_matrix,network_parcel_idx):

    new_idx_stop = 0
    new_matrix = np.zeros_like(old_matrix)
    for n in network_parcel_idx:

        new_idx_start = new_idx_stop
        new_idx_stop  = len(network_parcel_idx[n])+new_idx_start

        old_idx = network_parcel_idx[n]
        
        new_matrix[new_idx_start:new_idx_stop,:] = old_matrix[old_idx,:]

    out_matrix = np.zeros_like(old_matrix)
    new_idx_stop = 0
    for n in network_parcel_idx:

        new_idx_start = new_idx_stop
        new_idx_stop  = len(network_parcel_idx[n])+new_idx_start

        old_idx = network_parcel_idx[n]
        
        out_matrix[:,new_idx_start:new_idx_stop] = new_matrix[:,old_idx]

    return out_matrix



def convert_to_string_graph(graph_bool):
    
        graph = np.zeros(graph_bool.shape, dtype='<U3')
        graph[:] = ""
        graph[:,:,1:][graph_bool[:,:,1:]==1] = "-->"
        graph[:,:,0][np.logical_and(graph_bool[:,:,0]==1, 
                                    graph_bool[:,:,0].T==1)] = "o-o"
        graph[:,:,0][np.logical_and(graph_bool[:,:,0]==2, 
                                    graph_bool[:,:,0].T==2)] = "x-x"
        for (i,j) in zip(*np.where(
            np.logical_and(graph_bool[:,:,0]==1, graph_bool[:,:,0].T==0))):
            graph[i,j,0] = "-->"
            graph[j,i,0] = "<--"

        return graph

# this function reads the output txt file from causal-cmd software 
def monthToNum(shortMonth):
    return {
            'January': 1,
            'February': 2,
            'March': 3,
            'April': 4,
            'May': 5,
            'June': 6,
            'July': 7,
            'August': 8,
            'September': 9, 
            'October': 10,
            'November': 11,
            'December': 12
    }[shortMonth]
    
def tetrad_graph_parser(filename, ground_truth = 0):

    k = 0
    with open(filename) as f:
        for line in f:
            if line == "Graph Nodes:\n":
                nodes_line = k + 1
            if line == "Graph Edges:\n":
                edges_line = k + 1
            if line[0:5] == "Start":
                start_time_line = k
            if line[0:3] == "End":
                end_time_line = k
            k += 1 
    with open(filename) as f:
        all_lines = f.readlines()

    if ground_truth == 1:
        graph_nodes = all_lines[nodes_line].replace("\n","").split(",")
    else:
        graph_nodes = all_lines[nodes_line].replace("\n","").split(";")
        
    graph_edges = all_lines[edges_line:]

    num_nodes = len(graph_nodes)
    num_edges = len(graph_edges)


# calculate the execution time 
    execution_time = 0

    if ground_truth == 0:

        start_temp = all_lines[start_time_line][13:].replace("\n","").split(",")
        end_temp = all_lines[end_time_line][12:].replace("\n","").split(",")

        start_date = start_temp[1].split(" ")
        end_date = end_temp[1].split(" ")

        start_month = monthToNum(start_date[1])
        end_month = monthToNum(end_date[1])

        start_day = int(start_date[2])
        end_day = int(end_date[2])

        start_year = int(start_temp[2].split(" ")[1])
        end_year = int(end_temp[2].split(" ")[1])

        start_time = start_temp[2].split(" ")[2].split(":")
        end_time = end_temp[2].split(" ")[2].split(":")
        
        start_am_pm = start_temp[2].split(" ")[3]
        end_am_pm = end_temp[2].split(" ")[3]


        start_hour = int(start_time[0])
        start_min = int(start_time[1])
        start_second = int(start_time[2])

        end_hour = int(end_time[0])
        end_min = int(end_time[1])
        end_second = int(end_time[2])


        if (start_am_pm == 'PM') and (start_hour != 12):
            start_hour = start_hour + 12

        if (end_am_pm == 'PM') and (end_hour != 12):
            end_hour = end_hour + 12
        
        if (start_am_pm == 'AM') and (start_hour == 12):
            start_hour = 0

        if (end_am_pm == 'AM') and (end_hour == 12):
            end_hour = 0


        dt1 = datetime.datetime(start_year,start_month,start_day,start_hour,start_min,start_second) 
        dt2 = datetime.datetime(end_year,end_month,end_day,end_hour,end_min,end_second) 
        tdelta = dt2 - dt1 
        
        execution_time = tdelta.total_seconds()

        print(dt1, dt2,execution_time)


        # execution_time = end_total_sec - start_total_sec 
        
    graph_matrix = np.zeros((num_nodes,num_nodes))

    for e in range(0, num_edges):

        current_edge = graph_edges[e].replace(str(e+1)+".","").replace("\n","").replace(" ","").split("-->")
        i = int(current_edge[0].replace('X',''))-1
        j = int(current_edge[1].replace('X',''))-1

        graph_matrix[i,j] = 1

    return graph_nodes, graph_edges, graph_matrix, execution_time

# This function creats a time lag graph from the dictionary
def make_time_lag_graph_from_dict(graph_dict, tau_max):
    dict_vars = graph_dict.keys()
    num_vars = len(dict_vars)

    time_lag_graph = [[['' for t in range(0, tau_max+1)] for j in range(0,num_vars)] for i in range(0, num_vars)]

    for v1 in range(0,num_vars):
        var_name = 'X%d'%(v1+1)
        var_time_lags = graph_dict[var_name]

        for v2 in range(0,num_vars):
            for t in range(0,tau_max+1):
                if ('X%d'%(v2+1),-t) in var_time_lags:
                    time_lag_graph[v2][v1][t] = '-->'
                else:
                    time_lag_graph[v2][v1][t] = ''

    return time_lag_graph


# These set of functiona summarize a time lag graph to a summary graph by combining the edges accross the lags
# used for pcmci
def summarize_across_nonzero_lags(PAG):

        PAG_non_zero_lags = PAG[:,:,1:]
        m = PAG_non_zero_lags.shape[0]
        num_lags = PAG_non_zero_lags.shape[2]

        PAG_non_zero_lags_binary = np.zeros((m,m, num_lags))
        PAG_non_zero_lags_binary[PAG_non_zero_lags == '-->'] = 1

        binary_matrix = np.sum(PAG_non_zero_lags_binary, axis = 2)
        binary_matrix[binary_matrix >= 1] = 1

        return binary_matrix

def summarize_across_all_lags_v1(time_lag_graph):

    num_vars = time_lag_graph.shape[0]
    num_lags = time_lag_graph.shape[2]

    summary_graph = [['' for t in range(0, num_vars)] for i in range(0, num_vars)]

    adjacency_binary_matrix = np.zeros((num_vars, num_vars))
    orientation_binary_matrix = np.zeros((num_vars, num_vars))

    for i in range(0,num_vars):
        for j in range(0,num_vars):

            edges_all_lags  = time_lag_graph[i,j,:]

            if '-->' in edges_all_lags:
                summary_graph[i][j] = '-->'

                adjacency_binary_matrix[i,j] = 1
                orientation_binary_matrix[i,j] = 1

            elif ('-->' not in edges_all_lags and 'o-o' in edges_all_lags):
                summary_graph[i][j] = 'o-o'
                
                adjacency_binary_matrix[i,j] = 1
                orientation_binary_matrix[i,j] = 0

            else:
                summary_graph[i][j] = ''
                
                adjacency_binary_matrix[i,j] = 0
                orientation_binary_matrix[i,j] = 0

    
    return summary_graph, adjacency_binary_matrix, orientation_binary_matrix


def summarize_across_all_lags_v2(time_lag_graph):

    num_vars = time_lag_graph.shape[0]
    num_lags = time_lag_graph.shape[2]

    summary_graph = [['' for t in range(0, num_vars)] for i in range(0, num_vars)]

    adjacency_binary_matrix = np.zeros((num_vars, num_vars))
    orientation_binary_matrix = np.zeros((num_vars, num_vars))

    for i in range(0,num_vars):
        for j in range(0,num_vars):

            edges_all_lags  = time_lag_graph[i,j,:]

            if '-->' in edges_all_lags:
                summary_graph[i][j] = '-->'

                adjacency_binary_matrix[i,j] = 1
                orientation_binary_matrix[i,j] = 1

            elif ('-->' not in edges_all_lags and 'o-o' in edges_all_lags):
                summary_graph[i][j] = 'o-o'
                
                adjacency_binary_matrix[i,j] = 1
                orientation_binary_matrix[i,j] = 1

            else:
                summary_graph[i][j] = ''
                
                adjacency_binary_matrix[i,j] = 0
                orientation_binary_matrix[i,j] = 0
    
    return summary_graph, adjacency_binary_matrix, orientation_binary_matrix

def summarize_across_all_lags_v2_with_pvalues(time_lag_graph, p_matrix):

    num_vars = time_lag_graph.shape[0]
    num_lags = time_lag_graph.shape[2]

    summary_graph = [['' for t in range(0, num_vars)] for i in range(0, num_vars)]

    binary_matrix = np.zeros((num_vars, num_vars))
    binary_graph_p_matrix = np.zeros((num_vars, num_vars))

    for i in range(0,num_vars):
        for j in range(0,num_vars):

            edges_all_lags  = time_lag_graph[i,j,:]

            if '-->' in edges_all_lags:
                summary_graph[i][j] = '-->'
                binary_matrix[i,j] = 1

                which_lags = np.where(edges_all_lags =='-->')[0]
                binary_graph_p_matrix[i,j] = np.min(p_matrix[i,j,which_lags])

            elif ('-->' not in edges_all_lags and 'o-o' in edges_all_lags):
                summary_graph[i][j] = 'o-o'
                binary_matrix[i,j] = 1
                binary_graph_p_matrix[i,j]= p_matrix[i,j,0]

            else:
                summary_graph[i][j] = ''
                binary_matrix[i,j] = 0
                binary_graph_p_matrix[i,j] = np.min(p_matrix[i,j,:])
    
    return summary_graph, binary_matrix, binary_graph_p_matrix


def summarize_across_all_lags_v2_with_pvalues_more_directed_v2(time_lag_graph, p_matrix, statistics_value, alpha_level):

    num_vars = time_lag_graph.shape[0]
    num_lags = time_lag_graph.shape[2]

    binary_matrix = np.zeros((num_vars, num_vars))
    binary_graph_p_matrix = np.zeros((num_vars, num_vars))
    weighted_graph_statistics = np.zeros((num_vars, num_vars))
    alpha_level_ratio = 20   # was set to 20 for decnef graphs

    for i in range(0,num_vars):
        for j in range(0,i):

            edges_all_lags_ij  = time_lag_graph[i,j,:]
            edges_all_lags_ji  = time_lag_graph[j,i,:]

            if '-->' in edges_all_lags_ij:
                binary_matrix[i,j] = 1
                which_lags = np.where(np.logical_or(edges_all_lags_ij =='-->', edges_all_lags_ij =='o-o'))[0]
                binary_graph_p_matrix[i,j] = np.min(p_matrix[i,j,which_lags])
                weighted_graph_statistics[i,j] = statistics_value[i,j,which_lags[np.argmax(np.abs(statistics_value[i,j,which_lags]))]] 

            if '-->' in edges_all_lags_ji:
                binary_matrix[j,i] = 1
                which_lags = np.where(np.logical_or(edges_all_lags_ji =='-->' , edges_all_lags_ji =='o-o'))[0]
                binary_graph_p_matrix[j,i] = np.min(p_matrix[j,i,which_lags])
                weighted_graph_statistics[j,i] =  statistics_value[j,i,which_lags[np.argmax(np.abs(statistics_value[j,i,which_lags]))]] 
            

            elif ('-->' not in edges_all_lags_ij) and ('-->' not in edges_all_lags_ji) and ('o-o' in edges_all_lags_ji or 'o-o' in edges_all_lags_ij ):
                
                min_p_value_across_lags_ij = np.min(p_matrix[i,j,1:])
                min_p_value_across_lags_ji = np.min(p_matrix[j,i,1:])

                if min_p_value_across_lags_ij <= (alpha_level*alpha_level_ratio):
                    
                    binary_matrix[i,j] = 1
                    binary_graph_p_matrix[i,j]= np.min(p_matrix[i,j,:])
                    weighted_graph_statistics[i,j] = statistics_value[i,j,np.argmax(np.abs(statistics_value[i,j,:]))] 

                if min_p_value_across_lags_ji <= (alpha_level*alpha_level_ratio):
                    
                    binary_matrix[j,i] = 1
                    binary_graph_p_matrix[j,i]= np.min(p_matrix[j,i,:])
                    weighted_graph_statistics[j,i] = statistics_value[j,i,np.argmax(np.abs(statistics_value[j,i,:]))] 

                elif min_p_value_across_lags_ij> (alpha_level*alpha_level_ratio) and min_p_value_across_lags_ji> (alpha_level*alpha_level_ratio):

                    binary_matrix[j,i] = 1
                    binary_matrix[i,j] = 1

                    binary_graph_p_matrix[i,j]= p_matrix[i,j,0]
                    weighted_graph_statistics[i,j] = statistics_value[i,j,0]

                    binary_graph_p_matrix[j,i]= p_matrix[j,i,0]
                    weighted_graph_statistics[j,i] = statistics_value[j,i,0]  

            # else:
            elif ('-->' not in edges_all_lags_ij) and ('-->' not in edges_all_lags_ji) and ('o-o' not in edges_all_lags_ji) and ('o-o' not in edges_all_lags_ij ):
                binary_matrix[i,j] = 0
                binary_matrix[j,i] = 0

                binary_graph_p_matrix[i,j] = np.min(p_matrix[i,j,:])
                binary_graph_p_matrix[j,i] = np.min(p_matrix[j,i,:])

                weighted_graph_statistics[i,j] = 0
                weighted_graph_statistics[j,i] = 0

    return binary_matrix, binary_graph_p_matrix, weighted_graph_statistics


def summarize_across_all_lags_v2_with_pvalues_more_directed_test(time_lag_graph, p_matrix, statistics_value, alpha_level, alpha_level_ratio):

    num_vars = time_lag_graph.shape[0]
    num_lags = time_lag_graph.shape[2]

    binary_matrix = np.zeros((num_vars, num_vars))
    binary_graph_p_matrix = np.zeros((num_vars, num_vars))
    weighted_graph_statistics = np.zeros((num_vars, num_vars))
    
    for i in range(0,num_vars):
        for j in range(0,i):

            edges_all_lags_ij  = time_lag_graph[i,j,:]
            edges_all_lags_ji  = time_lag_graph[j,i,:]

            if '-->' in edges_all_lags_ij:
                binary_matrix[i,j] = 1
                which_lags = np.where(edges_all_lags_ij =='-->')[0]
                binary_graph_p_matrix[i,j] = np.min(p_matrix[i,j,which_lags])
                weighted_graph_statistics[i,j] = statistics_value[i,j,which_lags[np.argmax(np.abs(statistics_value[i,j,which_lags]))]] 

            if '-->' in edges_all_lags_ji:
                binary_matrix[j,i] = 1
                which_lags = np.where(edges_all_lags_ji =='-->')[0]
                binary_graph_p_matrix[j,i] = np.min(p_matrix[j,i,which_lags])
                weighted_graph_statistics[j,i] =  statistics_value[j,i,which_lags[np.argmax(np.abs(statistics_value[j,i,which_lags]))]] 
            

            elif ('-->' not in edges_all_lags_ij) and ('-->' not in edges_all_lags_ji) and ('o-o' in edges_all_lags_ji or 'o-o' in edges_all_lags_ij ):
                
                min_p_value_across_lags_ij = np.min(p_matrix[i,j,1:])
                min_p_value_across_lags_ji = np.min(p_matrix[j,i,1:])

                if min_p_value_across_lags_ij <= (alpha_level*alpha_level_ratio):
                    
                    binary_matrix[i,j] = 1
                    binary_graph_p_matrix[i,j]= np.min(p_matrix[i,j,:])
                    weighted_graph_statistics[i,j] = statistics_value[i,j,np.argmax(np.abs(statistics_value[i,j,:]))] 

                if min_p_value_across_lags_ji <= (alpha_level*alpha_level_ratio):
                    
                    binary_matrix[j,i] = 1
                    binary_graph_p_matrix[j,i]= np.min(p_matrix[j,i,:])
                    weighted_graph_statistics[j,i] = statistics_value[j,i,np.argmax(np.abs(statistics_value[j,i,:]))] 

                elif min_p_value_across_lags_ij> (alpha_level*alpha_level_ratio) and min_p_value_across_lags_ji> (alpha_level*alpha_level_ratio):

                    binary_matrix[j,i] = 1
                    binary_matrix[i,j] = 1

                    binary_graph_p_matrix[i,j]= p_matrix[i,j,0]
                    weighted_graph_statistics[i,j] = statistics_value[i,j,0]

                    binary_graph_p_matrix[j,i]= p_matrix[j,i,0]
                    weighted_graph_statistics[j,i] = statistics_value[j,i,0]  

            # else:
            elif ('-->' not in edges_all_lags_ij) and ('-->' not in edges_all_lags_ji) and ('o-o' not in edges_all_lags_ji) and ('o-o' not in edges_all_lags_ij ):
                binary_matrix[i,j] = 0
                binary_matrix[j,i] = 0

                binary_graph_p_matrix[i,j] = np.min(p_matrix[i,j,:])
                binary_graph_p_matrix[j,i] = np.min(p_matrix[j,i,:])

                weighted_graph_statistics[i,j] = 0
                weighted_graph_statistics[j,i] = 0

    return binary_matrix, binary_graph_p_matrix, weighted_graph_statistics

# The following function computes TPR,FPR, Recall, Precision, and F1 score for an estimated graph
def find_tpr_fpr_from_binary_matrix(true_graph_matrix, estimated_graph_matrix):

    adj_TP = 0
    adj_FN = 0
    adj_FP = 0
    adj_TN = 0

    n, m = np.shape(true_graph_matrix)

    for i in range(n):
        for j in range(i):

            if (true_graph_matrix[i,j] or  true_graph_matrix[j,i] ) == 1 and (estimated_graph_matrix[i,j] or  estimated_graph_matrix[j,i] ) == 1:
                adj_TP += 1
            elif (true_graph_matrix[i,j] or  true_graph_matrix[j,i] ) == 0 and (estimated_graph_matrix[i,j] or  estimated_graph_matrix[j,i] ) == 0:
                adj_TN += 1 
            elif (true_graph_matrix[i,j] or  true_graph_matrix[j,i] ) == 1 and (estimated_graph_matrix[i,j] or  estimated_graph_matrix[j,i] ) == 0:
                adj_FN += 1 
            elif (true_graph_matrix[i,j] or  true_graph_matrix[j,i] ) == 0 and (estimated_graph_matrix[i,j] or  estimated_graph_matrix[j,i] ) == 1:
                adj_FP += 1 

    if adj_FN == 0:
        adjacency_TPR = 1
    else:
        adjacency_TPR = adj_TP/(adj_TP + adj_FN)

    if adj_TN == 0:
        adjacency_FPR = 1
    else:
         adjacency_FPR = adj_FP/(adj_FP + adj_TN)

    if adj_FN == 0:
        adjacency_Recall = 1
    else:
        adjacency_Recall = adj_TP/(adj_TP + adj_FN)

    if adj_FP == 0:
        adjacency_Precision = 1
    else:
        adjacency_Precision = adj_TP/(adj_TP + adj_FP)

    if adjacency_Recall == 0 and adjacency_Precision == 0:
        adjacency_F1_score = 0
    else:
        adjacency_F1_score = 2*adjacency_Recall*adjacency_Precision/(adjacency_Precision + adjacency_Recall)


    ori_TP = 0
    ori_FN = 0
    ori_FP = 0
    ori_TN = 0

    n, m = np.shape(true_graph_matrix)

    for i in range(n):
        for j in range(m):

            if (true_graph_matrix[i,j]  == 1) and (estimated_graph_matrix[i,j] == 1):
                ori_TP += 1
            elif (true_graph_matrix[i,j]  == 0) and (estimated_graph_matrix[i,j] == 0):
                ori_TN += 1 
            elif (true_graph_matrix[i,j]  == 1) and (estimated_graph_matrix[i,j] == 0):
                ori_FN += 1 
            elif (true_graph_matrix[i,j]  == 0) and (estimated_graph_matrix[i,j] == 1):
                ori_FP += 1 

    if ori_FN == 0:
        orientation_TPR = 1
    else:
         orientation_TPR = ori_TP/(ori_TP + ori_FN)

    if ori_TN == 0:
        orientation_FPR = 1
    else:
        orientation_FPR = ori_FP/(ori_FP + ori_TN)

    if ori_FN == 0:
        orientation_Recall = 1
    else:
        orientation_Recall = ori_TP/(ori_TP + ori_FN)

    if ori_FP == 0:
        orientation_Precision = 1
    else:
        orientation_Precision = ori_TP/(ori_TP + ori_FP)

    if orientation_Recall == 0 and orientation_Precision == 0:
        orientation_F1_score = 0
    else:
        orientation_F1_score = 2*orientation_Recall*orientation_Precision/(orientation_Precision + orientation_Recall)

    return adjacency_TPR, adjacency_FPR, adjacency_Recall, adjacency_Precision, adjacency_F1_score, orientation_TPR, orientation_FPR, orientation_Recall,orientation_Precision,orientation_F1_score


def find_tpr_fpr_from_orientation_and_adjacency_matrices(true_graph_matrix, estimated_adjacency_binary_matrix, estimated_orientation_binary_matrix):

    adj_TP = 0
    adj_FN = 0
    adj_FP = 0
    adj_TN = 0

    n, m = np.shape(true_graph_matrix)

    for i in range(n):
        for j in range(i):

            if (true_graph_matrix[i,j] or  true_graph_matrix[j,i] ) == 1 and (estimated_adjacency_binary_matrix[i,j] or  estimated_adjacency_binary_matrix[j,i] ) == 1:
                adj_TP += 1
            elif (true_graph_matrix[i,j] or  true_graph_matrix[j,i] ) == 0 and (estimated_adjacency_binary_matrix[i,j] or  estimated_adjacency_binary_matrix[j,i] ) == 0:
                adj_TN += 1 
            elif (true_graph_matrix[i,j] or  true_graph_matrix[j,i] ) == 1 and (estimated_adjacency_binary_matrix[i,j] or  estimated_adjacency_binary_matrix[j,i] ) == 0:
                adj_FN += 1 
            elif (true_graph_matrix[i,j] or  true_graph_matrix[j,i] ) == 0 and (estimated_adjacency_binary_matrix[i,j] or  estimated_adjacency_binary_matrix[j,i] ) == 1:
                adj_FP += 1 

    if adj_FN == 0:
        adjacency_TPR = 1
    else:
        adjacency_TPR = adj_TP/(adj_TP + adj_FN)

    if adj_TN == 0:
        adjacency_FPR = 1
    else:
         adjacency_FPR = adj_FP/(adj_FP + adj_TN)

    if adj_FN == 0:
        adjacency_Recall = 1
    else:
        adjacency_Recall = adj_TP/(adj_TP + adj_FN)

    if adj_FP == 0:
        adjacency_Precision = 1
    else:
        adjacency_Precision = adj_TP/(adj_TP + adj_FP)

    if adjacency_Recall == 0 and adjacency_Precision == 0:
        adjacency_F1_score = 0
    else:
        adjacency_F1_score = 2*adjacency_Recall*adjacency_Precision/(adjacency_Precision + adjacency_Recall)


    ori_TP = 0
    ori_FN = 0
    ori_FP = 0
    ori_TN = 0

    n, m = np.shape(true_graph_matrix)

    for i in range(n):
        for j in range(m):

            if (true_graph_matrix[i,j]  == 1) and (estimated_orientation_binary_matrix[i,j] == 1):
                ori_TP += 1
            elif (true_graph_matrix[i,j]  == 0) and (estimated_orientation_binary_matrix[i,j] == 0):
                ori_TN += 1 
            elif (true_graph_matrix[i,j]  == 1) and (estimated_orientation_binary_matrix[i,j] == 0):
                ori_FN += 1 
            elif (true_graph_matrix[i,j]  == 0) and (estimated_orientation_binary_matrix[i,j] == 1):
                ori_FP += 1 

    if ori_FN == 0:
        orientation_TPR = 1
    else:
         orientation_TPR = ori_TP/(ori_TP + ori_FN)

    if ori_TN == 0:
        orientation_FPR = 1
    else:
        orientation_FPR = ori_FP/(ori_FP + ori_TN)

    if ori_FN == 0:
        orientation_Recall = 1
    else:
        orientation_Recall = ori_TP/(ori_TP + ori_FN)

    if ori_FP == 0:
        orientation_Precision = 1
    else:
        orientation_Precision = ori_TP/(ori_TP + ori_FP)

    if orientation_Recall == 0 and orientation_Precision == 0:
        orientation_F1_score = 0
    else:
        orientation_F1_score = 2*orientation_Recall*orientation_Precision/(orientation_Precision + orientation_Recall)

    return adjacency_TPR, adjacency_FPR, adjacency_Recall, adjacency_Precision, adjacency_F1_score, orientation_TPR, orientation_FPR, orientation_Recall,orientation_Precision,orientation_F1_score



def my_draw_networkx_edge_labels(
    G,
    pos,
    edge_labels=None,
    label_pos=0.1,
    font_size=10,
    font_color="k",
    font_family="sans-serif",
    font_weight="normal",
    alpha=None,
    bbox=None,
    horizontalalignment="center",
    verticalalignment="center",
    ax=None,
    rotate=True,
    clip_on=True,
    rad=0
):

    if ax is None:
        ax = plt.gca()
    if edge_labels is None:
        labels = {(u, v): d for u, v, d in G.edges(data=True)}
    else:
        labels = edge_labels
    text_items = {}
    for (n1, n2), label in labels.items():
        (x1, y1) = pos[n1]
        (x2, y2) = pos[n2]
        (x, y) = (
            x1 * label_pos + x2 * (1.0 - label_pos),
            y1 * label_pos + y2 * (1.0 - label_pos),
        )
        pos_1 = ax.transData.transform(np.array(pos[n1]))
        pos_2 = ax.transData.transform(np.array(pos[n2]))
        linear_mid = 0.5*pos_1 + 0.5*pos_2
        d_pos = pos_2 - pos_1
        rotation_matrix = np.array([(0,1), (-1,0)])
        ctrl_1 = linear_mid + rad*rotation_matrix@d_pos
        ctrl_mid_1 = 0.5*pos_1 + 0.5*ctrl_1
        ctrl_mid_2 = 0.5*pos_2 + 0.5*ctrl_1
        bezier_mid = 0.5*ctrl_mid_1 + 0.5*ctrl_mid_2
        (x, y) = ax.transData.inverted().transform(bezier_mid)

        if rotate:
            # in degrees
            angle = np.arctan2(y2 - y1, x2 - x1) / (2.0 * np.pi) * 360
            # make label orientation "right-side-up"
            if angle > 90:
                angle -= 180
            if angle < -90:
                angle += 180
            # transform data coordinate angle to screen coordinate angle
            xy = np.array((x, y))
            trans_angle = ax.transData.transform_angles(
                np.array((angle,)), xy.reshape((1, 2))
            )[0]
        else:
            trans_angle = 0.0
        # use default box of white with white border
        if bbox is None:
            bbox = dict(boxstyle="round", ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0))
        if not isinstance(label, str):
            label = str(label)  # this makes "1" and 1 labeled the same


 
        t = ax.text(
            x,
            y,
            label,
            size=font_size,
            color=font_color,
            family=font_family,
            weight=font_weight,
            alpha=alpha,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
            rotation=trans_angle,
            transform=ax.transData,
            bbox=bbox,
            zorder=1,
            clip_on=clip_on,
        )
        text_items[(n1, n2)] = t

    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )

    return text_items




def plot_graph_from_matrix(matrix ,array_for_node_colors, array_for_node_sizes, labels, ax, min_node_size= 1000, max_node_size= 10000, thick_width = 8, thin_width = 3 , edge_label_font_size = 20, node_label_font_size= 16, arc_rad= 0.07, arrow_size= 26):


    G = nx.DiGraph()

    # Add nodes to the graph with actual labels from net_labels_visualization
    node_labels = {i: label for i, label in enumerate(labels)}
    G.add_nodes_from(node_labels.keys())

    # Add edges with weights from the matrix
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] != 0:
                G.add_edge(i, j, weight=matrix[i][j])

    # pos = nx.spring_layout(G)
    pos = nx.kamada_kawai_layout(G)
    # pos = nx.shell_layout(G)
    # pos = nx.spectral_layout(G)


    ##
    # {0: 'Vis',
    #  1: 'Som-Mot',
    #  2: 'Dors-Att',
    #  3: 'Vent-Att',
    #  4: 'Lim',
    #  5: 'Cont',
    #  6: 'Def'}
    # pos[0] = pos[0] + [0.5,0.5]
    # pos[1] = pos[1] + [0.0, 0.3]
    # pos[2] = pos[2] + [0.1,0]
    # pos[3] = pos[3] + [0.9,0]
    # pos[4] = pos[4]+[-0.5,0]
    # pos[5] = pos[5] + [0.5, +0.2]
    # pos[6] = pos[6]+ [1.5,0]

    # Extracting edge weights
    curved_edges = [edge for edge in G.edges() if reversed(edge) in G.edges()]
    straight_edges = list(set(G.edges()) - set(curved_edges))

    edge_weights  = nx.get_edge_attributes(G,'weight')

    thick_width = thick_width
    thin_width = thin_width

    # straight edges
    straight_edges_weight=[]
    straight_edges_width = []
    for i in range(0,len(straight_edges)):
        straight_edges_weight.append(edge_weights[straight_edges[i]])
        straight_edges_width.append(thick_width)

    # curved edges
    curved_edges_weight = []
    curved_edges_width = []
    for i in range(0,len(curved_edges)):
        curved_edges_weight.append(edge_weights[curved_edges[i]])

        opposite_edge = (curved_edges[i][1],curved_edges[i][0])

        if opposite_edge in curved_edges:
            if edge_weights[curved_edges[i]]> edge_weights[opposite_edge]:
                curved_edges_width.append(thick_width)
            else:
                curved_edges_width.append(thin_width)
        else:

                curved_edges_width.append(thick_width)

    

    # Map normalized values to colors using the coolwarm colormap
    colormap = 'coolwarm'
    cmap = plt.get_cmap(colormap)
    norm = colors.Normalize(vmin=-np.max(np.abs(array_for_node_colors)), vmax=np.max(np.abs(array_for_node_colors)))

    node_colors = [cmap(norm(d)) for d in array_for_node_colors]


    min_node_size = min_node_size
    max_node_size = max_node_size


    node_sizes = min_node_size + (max_node_size - min_node_size) * ( array_for_node_sizes- min(array_for_node_sizes)) / (max(array_for_node_sizes) - min(array_for_node_sizes)) 

    # node_sizes = min_node_size + (max_node_size - min_node_size) * ( array_for_node_sizes- 0) / (2 - 0) 

    nx.draw_networkx_nodes(G, pos, ax = ax, node_color=node_colors, node_size=node_sizes )
    nx.draw_networkx_labels(G, pos,ax = ax, labels=node_labels,  font_size=node_label_font_size, font_weight="bold")

    plt.box(False)


    arc_rad = arc_rad
    nx.draw_networkx_edges(G, pos,ax = ax, edgelist=curved_edges, width = curved_edges_width,  arrows = True, connectionstyle=f'arc3, rad = {arc_rad}', node_size=node_sizes, arrowsize= arrow_size, edge_color = 'black')
    nx.draw_networkx_edges(G, pos,ax = ax, edgelist=straight_edges,width = straight_edges_width , arrows = True, node_size = node_sizes, arrowsize= arrow_size, edge_color = 'black')


    curved_edge_labels = {edge: edge_weights[edge] for edge in curved_edges}
    straight_edge_labels = {edge: edge_weights[edge] for edge in straight_edges}

    my_draw_networkx_edge_labels(G, pos, edge_labels=curved_edge_labels,rad = arc_rad, font_size=edge_label_font_size,font_color="k",font_family="sans-serif",font_weight="bold")
    my_draw_networkx_edge_labels(G, pos, edge_labels=straight_edge_labels,rotate=False,  font_size=edge_label_font_size,font_color="k",font_family="sans-serif",font_weight="bold")



def plot_graph_from_matrix_v2(matrix ,node_color, node_size, labels, ax, min_node_size= 1000, max_node_size= 10000, thick_width = 8, thin_width = 3 , edge_label_font_size = 20, node_label_font_size= 16, arc_rad= 0.07, arrow_size= 26):


    G = nx.DiGraph()

    # Add nodes to the graph with actual labels from net_labels_visualization
    node_labels = {i: label for i, label in enumerate(labels)}
    G.add_nodes_from(node_labels.keys())

    # Add edges with weights from the matrix
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] != 0:
                G.add_edge(i, j, weight=matrix[i][j])

    # pos = nx.spring_layout(G)
    pos = nx.kamada_kawai_layout(G)
    # pos = nx.shell_layout(G)
    # pos = nx.spectral_layout(G)


    ##
    # {0: 'Vis',
    #  1: 'Som-Mot',
    #  2: 'Dors-Att',
    #  3: 'Vent-Att',
    #  4: 'Lim',
    #  5: 'Cont',
    #  6: 'Def'}
    # pos[0] = pos[0] + [0.5,0.5]
    # pos[1] = pos[1] + [0.0, 0.3]
    # pos[2] = pos[2] + [0.1,0]
    # pos[3] = pos[3] + [0.9,0]
    # pos[4] = pos[4]+[-0.5,0]
    # pos[5] = pos[5] + [0.5, +0.2]
    # pos[6] = pos[6]+ [1.5,0]

    # Extracting edge weights
    curved_edges = [edge for edge in G.edges() if reversed(edge) in G.edges()]
    straight_edges = list(set(G.edges()) - set(curved_edges))

    edge_weights  = nx.get_edge_attributes(G,'weight')

    thick_width = thick_width
    thin_width = thin_width

    # straight edges
    straight_edges_weight=[]
    straight_edges_width = []
    for i in range(0,len(straight_edges)):
        straight_edges_weight.append(edge_weights[straight_edges[i]])
        straight_edges_width.append(thick_width)

    # curved edges
    curved_edges_weight = []
    curved_edges_width = []
    for i in range(0,len(curved_edges)):
        curved_edges_weight.append(edge_weights[curved_edges[i]])

        opposite_edge = (curved_edges[i][1],curved_edges[i][0])

        if opposite_edge in curved_edges:
            if edge_weights[curved_edges[i]]> edge_weights[opposite_edge]:
                curved_edges_width.append(thick_width)
            else:
                curved_edges_width.append(thin_width)
        else:

                curved_edges_width.append(thick_width)

    

    nx.draw_networkx_nodes(G, pos, ax = ax, node_color=node_color, node_size=node_size )
    nx.draw_networkx_labels(G, pos,ax = ax, labels=node_labels,  font_size=node_label_font_size, font_weight="bold")

    plt.box(False)

    arc_rad = arc_rad
    nx.draw_networkx_edges(G, pos,ax = ax, edgelist=curved_edges, width = curved_edges_width,  arrows = True, connectionstyle=f'arc3, rad = {arc_rad}', node_size=node_size, arrowsize= arrow_size, edge_color = 'black')
    nx.draw_networkx_edges(G, pos,ax = ax, edgelist=straight_edges,width = straight_edges_width , arrows = True, node_size = node_size, arrowsize= arrow_size, edge_color = 'black')


    curved_edge_labels = {edge: edge_weights[edge] for edge in curved_edges}
    straight_edge_labels = {edge: edge_weights[edge] for edge in straight_edges}

    my_draw_networkx_edge_labels(G, pos, edge_labels=curved_edge_labels,rad = arc_rad, font_size=edge_label_font_size,font_color="k",font_family="sans-serif",font_weight="bold")
    my_draw_networkx_edge_labels(G, pos, edge_labels=straight_edge_labels,rotate=False,  font_size=edge_label_font_size,font_color="k",font_family="sans-serif",font_weight="bold")





def plot_graph_from_matrix_left_right_seperated(matrix ,array_for_node_colors, array_for_node_sizes, labels, ax, min_node_size= 1000, max_node_size= 10000, thick_width = 8, thin_width = 3 , edge_label_font_size = 20, node_label_font_size= 16, arc_rad= 0.07, arrow_size= 26):


    G = nx.DiGraph()

    # Add nodes to the graph with actual labels from net_labels_visualization
    node_labels = {i: label for i, label in enumerate(labels)}
    G.add_nodes_from(node_labels.keys())

    # Add edges with weights from the matrix
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] != 0:
                G.add_edge(i, j, weight=matrix[i][j])

    # pos = nx.spring_layout(G)
    pos = nx.kamada_kawai_layout(G)
    # pos = nx.circular_layout(G)
    # pos = nx.shell_layout(G)
    # pos = nx.spectral_layout(G)

    # communities = nx.community.greedy_modularity_communities(G)
    # communities = nx.community.girvan_newman(G)
    # pos = {}
    # for i, community in enumerate(communities):
    #     for j, node in enumerate(community):
    #         pos[node] = (i, j)

    ##
    # {0: 'Vis',
    #  1: 'Som-Mot',
    #  2: 'Dors-Att',
    #  3: 'Vent-Att',
    #  4: 'Lim',
    #  5: 'Cont',
    #  6: 'Def'}

    # pos[5] = [-0.5,0.5]    # left cont
    # pos[13] = [0.5,0.5]    # right cont

    # pos[2] = [-0.3, -0.5]    # left Dors-Att
    # pos[10] = [0.7, -0.5]    # right Dors-Att
    
    # pos[3] = [-0.1,-1.5]         # left vent-att
    # pos[11] = [0.9,-1.5]        # right vent-att

    # pos[6] = [0.1,-2.5]       # left Def
    # pos[14] = [1.1, -2.5]        # right Def

    # pos[1] =  [0.3,-3.5]              # left Som-Mot
    # pos[9] = [1.3,-3.5]                  # Right Som-Mot
    
    # pos[4] = [0.5, -4.5]           # left Lim
    # pos[12] = [1.5,-4.5]           # Right Lim

    # pos[0] = [0.7,-5.5]                # left Vis
    # pos[8] = [1.7,-5.5]               # right Vis

    # pos[7] = [0.9, -6.5]            # left sub
    # pos[15] = [1.9,-6.5]                # right sub

    # Extracting edge weights
    curved_edges = [edge for edge in G.edges() if reversed(edge) in G.edges()]
    straight_edges = list(set(G.edges()) - set(curved_edges))

    edge_weights  = nx.get_edge_attributes(G,'weight')

    thick_width = thick_width
    thin_width = thin_width

    # straight edges
    straight_edges_weight=[]
    straight_edges_width = []
    for i in range(0,len(straight_edges)):
        straight_edges_weight.append(edge_weights[straight_edges[i]])
        straight_edges_width.append(thick_width)

    # curved edges
    curved_edges_weight = []
    curved_edges_width = []
    for i in range(0,len(curved_edges)):
        curved_edges_weight.append(edge_weights[curved_edges[i]])

        opposite_edge = (curved_edges[i][1],curved_edges[i][0])

        if opposite_edge in curved_edges:
            if edge_weights[curved_edges[i]]> edge_weights[opposite_edge]:
                curved_edges_width.append(thick_width)
            else:
                curved_edges_width.append(thin_width)
        else:

                curved_edges_width.append(thick_width)



    # Map normalized values to colors using the coolwarm colormap
    colormap = 'coolwarm'
    cmap = plt.get_cmap(colormap)
    norm = colors.Normalize(vmin=-np.max(np.abs(array_for_node_colors)), vmax=np.max(np.abs(array_for_node_colors)))

    node_colors = [cmap(norm(d)) for d in array_for_node_colors]

    min_node_size = min_node_size
    max_node_size = max_node_size

    node_sizes = min_node_size + (max_node_size - min_node_size) * ( array_for_node_sizes- min(array_for_node_sizes)) / (max(array_for_node_sizes) - min(array_for_node_sizes)) 

    nx.draw_networkx_nodes(G, pos, ax = ax, node_color=node_colors, node_size=node_sizes )
    nx.draw_networkx_labels(G, pos,ax = ax, labels=node_labels,  font_size=node_label_font_size, font_weight="bold")

    plt.box(False)

    arc_rad = arc_rad
    nx.draw_networkx_edges(G, pos,ax = ax, edgelist=curved_edges, width = curved_edges_width,  arrows = True, connectionstyle=f'arc3, rad = {arc_rad}', node_size=node_sizes, arrowsize= arrow_size, edge_color = 'black')
    nx.draw_networkx_edges(G, pos,ax = ax, edgelist=straight_edges,width = straight_edges_width , arrows = True, node_size = node_sizes, arrowsize= arrow_size, edge_color = 'black')

    curved_edge_labels = {edge: edge_weights[edge] for edge in curved_edges}
    straight_edge_labels = {edge: edge_weights[edge] for edge in straight_edges}

    my_draw_networkx_edge_labels(G, pos, edge_labels=curved_edge_labels,rad = arc_rad, font_size=edge_label_font_size,font_color="k",font_family="sans-serif",font_weight="bold")
    my_draw_networkx_edge_labels(G, pos, edge_labels=straight_edge_labels,rotate=False,  font_size=edge_label_font_size,font_color="k",font_family="sans-serif",font_weight="bold")

def cosine_similarity(matrix_a, matrix_b):
    dot_product = np.sum(matrix_a * matrix_b)
    norm_a = np.sqrt(np.sum(matrix_a**2))
    norm_b = np.sqrt(np.sum(matrix_b**2))
    similarity = dot_product / (norm_a * norm_b)
    return similarity


def compute_average_score_change(score_in_time):

    score_part1 = score_in_time[0:15]
    score_part2 = score_in_time[15:30]

    trial_idx = np.arange(1,16)
    part1_spearman_corr, part1_spearman_pval = scipy.stats.spearmanr(trial_idx, score_part1, nan_policy='propagate')
    part2_spearman_corr, part2_spearman_pval = scipy.stats.spearmanr(trial_idx, score_part2, nan_policy='propagate')
    
    return np.nanmean([part1_spearman_corr, part2_spearman_corr])


def create_weighted_digraph_from_matrix(weighted_matrix):
    num_nodes = weighted_matrix.shape[0]
    G = nx.DiGraph()
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            if weighted_matrix[i, j] != 0:
                G.add_edge(i, j, weight=weighted_matrix[i, j])
    
    return G



def local_efficiency(G):
    efficiency = {}
    for node in G:
        neighbors = list(G.successors(node)) + list(G.predecessors(node))
        subgraph = G.subgraph(neighbors).copy()
        if len(subgraph) > 1:
            inv_shortest_paths = []
            for source in subgraph:
                lengths = nx.single_source_dijkstra_path_length(subgraph, source, weight='weight')
                inv_shortest_paths.extend(1 / length for length in lengths.values() if length > 0)
            if inv_shortest_paths:
                efficiency[node] = sum(inv_shortest_paths) / (len(subgraph) * (len(subgraph) - 1))
            else:
                efficiency[node] = 0.0
        else:
            efficiency[node] = 0.0
    return efficiency