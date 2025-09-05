import pandas as pd
import numpy as np
import igraph as ig
import math
import os
import zipfile
import random

# Windows
# package_dir = "E:\\SPARSENN\\Modified_model\\PathNet"

# Macos
# package_dir = "/Users/watertank/Desktop/SPARSENN/Modified_model/PathNet"

# Server
package_dir = "/home/keyan/phynn/knowledge_informed_ml/PathNet"
# package_dir = os.path.abspath(os.path.dirname(__file__))


zip_path = os.path.join(package_dir, 'data', 'kegg.txt.zip')
output_dir = "data/"

# Check if the zip file exists
if os.path.exists(zip_path):
    # Open the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Extract all the files
        zip_ref.extractall(output_dir)
    os.remove(zip_path)

#################### Function for data pre-processing ####################

# define a function to remove rows with more than 75% of zeros
def remove_rows(dat, data, thres = 0.75):
    """
    dat contains mz and time and other info
    data has shape (n_features, n_samples)
    """
    rowsum = np.sum(data==0,1)
    new_dat = dat.iloc[np.where((rowsum < thres * data.shape[1])==True)[0],:]
    return new_dat

# define a function to find the potential KEGGID for each feature
def find_keggid(dat, kegg_sub, this_adductlist, match_tol_ppm=9):
    """
    dat contains mz and time and other info
    kegg_sub is a subset of kegg that contains only the compounds that are in the graph
    """
    this_kegg = kegg_sub[kegg_sub['Adduct'].isin(this_adductlist)]

    dic = {}
    for i in range(dat.shape[0]):
        # If the mz matches that in the database, we claim that this is a match.
        idx = list(np.where(np.abs(this_kegg['mz']-dat['mz'].iloc[i])/(dat['mz'].iloc[i])<=match_tol_ppm/1e6)[0])
        
        # Get the corresponding KEGGID.
        dic[dat.index[i]] = list(this_kegg['KEGGID'].values[idx])
    return dic

# define a function to get the feature-metabolite matching matrix, adj matrix and feature data
def get_data(dic, new_dat, g):
    # Get all the features and the metabolites.
    features = [key for key, value in dic.items() if value!=[]]
    
    metabolites = np.unique(sum([value for key, value in dic.items() if value!=[]], []))

    # get feature data  
    data_anno_new = new_dat.loc[features,:]
    print("The shape of data:", data_anno_new.shape)

    # get feature-metabolite matching matrix
    matching = np.zeros([len(features), len(metabolites)])
    for ix,i in enumerate(features):
        idx = np.where(np.in1d(metabolites, dic[i]))[0]
        matching[ix, idx] = 1

    print("The shape of feature-metabolites matching:", matching.shape)

    # get adjacency matrix of metabolites
    subgraph = ig.Graph()

    # add the vertices from order_list to the subgraph
    for v in metabolites:
        if v in g.vs["name"]:
            subgraph.add_vertex(v)

    # add the edges that connect the vertices in the subgraph
    for e in g.es:
        source = e.source
        target = e.target
        if g.vs['name'][source] in metabolites and g.vs['name'][target] in metabolites:
            subgraph.add_edge(g.vs['name'][source], g.vs['name'][target])

    # The original graph is multiconnected, so do the subgraph.
    subgraph.simplify()

    # # Not quite clear why didn't use the original adj matrix but a new one. TODO: check this
    # adj_new = np.array(subgraph.get_adjacency().data)
    # g_sub = ig.Graph.Adjacency((adj_new > 0).tolist(), mode = "undirected")

    # adj_matrix = np.array(g_sub.get_adjacency().data)
    print("The shape of metabolic network:", (matching.shape[1], matching.shape[1]))
    
    return(data_anno_new, matching, subgraph, metabolites)


def data_preprocessing(pos=None, neg=None, 
                       pos_adductlist=["M+H","M+NH4","M+Na","M+ACN+H","M+ACN+Na","M+2ACN+H","2M+H","2M+Na","2M+ACN+H"], 
                       neg_adductlist = ["M-H", "M-2H", "M-2H+Na", "M-2H+K", "M-2H+NH4", "M-H2O-H", "M-H+Cl", "M+Cl", "M+2Cl"], 
                       idx_feature = 4, match_tol_ppm=5, zero_threshold=0.75, log_transform=True, scale=1000):

    # Load data
    g = ig.Graph.Read_GraphML(os.path.join(package_dir, 'data', 'graph.graphhml'))
    all_compound = list(g.vs["name"])
    
    # Filter out the lines in the DB where **(kegg['mz']-kegg['AdductMass']==kegg['MonoisotopicMass'])**.
    kegg = pd.read_csv(os.path.join(package_dir, 'data', 'kegg.txt'), sep='\t')
    kegg['r'] = (kegg['mz']-kegg['AdductMass']==kegg['MonoisotopicMass'])
    kegg = kegg[kegg['r']==True]
    kegg_sub = kegg[kegg['KEGGID'].isin(all_compound)]

    if pos is not None:
        pos.columns.values[0] = 'mz'
        pos.columns.values[1] = 'time'
        pos.columns = pos.columns.str.replace('pos', '')
        pos.index = ['pos.' + str(i) for i in pos.index]

    if neg is not None:
        neg.columns.values[0] = 'mz'
        neg.columns.values[1] = 'time'
        neg.columns = neg.columns.str.replace('neg', '')
        neg.index = ['neg.' + str(i) for i in neg.index]

    # concatenate the two dataframes
    if pos is not None and neg is not None:
        dat = pd.concat([pos, neg], axis=0)
    elif pos is not None:
        dat = pos
    elif neg is not None:
        dat = neg
        
    # leave out those with very low expression rate
    new_dat = remove_rows(dat, dat.iloc[:,idx_feature:], thres = zero_threshold)

    # select only the compounds that are in the graph
    if pos is not None and neg is not None:
        dic_pos = find_keggid(new_dat.loc[new_dat.index.str.contains('pos')], kegg_sub, pos_adductlist)
        dic_neg = find_keggid(new_dat.loc[new_dat.index.str.contains('neg')], kegg_sub, neg_adductlist)

        dic = {**dic_pos, **dic_neg}
    elif pos is not None:
        dic = find_keggid(new_dat.loc[new_dat.index.str.contains('pos')], kegg_sub, pos_adductlist)
    elif neg is not None:
        dic = find_keggid(new_dat.loc[new_dat.index.str.contains('neg')], kegg_sub, neg_adductlist)

    data_annos, matchings, sub_graph, metabolites = get_data(dic, new_dat, g)
    
    if log_transform:
        data_annos.iloc[:,idx_feature:] = np.log(data_annos.iloc[:,idx_feature:]+1)
        
    if scale:
        expression = data_annos.iloc[:,idx_feature:].T
        
        mean_centered = expression - np.mean(expression, axis=0)
        
        std_dev = np.std(expression, axis=0)
        
        std_dev[std_dev == 0] = 1.0
        expression = mean_centered / np.sqrt(std_dev)
        
        data_annos.iloc[:,idx_feature:] = expression.T

    return(data_annos, matchings, sub_graph, metabolites, dic)

###################### Function for main model ######################

def getLayerSizeList(n_meta, final_layer_size, maximum_step):
    """
    Obtain the size of each sparse layer
    
    INPUT:
    final_layer_size: the final of sparse layer
    maximum_step: the coefficient of each sparse level
    
    OUTPUT:
    sparsify_hidden_layer_size_dict: a dictionary indicating the sparse layer
    """
    ### New signiture: getLayerSizeList(knowledge_graph, final_layer_feature, maximum_step):
    
    # indices = set([knowledge_graph.vs.find(idx).index for idx in final_layer_feature])

    # # keep track of nodes that we have recorded
    # nodesBeenEncoded = set()
    # nodesBeenEncoded.update(indices)

    # for i in range(maximum_step):
    #     # keep track of all nodes to be operated
    #     roughSet = set()
    #     for node in sorted(list(nodesBeenEncoded)):
    #         # extend all these by one step
    #         neighbours = knowledge_graph.neighbourhood(node, order = 1, mindist = 0)
    #         roughSet.update(neighbours)
    #     # now we find all nodes that are one step away from what have been encoded (maybe not, if so then ::)

    #     # if we cannot spread out this too far
    #     if roughSet == nodesBeenEncoded:
    #         raise ValueError(f"The graph do not have depth of {i+1}, try little maximum depth instead.")
        
    #     #[prevLayer.index(x) for x in nextLayer]
    #     nodesToBeEncoded = sorted(list(roughSet - nodesBeenEncoded))
        
    #     # find where we want to maintain the whole feature
    #     deepNodeList = sorted(list(nodesBeenEncoded))
    #     shallowNodeList = sorted(list(roughSet))
    #     shortcut = [shallowNodeList.index(x) for x in deepNodeList]
    n_layer = math.floor(np.log10(1.0 * final_layer_size / n_meta) / np.log10(maximum_step)) + 1
    
    # dict for number of neurons in each layer
    sparsify_hidden_layer_size_dict = {}

    sparsify_hidden_layer_size_dict['n_hidden_0'] = int(n_meta)

    for i in range(1, n_layer):
        sparsify_hidden_layer_size_dict['n_hidden_%d' % (i)] = int(final_layer_size / (maximum_step) ** (n_layer - i))

    sparsify_hidden_layer_size_dict['n_hidden_%d' % (n_layer)] = final_layer_size

    return sparsify_hidden_layer_size_dict


def getPartitionMatricesList(target_keggids, knowledge_graph: ig.Graph, maximum_step, meta = True, abla_graph=False):
    """
    Obtain the linkage matrix among two sparse layers
    """
    
    partition_mtx_dict = {}
    residual_connection_dic = {}
    if meta:
        connectionList = backwardSelect(target_keggids, knowledge_graph, maximum_depth=maximum_step)
    # The code below adopted a seemingly very **stupid** way of determining the linkage. TODO: rewrite this
    for i in range(2, len(connectionList) + 1):
        nextLayer = connectionList[i - 1].copy()
        prevLayer = connectionList[i - 2].copy()
        temp_partition = np.zeros((len(prevLayer), len(nextLayer)))
        
        encoding_nodes = sorted(list(set(prevLayer) - set(nextLayer)))
        if abla_graph:
            temp_partition = np.random.choice([0, 1], size=temp_partition.shape, p=[0.95, 0.05])
        else: 

    # I think there is better solution... This implementation is stupid.

            for idx, nodex in enumerate(encoding_nodes):

                # find the small connected component
                connections = knowledge_graph.neighborhood(nodex, order=1, mindist=1)

                for idy, nodey in enumerate(nextLayer):
                    # if in the original graph, they are connected,
                    if nodey in connections:
                        temp_partition[prevLayer.index(nodex), idy] = 1

        # Residual connection layer
        residual_location = [prevLayer.index(x) for x in nextLayer]
        
        partition_mtx_dict["p%d" % i] = temp_partition

        residual_connection_dic["p%d" % i] = residual_location

    return partition_mtx_dict, residual_connection_dic, connectionList


## Functions for backward selection.

# The logic here is disastrous... Maybe there is better implementation
def backwardSelect(final_keggids, subgraph: ig.Graph, maximum_depth = 3, with_mirna = False, n_genes = None):
    indices = set([subgraph.vs.find(idx).index for idx in final_keggids])

    # keep track of nodes that we have recorded
    nodesBeenEncoded = set()
    nodesBeenEncoded.update(indices)

    mergedNodeList = [sorted(list(nodesBeenEncoded))]
    for i in range(maximum_depth):
        # keep track of all nodes to be operated
        roughSet = set()
        for node in sorted(list(nodesBeenEncoded)):
            # extend all these by one step
            neighbours = subgraph.neighborhood(node, order = 1, mindist = 0)
            roughSet.update(neighbours)
        # now we find all nodes that are one step away from what have been encoded (maybe not, if so then ::)

        # if we cannot spread out this too far
        if roughSet == nodesBeenEncoded:
            raise ValueError(f"The graph do not have depth of {i+1}, try little maximum depth instead.")
        
        # find where we want to maintain the whole feature
        
        nodesBeenEncoded.update(roughSet)
        deepNodeList = sorted(list(nodesBeenEncoded))
        mergedNodeList.append(sorted(list(deepNodeList)))
    node_count = subgraph.vcount()    
    if len(mergedNodeList[-1]) < node_count:
        node_list = list(range(0, node_count))
        mergedNodeList.append(sorted(list(node_list)))
    mergedNodeList.reverse()
    return mergedNodeList
    # indices = set([subgraph.vs.find(idx).index for idx in final_keggids])
    # # Keep track of all nodes in each layer for sparse connection
    # idsOfConnectedNodesEachLayer = [indices.copy()]

    # # Keep track of all nodes that have been connected
    # idxsHaveBeenConnected = indices.copy()

    # # number of output nodes must equal to number we pre-set
    # assert numberOfNodesList[-1] == len(final_keggids)

    # # Backward selection
    # numberOfNodesList.reverse()
    # numberOfNodesList.remove(len(final_keggids))
    
    # for layerNumber, numberOfEachLayer in enumerate(numberOfNodesList):
    #     currentNumber = len(idxsHaveBeenConnected)
    #     numberOfNodesToBeConnected = numberOfEachLayer - currentNumber
        
    #     # The idxs to be newly connected in this layer
    #     idxsToBeConnected = set()

    #     # We only want those haven't been connected
    #     idxsCanBeConnected = set(np.concatenate([subgraph.neighborhood(idx, order=1, mindist=1) for idx in idxsHaveBeenConnected]).astype(np.int32).flatten().tolist()) \
    #                         - idxsHaveBeenConnected

    #     # if we happen to have more than we want to remove
    #     if len(idxsCanBeConnected) >= numberOfNodesToBeConnected:
    #         idxsCanBeConnected = random.sample(sorted(idxsCanBeConnected), numberOfNodesToBeConnected)
    #         idxsToBeConnected.update(idxsCanBeConnected)

    #         # add all the nodes that have been connected to the list
    #         # idxsToBeConnected.update(idxsHaveBeenConnected)
    #         idxsHaveBeenConnected.update(idxsCanBeConnected)
    #         idsOfConnectedNodesEachLayer.append(sorted(idxsToBeConnected))
    #         continue

    #     # else we have less nodes
    #     elif len(idxsCanBeConnected) < numberOfNodesToBeConnected:
    #         # first we add them all
    #         currentNumber += len(idxsCanBeConnected)
            
    #         idxsToBeConnected.update(idxsCanBeConnected)
    #         idxsHaveBeenConnected.update(idxsCanBeConnected)

    #         # while we don't have enough, we keep sampling until we have all we want
    #         while currentNumber < numberOfEachLayer:

    #             idxsCanBeConnected = set(np.concatenate([subgraph.neighborhood(idx, order=1, mindist=1) for idx in idxsHaveBeenConnected]).astype(np.int32).flatten().tolist()) \
    #                         - idxsHaveBeenConnected
                
    #             # If the connected subgraph is all selected, which is unlikely to happen, we randomly put needed node into the connection...
    #             if len(idxsCanBeConnected) == 0:

    #                 idxsCanBeConnected = set(range(subgraph.vcount())) - idxsHaveBeenConnected
    #                 assert idxsCanBeConnected.isdisjoint(idxsHaveBeenConnected)

    #                 idxsCanBeConnected = random.sample(sorted(idxsCanBeConnected), numberOfEachLayer - currentNumber)

    #                 idxsToBeConnected.update(idxsCanBeConnected)
    #                 idxsHaveBeenConnected.update(idxsCanBeConnected)
    #                 break
                
    #             # If we still need more nodes, we just add them all
    #             elif len(idxsCanBeConnected) < numberOfEachLayer - currentNumber:
    #                 currentNumber += len(idxsCanBeConnected)
    #                 idxsToBeConnected.update(idxsCanBeConnected)
    #                 idxsHaveBeenConnected.update(idxsCanBeConnected)
    #                 continue
                
    #             # When we have more nodes than we need
    #             elif len(idxsCanBeConnected) >= numberOfEachLayer - currentNumber:
    #                 idxsCanBeConnected = random.sample(sorted(idxsCanBeConnected), numberOfEachLayer - currentNumber)
    #                 idxsToBeConnected.update(idxsCanBeConnected)
    #                 idxsHaveBeenConnected.update(idxsCanBeConnected)
    #                 break
                    
    #         # when we have enough, then just go to another layer
    #         # idxsToBeConnected.update(idxsHaveBeenConnected)
    #         idsOfConnectedNodesEachLayer.append(sorted(idxsToBeConnected))


    # mergedNodeList = [sorted(idsOfConnectedNodesEachLayer[0])]
    # for i in range(1, len(idsOfConnectedNodesEachLayer)):
    #     temp = sorted(mergedNodeList[i-1] + list(idsOfConnectedNodesEachLayer[i]))
    #     mergedNodeList.append(temp)
    # mergedNodeList.reverse()
    return mergedNodeList