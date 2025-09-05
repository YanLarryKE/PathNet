from .utils import *
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch
from torch.autograd import Variable
import torch.utils.data as Data
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import random
import warnings
import os, gc
import numpy as np
import math
import torch.nn.init as init
from scipy.sparse import lil_matrix
from IPython.display import clear_output


def truncated_normal_(tensor,mean=0,std=0.09):
    with torch.no_grad():
        size = tensor.shape
        tmp = tensor.new_empty(size+(4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)
        return tensor

class GeneExpressionModel(nn.Module):
    def __init__(self, partition_mtx_dict, residual_connection_dict, num_hidden_layer_neuron_list, keep_prob, n_genes):
        super(GeneExpressionModel, self).__init__()
        layer0 = nn.Sequential()
        layer0.add_module('e1', MiRNAAdjustment(partition_mtx_dict["p%d" % 0], n_genes))
        self.layer0 = layer0

        layer1 = nn.Sequential()
        for i in range(2, len(partition_mtx_dict) + 1):
            mtx = partition_mtx_dict["p%d" % i]  # the mask matrix
            residual_connect = residual_connection_dict["p%d" % i]
            layer1.add_module('f'+ str(i), SparseLinear(mtx, residual_connect))
        self.layer1 = layer1
        
        layer2 = nn.Sequential()
        num_hidden_layer_neuron_list = [mtx.shape[1]] + num_hidden_layer_neuron_list + [2]
        for j in range(1, len(num_hidden_layer_neuron_list)-1):
            layer2.add_module('h'+str(j), nn.Linear(num_hidden_layer_neuron_list[j-1], num_hidden_layer_neuron_list[j]))
            layer2.add_module('h_relu'+str(j), nn.ReLU(True))
            layer2.add_module('h_drop'+str(j), nn.Dropout(p=keep_prob))
        j = len(num_hidden_layer_neuron_list)-2
        layer2.add_module('h'+str(j+1), nn.Linear(num_hidden_layer_neuron_list[j], num_hidden_layer_neuron_list[j+1]))
        layer2.add_module('h'+str(j+2), nn.Softmax(dim=-1))
        self.layer2 = layer2
        
    def forward(self,input):
        out = self.layer0(input)
        out = self.layer1(out)
        out = self.layer2(out)
        return out

class MiRNAAdjustment(nn.Module):
    def __init__(self, reg_matrix: lil_matrix, n_genes):
        super().__init__()
        self.mask = nn.Parameter(torch.tensor(reg_matrix.toarray(), dtype=torch.bool), requires_grad=False)  # 形状: (n_genes, n_mirnas)
        self.n_genes = n_genes
        self.in_features = reg_matrix.shape[0]
        self.out_features = reg_matrix.shape[1]
        self.weight_0 = nn.Parameter(torch.Tensor(reg_matrix.shape[0], reg_matrix.shape[1]))
        self.bias_0 = nn.Parameter(torch.zeros(reg_matrix.shape[0], reg_matrix.shape[1]))
        self.weight_1 = nn.Parameter(torch.Tensor(reg_matrix.shape[0], reg_matrix.shape[1]))
        self.bias_1 = nn.Parameter(torch.zeros(reg_matrix.shape[0], reg_matrix.shape[1]))
        
        self._init_parameters()

        self.weight_0.data *= self.mask.float()
        self.bias_0.data *= self.mask.float()
        self.weight_1.data *= self.mask.float()
        self.bias_1.data *= self.mask.float()

    def _init_parameters(self):
        # N(0, 0.01)
        self.weight_0 = nn.Parameter(torch.empty((self.in_features, self.out_features)))
        self.weight_1 = nn.Parameter(torch.empty((self.in_features, self.out_features)))
        nn.init.normal_(self.weight_0, mean=0, std=0.01)
        nn.init.normal_(self.weight_1, mean=0, std=0.01)

    def forward(self, input:torch.Tensor):
        # gene_expr: (batch_size, n_genes)
        # mirna_expr: (batch_size, n_mirnas)
        genes = input[:, :self.n_genes]
        mirnas = input[:, self.n_genes:]
        mirnas_expanded = mirnas.unsqueeze(-1)           # [B, D, 1]
        weight0_expanded = self.weight_0.unsqueeze(0)
        weight1_expanded = self.weight_1.unsqueeze(0)      # [1, D, K]
        # print(mirnas_expanded.shape, weight0_expanded.shape)
        # [B, D, 1] * [1, D, K] → [B, D, K]
        out = mirnas_expanded * (weight0_expanded * self.mask.float()) + self.bias_0 * self.mask.float()
        out = F.tanh(out)

        # out = out * (self.weight_1 * self.influence.float()) + self.bias_0 * self.influence.float()
        
        out = out * (weight1_expanded * self.mask.float())+ self.bias_1 * self.mask.float()
        out = torch.sum(input=out, dim=1)
        
        # do not influence the nodes that are not connected
        # Residual connection
        out += genes
        return out


class SparseLinear(nn.Module):
    """
    Define our linear connection layer which enabled sparse connection
    """
    def __init__(self, m, residual_connection=None, bias=True, re=False):
        super(SparseLinear, self).__init__()
        self.in_features = m.shape[0]
        self.out_features = m.shape[1]

        self.weight_0 = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.weight_1 = Parameter(torch.Tensor(self.in_features, self.out_features))

        self.residual_connection = residual_connection
        self.re = re
        if residual_connection is not None:
            self.residual_connection = Parameter(torch.tensor(residual_connection), requires_grad=False)
        if bias:
            self.bias_0 = Parameter(torch.zeros(self.in_features, self.out_features))
            self.bias_1 = Parameter(torch.zeros(self.in_features, self.out_features))
        else:
            self.register_parameter('bias_0', None)
            self.register_parameter('bias_1', None)
        
        self.reset_parameters()
        self.influence = Parameter(
            torch.from_numpy((np.sum(m, axis=0) > 0).astype(int)).bool(),
            requires_grad=False
        )
        # register mask
        self.mask = Parameter(
            torch.zeros(self.in_features, self.out_features).bool(),
            requires_grad=False
        )

        # modify residual connection
        indices_mask = [np.where(m==1)[0].tolist(), np.where(m==1)[1].tolist()]
        
        self.mask.data[indices_mask] = 1

        self.weight_0.data *= self.mask.float()
        self.bias_0.data *= self.mask.float()
        self.weight_1.data *= self.mask.float()
        self.bias_1.data *= self.mask.float()

    def reset_parameters(self):
        self.weight_0 = truncated_normal_(self.weight_0, mean = 0, std = 0.1)
        self.weight_1 = truncated_normal_(self.weight_1, mean = 0, std = 0.1)
        if self.bias_0 is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_0)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias_0, -bound, bound)
            init.uniform_(self.bias_1, -bound, bound)

    def forward(self, input):
        # print(input.size(), self.bias_0.size(), self.influence.size())
        # Maybe later next week, I shall try if residual as one and one only is enough...
        # out = input @ (self.mask.float() * self.weight_0) + self.bias_0 * self.influence.float()
        
        input_expanded = input.unsqueeze(-1)        # [B, D, 1]
        weight0_expanded = self.weight_0.unsqueeze(0)
        weight1_expanded = self.weight_1.unsqueeze(0)      # [1, D, K]
        
        # [B, D, 1] * [1, D, K] → [B, D, K]
        out = input_expanded * (weight0_expanded * self.mask.float()) + self.bias_0 * self.mask.float()

        if not self.re:
            out = F.tanh(out)
        else:
            out = F.relu(out)

        # out = out * (self.weight_1 * self.influence.float()) + self.bias_0 * self.influence.float()
        
        out = out * (weight1_expanded * self.mask.float())+ self.bias_1 * self.mask.float()
        out = torch.sum(input=out, dim=1)
        
        # do not influence the nodes that are not connected
        # Residual connection
        if self.residual_connection is None:
            return out
        else:
            out = input[:, self.residual_connection] + out
            return out
        

class meta_Net(nn.Module):
    """
    The network structure
    """
    def __init__(self,partition_mtx_dict, residual_connection_dict, num_hidden_layer_neuron_list, keep_prob):
        super(meta_Net,self).__init__()

        layer1 = nn.Sequential()
        feature_meta_mtx = partition_mtx_dict["p%d" % 0]
        layer1.add_module('f0', SparseLinear(feature_meta_mtx))
        # meta_2_meta_mtx = partition_mtx_dict["p%d" % 1]
        # layer1.add_module('f1', SparseLinear(meta_2_meta_mtx, np.arange(0, meta_2_meta_mtx.shape[0], step=1, dtype=np.int32)))
        for i in range(2, len(partition_mtx_dict) + 1):
            mtx = partition_mtx_dict["p%d" % i]  # the mask matrix
            residual_connect = residual_connection_dict["p%d" % i]
            if i == len(partition_mtx_dict):
                layer1.add_module('f'+ str(i), SparseLinear(mtx, residual_connect, re=True))
            else:
                layer1.add_module('f'+ str(i), SparseLinear(mtx, residual_connect))
                
        self.layer1 = layer1
        layer2 = nn.Sequential()
        num_hidden_layer_neuron_list = [mtx.shape[1]] + num_hidden_layer_neuron_list + [2]
        for j in range(1, len(num_hidden_layer_neuron_list)-1):
            layer2.add_module('h'+str(j), nn.Linear(num_hidden_layer_neuron_list[j-1], num_hidden_layer_neuron_list[j]))
            layer2.add_module('h_relu'+str(j), nn.ReLU(True))
            layer2.add_module('h_drop'+str(j), nn.Dropout(p=keep_prob))
        j = len(num_hidden_layer_neuron_list)-2
        layer2.add_module('h'+str(j+1), nn.Linear(num_hidden_layer_neuron_list[j], num_hidden_layer_neuron_list[j+1]))
        layer2.add_module('h'+str(j+2), nn.Softmax(dim=-1))
        self.layer2 = layer2
        
    def forward(self,input):
        out = self.layer1(input)
        out = self.layer2(out)
        return out

class pathNet(nn.Module):
    """
    The network structure
    """
    def __init__(self,partition_mtx_dict, residual_connection_dict, num_hidden_layer_neuron_list, keep_prob):
        super(pathNet,self).__init__()

        layer1 = nn.Sequential()
        # meta_2_meta_mtx = partition_mtx_dict["p%d" % 1]
        # layer1.add_module('f1', SparseLinear(meta_2_meta_mtx, np.arange(0, meta_2_meta_mtx.shape[0], step=1, dtype=np.int32)))
        for i in range(2, len(partition_mtx_dict) + 1):
            mtx = partition_mtx_dict["p%d" % i]  # the mask matrix
            residual_connect = residual_connection_dict["p%d" % i]
            if i == len(partition_mtx_dict):
                layer1.add_module('f'+ str(i), SparseLinear(mtx, residual_connect))
            else:
                layer1.add_module('f'+ str(i), SparseLinear(mtx, residual_connect))
                
        self.layer1 = layer1
        layer2 = nn.Sequential()
        num_hidden_layer_neuron_list = [mtx.shape[1]] + num_hidden_layer_neuron_list + [2]
        for j in range(1, len(num_hidden_layer_neuron_list)-1):
            layer2.add_module('h'+str(j), nn.Linear(num_hidden_layer_neuron_list[j-1], num_hidden_layer_neuron_list[j]))
            layer2.add_module('h_relu'+str(j), nn.ReLU(True))
            layer2.add_module('h_drop'+str(j), nn.Dropout(p=keep_prob))
        j = len(num_hidden_layer_neuron_list)-2
        layer2.add_module('h'+str(j+1), nn.Linear(num_hidden_layer_neuron_list[j], num_hidden_layer_neuron_list[j+1]))
        layer2.add_module('h'+str(j+2), nn.Softmax(dim=-1))
        self.layer2 = layer2
        
    def forward(self,input):
        out = self.layer1(input)
        out = self.layer2(out)
        return out

# Define the function for train the model 
def sparse_nn(expression: np.ndarray, labels, target_feature, feature_meta, knowledge_graph, maximum_step=3, meta=True, mirna_expr=None,
              num_hidden_layer_neuron_list=[20], drop_out=0.3, random_seed=10, batch_size=32, lr=0.001, weight_decay=0,
              num_epoch=100, test_mode=False, log=True, abla_graph=False):
    
    # set random seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    # set the partition to be self-connected
    # partition = np.zeros((feature_meta.shape[1], feature_meta.shape[1]))
    # np.fill_diagonal(partition, 1)

    # For sparsity control
    # sparsify_hidden_layer_size_dict = getLayerSizeList(feature_meta.shape[1], len(target_feature), sparsify_coefficient)
    
    # degree_dict = getNodeDegreeDict(partition)
    partition_mtx_dict, residual_connection_dict, connectionList = getPartitionMatricesList(target_feature, knowledge_graph, maximum_step, abla_graph=abla_graph)
    partition_mtx_dict["p0"] = feature_meta 
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.manual_seed_all(random_seed)
    else:
        device = torch.device('cpu')

    if test_mode == True:
        net=pathNet(partition_mtx_dict, residual_connection_dict, num_hidden_layer_neuron_list, drop_out).to(device)
        x_train, x_val, y_train, y_val = train_test_split(
            expression, labels, test_size = 0.3, random_state = 1)
    elif meta:
        net = meta_Net(partition_mtx_dict, residual_connection_dict, num_hidden_layer_neuron_list, drop_out).to(device)
        x_train, x_val, y_train, y_val = train_test_split(
            expression, labels, test_size = 0.3, random_state = 1)
    else:
        n_genes = expression.shape[1]
        net = GeneExpressionModel(partition_mtx_dict, residual_connection_dict, num_hidden_layer_neuron_list, drop_out, n_genes).to(device)
        x_train, x_val, y_train, y_val = train_test_split(
            np.concatenate([expression, mirna_expr], axis=1), labels, test_size = 0.3, random_state = 1)
    warnings.filterwarnings("ignore", category=UserWarning)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay = weight_decay)

    loss_func = torch.nn.CrossEntropyLoss()
    # split train and test set

    x_train_tensor = torch.from_numpy(x_train).type(torch.FloatTensor)
    y_train_tensor = torch.from_numpy(y_train).type(torch.LongTensor)
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)

    x_val_tensor = torch.from_numpy(x_val).type(torch.FloatTensor)
    y_val_tensor = torch.from_numpy(y_val).type(torch.LongTensor)
    val_dataset = TensorDataset(x_val_tensor, y_val_tensor)


    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )


    acc_train = []
    acc_val = []
        
    # save the result in a new folder
    if not os.path.exists("res"):
        os.mkdir("res")

    for epoch in range(1, num_epoch + 1): 
        net.train()
        for step, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            prediction = net(x_batch)
            loss = loss_func(prediction, y_batch)
            loss.backward()
            optimizer.step()

        net.eval()
        train_correct = 0
        train_total = 0
        with torch.no_grad():
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = net(x_batch)
                _, predicted = torch.max(outputs.data, 1)
                train_total += y_batch.size(0)
                train_correct += (predicted == y_batch).sum().item()
        train_acc = train_correct / train_total
        acc_train.append(train_acc)

        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = net(x_batch)
                _, predicted = torch.max(outputs.data, 1)
                val_total += y_batch.size(0)
                val_correct += (predicted == y_batch).sum().item()
        val_acc = val_correct / val_total
        acc_val.append(val_acc)
        if log:
            print("epoch: %d, test acc: %f, train acc: %f" % (epoch, val_acc, train_acc))
            
            
    # Downstream analysis
    params = {k:v.cpu().numpy() for k,v in net.state_dict().items()}

    del net, optimizer

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    import gc
    gc.collect()

    return params, partition_mtx_dict, connectionList, acc_val