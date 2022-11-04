import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class SAGE(torch.nn.Module):
    def __init__(self, in_features, hidden_features, n_class, n_layer, dropout):
        super(SAGE, self).__init__()
        self.layer_nums = n_layer
        self.dropout = dropout
        self.nclass = n_class
        
        self.pre = torch.nn.Sequential(torch.nn.Linear(in_features, hidden_features))
        self.post = torch.nn.Sequential(torch.nn.Linear(hidden_features, n_class))

        self.graph_convs = torch.nn.ModuleList()
        for _ in range(self.layer_nums - 1):
            self.graph_convs.append(SAGEConv(hidden_features, hidden_features))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.pre(x)
        for i in range((len(self.graph_convs))):
            x = self.graph_convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.post(x)
        return F.log_softmax(x, dim=1)
    
    def loss(self, pred, label):
        return F.nll_loss(pred, label)
