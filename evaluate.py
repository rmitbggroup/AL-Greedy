import torch
import copy
import os
import argparse
import json
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from utils import accuracy, load_data, sparse_mx_to_torch_sparse_tensor, aug_normalized_adjacency
from graphConvolution import *
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help='input dataset')
parser.add_argument('--B', type=int, default=None, help='the labelling budget')
parser.add_argument('--sim_metric', type=str, default='COSINE', help='the embedding similarity metric: ED or COSINE')
parser.add_argument('--t', type=float, default=0.9999, help='the target similarity')
parser.add_argument('--k', type=int, default=1, help='the number of aggregation iteration')
parser.add_argument('--sample_size', type=int, default=0, help='preprocessing sample size, 0 means no sampling')
args = parser.parse_args()


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid, bias=True)
        self.gc2 = GraphConvolution(nhid, nclass, bias=True)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


def train(epoch, model, record):
    model.train()
    optimizer.zero_grad()
    output = model(features_GCN, adj)
    loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    model.eval()
    output = model(features_GCN, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    record[acc_val.item()] = acc_test.item()


# read dataset
adj, features, labels, idx_val, idx_test = load_data(dataset=args.dataset)
features_GCN = copy.deepcopy(features)
features_GCN = torch.FloatTensor(features_GCN).cpu()
adj = aug_normalized_adjacency(adj)
adj = sparse_mx_to_torch_sparse_tensor(adj).float().cpu()
labels = labels.cpu()
idx_val = torch.LongTensor(idx_val).cpu()
idx_test = torch.LongTensor(idx_test).cpu()


# read selected seeds
seeds_dir = "output_seeds"
method = "greedyET_{}_t{}_k{}_ss{}_b{}".format(args.sim_metric, args.t, args.k, args.sample_size, args.B)
with open("{}/{}_{}.json".format(seeds_dir, args.dataset, method), 'r') as f:
    idx_selected_seeds = json.load(f)


# seed evaluation
idx_train = torch.LongTensor(idx_selected_seeds).cpu()
record = {}
model = GCN(nfeat=features_GCN.shape[1], nhid=128, nclass=labels.max().item()+1, dropout=0.85)
model.cpu()
optimizer = optim.Adam(model.parameters(), lr=0.05, weight_decay=5e-4)

for epoch in range(400):
    train(epoch, model, record)

bit_list = sorted(record.keys())
bit_list.reverse()
test_acc = record[bit_list[0]]
print("Test accuracy: {}".format(test_acc))
