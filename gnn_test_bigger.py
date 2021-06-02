import os
import time
import math
import json
import random
import numpy as np
import argparse
import torch
import torch.nn as nn

from tqdm import tqdm

from gnn_data import GNN_DATA
from gnn_model import GIN_Net2
from utils import Metrictor_PPI, print_file

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

parser = argparse.ArgumentParser(description='Test Model')
parser.add_argument('--description', default=None, type=str,
                    help='train description')
parser.add_argument('--ppi_path', default=None, type=str,
                    help="ppi path")
parser.add_argument('--pseq_path', default=None, type=str,
                    help="protein sequence path")
parser.add_argument('--vec_path', default=None, type=str,
                    help='protein sequence vector path')
parser.add_argument('--index_path', default=None, type=str,
                    help='cnn_rnn and gnn unified train and valid ppi index')
parser.add_argument('--gnn_model', default=None, type=str,
                    help="gnn trained model")
parser.add_argument('--bigger_ppi_path', default=None, type=str,
                    help="if use bigger ppi")
parser.add_argument('--bigger_pseq_path', default=None, type=str,
                    help="if use bigger ppi")

def test(model, graph, test_mask, device):
    valid_pre_result_list = []
    valid_label_list = []

    model.eval()

    batch_size = 256

    valid_steps = math.ceil(len(test_mask) / batch_size)

    for step in tqdm(range(valid_steps)):
        if step == valid_steps-1:
            valid_edge_id = test_mask[step*batch_size:]
        else:
            valid_edge_id = test_mask[step*batch_size : step*batch_size + batch_size]

        output = model(graph.x, graph.edge_index, valid_edge_id)
        label = graph.edge_attr_1[valid_edge_id]
        label = label.type(torch.FloatTensor).to(device)

        m = nn.Sigmoid()
        pre_result = (m(output) > 0.5).type(torch.FloatTensor).to(device)

        valid_pre_result_list.append(pre_result.cpu().data)
        valid_label_list.append(label.cpu().data)

    valid_pre_result_list = torch.cat(valid_pre_result_list, dim=0)
    valid_label_list = torch.cat(valid_label_list, dim=0)

    metrics = Metrictor_PPI(valid_pre_result_list, valid_label_list)

    metrics.show_result()

    print("Recall: {}, Precision: {}, F1: {}".format(metrics.Recall, metrics.Precision, metrics.F1))

def main():

    args = parser.parse_args()

    ppi_data = GNN_DATA(ppi_path=args.ppi_path, bigger_ppi_path=args.bigger_ppi_path)

    ppi_data.get_feature_origin(pseq_path=args.bigger_pseq_path, vec_path=args.vec_path)

    ppi_data.generate_data()

    graph = ppi_data.data
    temp = graph.edge_index.transpose(0, 1).numpy()
    ppi_list = []

    for edge in temp:
        ppi_list.append(list(edge))

    truth_edge_num = len(ppi_list) // 2
    # fake_edge_num = len(ppi_data.fake_edge) // 2
    fake_edge_num = 0
    
    with open(args.index_path, 'r') as f:
        index_dict = json.load(f)
        f.close()
    graph.train_mask = index_dict['train_index']

    all_mask = [i for i in range(truth_edge_num)]
    graph.val_mask = list(set(all_mask).difference(set(graph.train_mask)))

    print("train gnn, train_num: {}, valid_num: {}".format(len(graph.train_mask), len(graph.val_mask)))

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = GIN_Net2(in_len=2000, in_feature=13, gin_in_feature=256, num_layers=1, pool_size=3, cnn_hidden=1).to(device)

    model.load_state_dict(torch.load(args.gnn_model)['state_dict'])

    graph.to(device)
    
    test(model, graph, graph.val_mask, device)

if __name__ == "__main__":
    main()
