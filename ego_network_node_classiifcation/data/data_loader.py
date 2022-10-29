from cgi import print_arguments
import random
from re import L
import numpy as np
import os
import pickle
import logging
import torch_geometric.data as DataLoader

from fedml.core import *

def get_data(path, data):
    subgraphs, num_graphs, num_features, num_labels = pickle.load(
        open(os.path.join(path, data, "egonetwork.pkl"), "rb")
    )

    return subgraphs, num_graphs, num_features, num_labels

def create_random_split(path, data):
    subgraphs, _, _, _ = get_data(path, data)
    random.shuffle(subgraphs)
    
    # train & test data are from different subgraphs
    train_size = int(len(subgraphs) * 0.8)
    val_size, test_size = int(len(subgraphs) * 0.1)

    logging.info("train_size is {}, val_size is {}, test_size is {}", train_size, val_size, test_size)

    graphs_train = subgraphs[:train_size]
    graphs_val = subgraphs[train_size:train_size+val_size]
    graphs_test = subgraphs[train_size+val_size:]

    return graphs_train, graphs_val, graphs_test

def create_non_uniform_split(args, idxs, client_number, data_type="train", is_loading_cache=True):
    logging.info("create non uniform split: =>")
    N  = len(idxs)
    alpha = args.partition_alpha
    logging.info("idxs = d%, sample_number = d%, client_number = d%" % (idxs, N, client_number))
    partition_cache_file_path = args.part_file + "-" + str(client_number) + "-" + str("alpha") + "-" + data_type + ".pkl"
    logging.info("partition_cache_file_path = {}", format(partition_cache_file_path))

    if is_loading_cache and os.path.exists(partition_cache_file_path):
        logging.info("loading perset partition")
        pickle_file = open(partition_cache_file_path, "rb")
        idx_batch_per_client = pickle.load(pickle_file)
    else:
        min_size = 0
        while min_size < 1:
            idx_batch_per_client = [[] for _ in range(client_number)]
            (idx_batch_per_client, min_size) = partition_class_samples_with_dirichlet_distribution(
                N, alpha, client_number, idx_batch_per_client, idxs
            )
            logging.info("search the min size < 1 ")

        with open(partition_cache_file_path, "wb") as hanndle:
            pickle.dump(idx_batch_per_client, hanndle)

        
    logging.info("saving partition")
    logging.info(idx_batch_per_client)

    sample_num_distribution = []
    for client_id in range(client_number):
        sample_num_distribution.append(len(idx_batch_per_client[client_id]))
        logging.info("client_id = {}, sample_number = {}", client_id, len(idx_batch_per_client[client_id]))
    logging.info("create non uniform split: =>")

    return idx_batch_per_client

def partition_data_by_sample_size(
    args,
    path,
    client_number,
    uniform = True,
    compact = True,
):
    graphs_train, graphs_val, graphs_test = create_random_split(path, args.dataset)
    
    num_train_samples = len(graphs_train)
    num_val_samples = len(graphs_val)
    num_test_samples = len(graphs_test)

    train_idxs = list(range(num_test_samples))
    val_idxs = list(range(num_val_samples))
    test_idxs = list(range(num_test_samples))

    partition_dicts = [None] * client_number
    if uniform:
        client_idxs_train = np.array_split(train_idxs, client_number)
        client_idxs_val = np.array_split(val_idxs, client_number)
        client_idxs_test = np.array_split(test_idxs, client_number)
    else:
        clients_idxs_train = create_non_uniform_split(args, train_idxs, client_number, data_type = "train")
        clients_idxs_val = create_non_uniform_split(args, val_idxs, client_number, data_type = "val")
        clients_idxs_test = create_non_uniform_split(args, test_idxs, client_number, data_type = "test")

    labels_of_all_clients = []
    for client in range(client_number):
        client_train_idxs= client_idxs_train[client]
        client_val_idxs = client_idxs_val[client]
        client_test_idxs = client_idxs_test[client]

        train_graphs_client, train_label_client = [(graphs_train[idx], graphs_train[idx].y) for idx in client_train_idxs]
        labels_of_all_clients.append(train_label_client)

        val_graphs_client, val_label_client = [(graphs_val[idx], graphs_val[idx].y) for idx in client_val_idxs]
        labels_of_all_clients.append(val_label_client)

        test_graphs_client, test_label_client = [(graphs_test[idx], graphs_test[idx].y) for idx in client_test_idxs]
        labels_of_all_clients.append(test_label_client)

        partition_dict = {
            "train": train_graphs_client,
            "val": val_graphs_client,
            "test": test_graphs_client,
        }
        partition_dict[client] = partition_dict

        global_data_dict = {"train":graphs_train, "val": graphs_val, "test": graphs_test}
        return global_data_dict, partition_dicts

def load_partition_data(
    args,
    path,
    client_number,
    uniform = True,
    global_test = True,
    compact = True,
    normalize_features = False,
    normalize_adj = False,
):
    global_data_dict, partition_dicts = partition_data_by_sample_size(
        args, path, client_number, uniform, compact = compact
    )

    data_local_num_dict = dict()
    train_data_local_dict = dict()
    val_data_local_dict = dict()
    test_data_local_dict = dict()

    collator = (
        WalkForestCollator(normalize_features=normalize_features)
        if compact
        else DefaultCollator(normalize_features=normalize_features, normalize_adj=normalize_adj)
    )

    train_data_global = DataLoader(
        global_data_dict["train"], batch_size=args.batch_size, shuffle=True, collate_fn=collator, pin_memory=True
    )
    val_data_global = DataLoader(
        global_data_dict["val"], batch_size=args.batch_size, shuffle=True, collate_fn=collator, pin_memory=True
    )
    test_data_global = DataLoader(
        global_data_dict["test"], batch_size=args.batch_size, shuffle=True, collate_fn=collator, pin_memory=True
    )

    train_data_num = len(global_data_dict["train"])
    val_data_num = len(global_data_dict["val"])
    test_data_num = len(global_data_dict["test"])

    for client_idx in range(client_number):
        train_data_client = partition_dicts[client]["train"]