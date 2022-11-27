import logging

from torch_geometric.datasets import Planetoid

citation_network_name = ['Cora', 'CiteSeer', 'PubMed']

def data_load(*args):
    if args.dataset in citation_network_name:
        dataset = Planetoid(root=args.data_root, name=args.dataset)
        logging.info("dataset.num_feature: {}", dataset.num_feature)
        logging.info("dataset.num_classes: {}", dataset.num_classes)
        logging.info("data.info: {}", dataset.data)

    return dataset