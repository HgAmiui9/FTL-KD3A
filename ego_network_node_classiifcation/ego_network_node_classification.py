from pickle import NONE
import fedml
from fedml import FedMLRunner
from data.data_loader import *

def load_data(args):
    num_cats, feature_dim = 0, 0
    # if args.dataset not in ["CS", "Physics", "cora", "citeseer", "DBLP", "PubMed"]:
    #     raise Exception("no such dataset")
    # elif args.dataset in ["CS", "Physics"]:
    #     args.type_network = "coauthor"
    # else:
    #     args.type_network = "citation"

    args.dataset = "cora"
    args.type_network = "citation"

    # compact = args.model == "graphsage"

    compact = True

    uniform = True if args.partion_method == "homo" else False

    if args.model == "gcn":
        args.normalize_feature = True
        args.normalize_adjacency = True

    _, _, feature_dim, num_cats = get_data(args.data_cacje_dir, args.dataset)

    (
        train_data_num, 
        val_data_num,
        test_data_num,
        train_data_global,
        val_data_global,
        test_data_global,
        data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        num_cats,
    ) = load_partition_data(
        args,
        args.data_cache_dir,
        args.client_num_in_total,
        uniform,
        compact,
        normalize_feature=args.normalize_features,
        normalize_adjacency=args.normalize_adjacency,
    )

    dataset = [
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        num_cats
    ]

    return dataset, num_cats, feature_dim

# def create_model(args, feature_dim, num_cats, output_dim=None):
#     logging.info("create model model name is {}, output_dim is {}".format(args.model, output_dim))
#     if args.model == "gcn":
#         mdoel = GCNNodeCLF(
#             nfeat=feat_dim, nhid=args.hidden_size, nclass=num_cats,nlayer=args.n_layers, dropout=args.dropout
#         )

if __name__ == "__main__":
    args = fedml.init()
    
    device = fedml.device.get_device(args)
    
    dataset, num_cats, feature_dim = load_data(args)
    
    model, trainer = create_model(args, feature_dim, num_cats)
    aggregator = FedNodeClfAggregator(model, args)
    
    fedml_runner = FedMLRunner(args, device, dataset, model, trainer, aggregator)
    fedml.runner.run()