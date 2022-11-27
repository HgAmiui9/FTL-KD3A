from pickle import NONE
import fedml
from fedml import FedMLRunner
from data.data_loader import *
from trainer.aggregator import FedNodeClfAggregator
from trainer.trainer import FedNodeCLFTrainer
from model.sage import SAGE
import logging
import faulthandler

faulthandler.enable()

def load_data(args):
    num_cats, feature_dim = 0, 0

    args.dataset = "cora"
    args.type_network = "citation"

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

def create_model(model_name, feature_dim, num_cats):
    logging.info("create_model. model_name = {model_name}, output_dim = {num_cats}")
    model = SAGE(in_features=feature_dim, hidden_features=args.hidden_dim, n_class=num_cats, n_layer=args.n_layers, dropout=args.dropout)
    trainer = FedNodeCLFTrainer(model, args)

if __name__ == "__main__":
    logging.basicConfig(filename="run.log", encoding="utf-8", level=logging.DEBUG)
    args = fedml.init()
    logging.info('fedml init ~ ')

    device = fedml.device.get_device(args)
    
    dataset, num_cats, feature_dim = load_data(args)
    logging.info('load data ~ ')
    
    model, trainer = create_model(args, feature_dim, num_cats)
    logging.info('create_mode')

    aggregator = FedNodeClfAggregator(model, args)
    logging.info('model affregator')
    
    fedml_runner = FedMLRunner(args, device, dataset, model, trainer, aggregator)
    logging.info('fedml running')
    
    fedml.runner.run()
    