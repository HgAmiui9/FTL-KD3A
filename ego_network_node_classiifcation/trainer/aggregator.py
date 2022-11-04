import logging
import numpy as np
import torch
import wandb
from sklearn.metrics import confusion_matrix

from fedml.core import ServerAggregator

class FedNodeClfAggregator(ServerAggregator):
    def get_mode_params(self):
        return self.model.cpu().state_dict()
    
    def set_model_params(self, model_params):
        logging.info("set model params")
        self.model.load_state_dict(model_params)
        
    def test(self, test_data, device, args):
        pass

    def test_all(self, train_data_local_dict, test_data_local_dict, device, args)-> bool:
        logging.info("--------------test on the sever-----------------")
        
        model_list, micro_list, macro_list = [], [], []
        for client_idx in test_data_local_dict.keys():
            test_data = test_data_local_dict[client_idx]
            score, model = self._test(test_data, device)
            for idx in range(len(model_list)):
                self._compare_models(model, model_list[idx])
            model_list.append(model)
            micro_list.append(score)
            logging.info("Client {}, Test Micro F1 = {}".format(client_idx, score))
            if args.enable_wandb:
                wandb.log({"Client {}, Test Micro F1 = {}".format(client_idx, score)})
            
        avg_micro = np.mean(np.array(micro_list))
        logging.info("Test Micro F1 = {}".format(avg_micro))
        if args.enable_wandb:
            wandb.log({"Test Micro F1": avg_micro})
       
        return True
        