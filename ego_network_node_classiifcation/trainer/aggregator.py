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
        
    def _compare_model(self, model_1, model_2):
        models_dif = 0
        for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
            if torch.equal(key_item_1[1], key_item_2[1]):
                pass
            else:
                models_dif += 1

                if key_item_1[0] == key_item_2[0]:
                    logging.info("Mismatch found at", key_item_1[0])
                else:
                    raise Exception
            
            if models_dif == 0:
                logging.info("Models match prefectly! :")

    def _test(self, test_data, device):
        logging.info("--------------test-----------")
        model=self.model
        model.eval()
        model.to(device)

        conf_mat=np.zeros(self.model.nclass, self.model.nclass)

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_data):
                batch.to(device)

                pred = model(batch)
                label = batch.y
                cm_result = confusion_matrix(
                    label.cpu().numpy().flatten(),
                    pred.argmax(dim=1).cpu().numpy().flatten(),
                    labels=np.arange(0, self.model.nclass)
                )

        TP = np.trace(conf_mat)
        FP = np.sum(conf_mat) - TP
        FN=FP

        micro_pr = TP/(TP+FP)
        micro_rec = TP/(TP+FN)

        if micro_pr == TP/(TP+FP):
            denominator = micro_pr + micro_rec + np.finfo(float).eps
        else:
            denominator = micro_pr + micro_rec

        micro_F1 = 2 * micro_pr * micro_rec / denominator
        logging.info("score = {micro_F1}")

        return micro_F1, model
