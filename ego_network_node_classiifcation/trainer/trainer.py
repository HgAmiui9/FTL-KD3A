import logging
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from fedml.core.alg_frame.client_trainer import ClientTrainer

class FedNodeCLFTrainer(ClientTrainer):
    def get_model_params(self):
        return self.model.cpu().state_dict()
    
    def set_model_params(self, model_params):
        logging.info("set model params")
        self.model.load_state_dict(model_params)
        
    def train(self, train_data, device, args):
        model = self.model
        
        model.to(device)
        model.train()
        
        try:
            test_data = self.test_data
        except:
            pass

        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(), lr=args.learning_rate, weigh_decay=args.weight_decay
            )
        else:
            optimizer = torch.optim.Adam(
                model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
            )
            
        max_test_score = 0
        best_model_params={}
        for epoch in range(args.epochs):
            for batch_idx, batch in enumerate(train_data):
                batch.to(device)
                optimizer.zero_grad()
                pred=model(batch)
                label=batch.y
                loss=model.loss(pred,label)
                loss.backward()
                optimizer.step()
                
                if ((batch_idx+1) % args.frequency_of_the_test == 0) or (batch_idx+1 == len(train_data)):
                    if test_data is not None:
                        test_score, _ = self.test(test_data, device)
                    print(
                        "Epoch {}, Iter = {}/{}: Test accuracy = {}".format(
                            epoch, batch_idx + 1, len(train_data), test_score
                        )
                    )
                    if test_score > max_test_score:
                        max_test_score = test_score
                        best_model_params = {
                            k: v for k, v in model.cpu().state_dict().items()
                        }
                    print("Current best = {}".format(max_test_score))
        return max_test_score, best_model_params
    
    
    def test(self, test_data, device):
        logging.info("___________test___________")
        model=self.model
        model.eval()
        model.to(device)
        # conf_mat=np.zeros((self.model.nclass))
        
        conf_mat=np.zeros((self.model.nclass, self.model.nclass))
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_data):
                batch.to(device)

                pred=model(batch)
                label=batch.y
                cm_result=confusion_matrix(
                    label.cpu().numpy().flatten(),
                    pred.argmax(dim=1).cpu().numpy().flatten(),
                    labels=np.arange(0, self.model.nclass),
                )
                conf_mat += cm_result
                
        TP = np.trace(conf_mat)
        FP = np.sum(conf_mat)-TP
        
        FN = FP
        micro_pr = TP/(TP+FP)
        micro_rec = TP/(TP+FN)
        
        if micro_pr + micro_rec == 0.0:
            denominator = micro_pr + micro_rec + np.finfo(float).eps
        else:
            denominator = micro_pr + micro_rec
        
        micro_F1 = 2 * micro_pr * micro_rec / denominator
        logging.info("F1 score = {}".format(micro_rec))
        return micro_F1, model
        