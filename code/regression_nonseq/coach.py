import copy
import time

import numpy as np
import torch
from tqdm import tqdm
from sklearn import metrics
import utils
from sklearn.metrics import accuracy_score

log = utils.get_logger()

class Coach:

    def __init__(self, trainset, testset, model, opt, args):
        self.trainset = trainset
        self.testset = testset
        self.model = model
        self.opt = opt
        self.args = args

    def train(self):
        log.info("lstm:{}, lr:{}, drop: {}, modal: {}, epoch: {}".format(self.args.rnn, self.args.learning_rate, self.args.drop_rate, \
                                                                                     self.args.modal, self.args.epochs))
        best_test_rmse = None
        train_loss_list = []
        test_loss_list = []
        test_golds_list = []
        test_preds_list = []
        conv_list = []
        
        # Train
        for epoch in range(1, self.args.epochs + 1):
            loss_train = self.train_epoch(epoch)
            loss_test, test_golds, test_preds, conv = self.evaluate()
            log.info("[Test set] [Loss {:.4f}]".format(loss_test))
            if best_test_rmse is None or loss_test < best_test_rmse:
                best_test_rmse = loss_test
                log.info("best loss model.")

            train_loss_list.append(loss_train)
            test_loss_list.append(loss_test)
            test_golds_list.append(test_golds)
            test_preds_list.append(test_preds)
            conv_list.append(conv)

        log.info("-----------------------------------------------")
        return {"test_golds_list": test_golds_list, "test_preds_list": test_preds_list, "train_loss_list": train_loss_list, "test_loss_list": test_loss_list, "conv_list": conv_list}

    def train_epoch(self, epoch):
        dataset = self.trainset
        start_time = time.time()
        self.model.train()

        # ここからバッチごとの処理
        loss_epoch = 0
        for step, batch in enumerate(dataset):
            self.model.zero_grad()
            for k, v in batch.items():
                if k not in ["id_list"]:
                    batch[k] = v.to("cuda:0")
            loss, _  = self.model(batch)
            loss_epoch += np.sqrt(loss.detach().to("cpu").tolist())
            loss.backward()
            self.opt.step()

        end_time = time.time()
        log.info("[Epoch %d] [Loss: %f][Time: %f]" %
                 (epoch, loss_epoch, end_time - start_time))

        return loss_epoch 

    def evaluate(self):
        dataset = self.testset
        self.model.eval()
        with torch.no_grad():
            golds_list = []
            preds_list = []
            conv_list = []
            loss_epoch = 0
            for step, batch in enumerate(dataset):
                golds_list.extend(batch["golds"].tolist())
                for k, v in batch.items():
                    if k not in ["id_list"]:
                        batch[k] = v.to("cuda:0")
                loss, preds  = self.model(batch)
                loss_epoch += np.sqrt(loss.detach().to("cpu").tolist())
                preds_list.extend(np.squeeze(preds.detach().to("cpu").tolist()))
                conv_list.extend(batch["id_list"])

        return loss_epoch, golds_list, preds_list, conv_list
