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
        self.opt = opt
        self.model = model
        self.args = args

    def train(self):
        log.info("lstm:{}, lr:{}, drop: {}, margin: {}: epoch: {}".format(self.args.rnn, self.args.learning_rate, self.args.drop_rate, \
                                                                                     self.args.margin, self.args.epochs))
        best_test_rmse = None
        train_loss_list, test_loss_list = [], []
        train_acc_list, test_acc_list = [], []
        test_golds_list, test_preds_list = [], []
        train_golds_list, pair_list, score_list = [], [], []
        
        # Train
        for epoch in range(1, self.args.epochs + 1):
            loss_train, acc_train, train_golds = self.train_epoch(epoch)
            loss_test, acc_test, test_golds, test_preds, pairs, scores = self.evaluate()
            log.info("[Test set] [Loss {:.4f}] [Acc: {:.4f}]".format(loss_test, acc_test))
            if best_test_rmse is None or loss_test < best_test_rmse:
                best_test_rmse = loss_test
                log.info("best loss model.")

            train_loss_list.append(loss_train)
            test_loss_list.append(loss_test)
            train_acc_list.append(acc_train)
            test_acc_list.append(acc_test)
            test_golds_list.append(test_golds)
            test_preds_list.append(test_preds)
            train_golds_list.append(train_golds)
            pair_list.append(pairs)
            score_list.append(scores)

        log.info("-----------------------------------------------")
        return {"test_golds_list": test_golds_list, "test_preds_list": test_preds_list, "train_loss_list": train_loss_list, "test_loss_list": test_loss_list, \
                "train_acc_list": train_acc_list,"test_acc_list": test_acc_list, "pair_list": pair_list, "score_list": score_list}

    def train_epoch(self, epoch):
        dataset = self.trainset
        start_time = time.time()
        self.model.train()

        golds = []
        preds = []
        loss_epoch = 0
        for step, batch in enumerate(dataset):
            golds.extend(batch["golds"].tolist())
            self.model.zero_grad()
            for k, v in batch.items():
                if k not in ["pair_list"]:
                    batch[k] = v.to("cuda:0")
            loss, y_hat, _  = self.model(batch)
            loss_epoch += loss
            preds.extend(y_hat.detach().to("cpu").tolist())
            loss.backward()
            self.opt.step()

        accuracy = accuracy_score(preds, golds)
        end_time = time.time()
        log.info("[Epoch %d] [Loss: %f] [Acc: %f] [Time: %f]" %
                 (epoch, loss_epoch, accuracy, end_time - start_time))

        return loss_epoch, accuracy, golds

    def evaluate(self):
        dataset = self.testset
        self.model.eval()
        with torch.no_grad():
            golds = []
            preds = []
            pair_list = []
            score_list = []
            loss_epoch = 0
            for step, batch in enumerate(dataset):
                golds.extend(batch["golds"].tolist())
                for k, v in batch.items():
                    if k not in ["pair_list"]:
                        batch[k] = v.to("cuda:0")
                loss, y_hat, score  = self.model(batch)
                loss_epoch += loss
                preds.extend(y_hat.detach().to("cpu").tolist())
                pair_list.extend(batch["pair_list"])
                score_list.extend(score.tolist())

        accuracy = accuracy_score(preds, golds)

        return loss_epoch, accuracy, golds, preds, pair_list, score_list
