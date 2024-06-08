import torch, json, math
import torch.nn as nn
from torch.nn import functional as F
from collections import OrderedDict
from argparse import ArgumentParser
from torch.optim import Adam, SGD, Adadelta
import seq_context
import linear_regression
import utils
import numpy as np

log = utils.get_logger()

class MainModel(nn.Module):
    def __init__(self, args):
        super(MainModel, self).__init__()
        # パラメータの設定
        self.mse_loss = nn.MSELoss()
        # # 層数の指定
        input_size = 8
        intermediate_size = 32
        # モデルの指定
        self.predictor = linear_regression.LinearRegression(input_size, intermediate_size, 1, args)
        log.info("input_uni: {}, intermediate_uni: {}".format(input_size, intermediate_size))

    def forward(self, data):

        preds = self.predictor(data["features"])
        if len(preds.detach().to("cpu").tolist()) != 1:
            loss = self.mse_loss(torch.squeeze(preds), data["golds"])
        elif len(preds.detach().to("cpu").tolist()) == 1:
            loss = self.mse_loss(preds[0], data["golds"])

        return loss, preds
