import torch, json, math
import torch.nn as nn
from torch.nn import functional as F
from collections import OrderedDict
from argparse import ArgumentParser
from torch.optim import Adam, SGD, Adadelta
import seq_context
import linear_regression
import utils

log = utils.get_logger()

class MainModel(nn.Module):
    def __init__(self, args):
        super(MainModel, self).__init__()
        self.mse_loss = nn.MSELoss()
        input_size = 8
        intermediate_size = 32
        self.predictor = linear_regression.LinearRegression(input_size, intermediate_size, 1, args)
        log.info("input_uni: {}, intermediate_uni: {}".format(input_size, intermediate_size))

    def forward(self, data):
        score_id1 = self.predictor(data["features_id1"])
        score_id2 = self.predictor(data["features_id2"])
        loss = self.pairweise_loss(score_id1, score_id2, data["golds"])
        scores = torch.cat((score_id1, score_id2), 1)
        y_hat =  torch.argmin(scores, dim=-1)
        return loss, y_hat, scores

    def pairweise_loss(self, score_id1, score_id2, golds):

        o = score_id1 - score_id2
        pos = torch.mul(-golds.unsqueeze(dim=-1), o) + F.softplus(o)
        loss = (pos).mean()
        return loss
