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
        # パラメータの設定
        self.mse_loss = nn.MSELoss()
        # # 層数の指定
        input_size = 8
        intermediate_size = 32
        # モデルの指定
        self.seq_context_encoder = seq_context.SeqContext(input_size, intermediate_size, args)
        self.predictor = linear_regression.LinearRegression(intermediate_size, intermediate_size, 1, args)
        log.info("input_uni: {}, intermediate_uni: {}".format(input_size, intermediate_size))

    def get_rep(self, features, seq_len):
        _, lstm_conv = self.seq_context_encoder(features, seq_len)
        return lstm_conv

    def forward(self, data):

        context_rep_id1= self.get_rep(data["features_id1"], data["seq_num_list_id1"])
        context_rep_id2= self.get_rep(data["features_id2"], data["seq_num_list_id2"])

        score_id1 = self.predictor(context_rep_id1)
        score_id2 = self.predictor(context_rep_id2)

        loss = self.pairweise_loss(score_id1, score_id2, data["golds"])
        scores = torch.cat((score_id1, score_id2), 1)
        y_hat =  torch.argmin(scores, dim=-1)

        return loss, y_hat, scores

    def pairweise_loss(self, score_id1, score_id2, golds):

        o = score_id1 - score_id2
        pos = torch.mul(-golds.unsqueeze(dim=-1), o) + F.softplus(o)
        loss = (pos).mean()

        return loss
