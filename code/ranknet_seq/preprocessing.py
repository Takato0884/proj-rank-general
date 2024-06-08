import json
import torch, os
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler
import utils
import pickle
import math
import numpy as np
import random

log = utils.get_logger()

def collator(minibatch_data):
    batch_size = len(minibatch_data)
    input_size_features = len(minibatch_data[0]["features_id1"][0])
    seq_num_list_id1, seq_num_list_id2 = [], []
    for sample in minibatch_data:
        seq_num_list_id1.append(len(sample["features_id1"]))
        seq_num_list_id2.append(len(sample["features_id2"]))
    utt_num_list = seq_num_list_id1 + seq_num_list_id2

    temp_features_id1 = torch.zeros((batch_size, max(utt_num_list), input_size_features))
    temp_features_id2 = torch.zeros((batch_size, max(utt_num_list), input_size_features))

    gold = torch.tensor([s["gold"] for s in minibatch_data]).to(torch.float32)

    pair_list = []
    for idx, data in enumerate(minibatch_data):
        pair_list.append((data["id1"], data["id2"]))
        seq_len_id1 = len(data["features_id1"])
        seq_len_id2 = len(data["features_id2"])


        tensor_features_id1 = torch.tensor(np.array(data["features_id1"])).to(torch.float32)
        tensor_features_id2 = torch.tensor(np.array(data["features_id2"])).to(torch.float32)

        temp_features_id1[idx, :seq_len_id1, :] = tensor_features_id1
        temp_features_id2[idx, :seq_len_id2, :] = tensor_features_id2

    data_to_return = {"features_id1": temp_features_id1, "features_id2": temp_features_id2, \
                      "golds": gold, "pair_list": pair_list, "seq_num_list_id1": seq_num_list_id1, "seq_num_list_id2": seq_num_list_id2}

    return data_to_return

class ConversationRelDataModule():

    def __init__(self, train_dataset, test_dataset, batch_size, collator, features_dict, margin):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.collator = collator
        self.features_dict = features_dict
        self.margin = margin

    def setup(self, stage):

        if stage == "fit" or stage is None:
            self.train_data = ConversationRelDataset(self.train_dataset, self.features_dict, self.margin)

        if stage == "test" or stage is None:
            self.test_data = ConversationRelDataset(self.test_dataset, self.features_dict, 0)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            shuffle=True,
            batch_size=self.batch_size,
            collate_fn=self.collator)

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            collate_fn=self.collator)

class ConversationRelDataset(Dataset):
    def __init__(self, samples, features_dict, margin):
            
        self.dataset = []
        for s in samples:
            k1 = s[0]
            k2 = s[1]
            if abs(features_dict[k1]["score"] - features_dict[k2]["score"]) >= margin:
                gold = 1 if features_dict[k1]["score"] > features_dict[k2]["score"] else 0
                self.dataset.append(
                    {"id1": k1, "id2": k2, "score_id1": features_dict[k1]["score"], "score_id2": features_dict[k2]["score"], "gold": gold, \
                    "features_id1": features_dict[k1]["features"], "features_id2": features_dict[k2]["features"]})   
        log.info("finished loading {} examples".format(len(self.dataset)))
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
