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
    input_size_features = len(minibatch_data[0]["features"][0])
    seq_num_list = []
    for sample in minibatch_data:
        seq_num_list.append(len(sample["features"]))
    temp_features = torch.zeros((batch_size, max(seq_num_list), input_size_features))
    gold = torch.tensor([s["score"] for s in minibatch_data]).to(torch.float32)

    id_list = []
    for idx, data in enumerate(minibatch_data):
        id_list.append(data["id"])
        seq_len = len(data["features"])
        tensor_features = torch.tensor(np.array(data["features"])).to(torch.float32)
        temp_features[idx, :seq_len, :] = tensor_features
    data_to_return = {"features": temp_features, "golds": gold, "id_list": id_list, "seq_num_list": seq_num_list}
    return data_to_return

class ConversationRelDataModule():
    def __init__(self, train_dataset, test_dataset, batch_size, collator, features_dict):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.collator = collator
        self.features_dict = features_dict

    def setup(self, stage):

        if stage == "fit" or stage is None:
            self.train_data = ConversationRelDataset(self.train_dataset, self.features_dict)

        if stage == "test" or stage is None:
            self.test_data = ConversationRelDataset(self.test_dataset, self.features_dict)

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
    def __init__(self, samples, features_dict):
            
        self.dataset = []
        for s in samples:
            self.dataset.append(
                {"id": s, "score": features_dict[s]["score"], "features": features_dict[s]["features"]})   
        log.info("finished loading {} examples".format(len(self.dataset)))
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]