# Requirements
<img src="https://img.shields.io/badge/-Python-3776AB.svg?logo=python&style=flat&logoColor=white"> <img src="https://img.shields.io/badge/-PyTorch-EE4C2C.svg?logo=pytorch&style=flat&logoColor=white">

# Description
This project demonstrates the use of a ranknet model and regression model. It includes training and evaluating the model using various parameters and settings. There are two types of models for each training method. One is for sequantial data and the other is for non-sequential data. The training process is configurable via command-line arguments.

# Task Settings
The task is intended to rank each perceiver's subjective ratings for the five stimuli. For more information on this task, we recommend referring to Section 3.1 in this [paper](https://link.springer.com/chapter/10.1007/978-3-031-61312-8_2).

# Dataset
This is a toy dataset. The dataset is stored in dictionary format.
`data/features_dict_seq.pkl`: dataset for sequential features.
`data/features_dict_nonseq.pkl`: dataset for non-sequential features.
Keys of these dictionary is `[participant_id]_[stimuli_id]`. 
Values of these dictionary is `{'features': [list], 'score': [float]}`.


Next, thise dictionary is keys corresponding to training set and test set.
`data/lopocv_dict.pkl`: keys for ranknet.
`data/lopocv_dict_reg.pkl`: keys for regression.