# Requirements
<img src="https://img.shields.io/badge/-Python-3776AB.svg?logo=python&style=flat&logoColor=white"> <img src="https://img.shields.io/badge/-PyTorch-EE4C2C.svg?logo=pytorch&style=flat&logoColor=white">

# Description
This project demonstrates the use of a ranknet model and regression model. It includes training and evaluating the model using various parameters and settings. There are two types of models for each training method. One is for sequantial data and the other is for non-sequential data. The training process is configurable via command-line arguments.

# Usage Instructions
1. Change path<br>

`train/20240603/rank_sample_seq.ipynb: In [3]`
```train/20240603/rank_sample_seq.ipynb: In [3]
sys.path.append("C:\\Users\\hayas\\proj-rank-general\\git\\code\\ranknet_seq\\")
with open('c:\\Users\\hayas\\proj-rank-general\\git\\data\\features_dict_seq.pkl', 'rb') as p:
    features_dict = pickle.load(p)
with open('c:\\Users\\hayas\\proj-rank-general\\git\\data\\lopocv_dict.pkl', 'rb') as p:
    lopocv_dict = pickle.load(p)
```

`train/20240603/rank_sample_seq.ipynb:In [5]`
```train/20240603/rank_sample_seq.ipynb:In [5]
with open("C:\\Users\\hayas\\proj-rank-general\\git\\output\\ret\\20240603\\{}.pickle".format(file_name), mode="wb") as f:
      pickle.dump(return_dict, f)
```

`code/ranknet_seq/utils.py: line 7`
```code/ranknet_seq/utils.py: line 7
logging.basicConfig(filename="C:\\Users\\hayas\\proj-rank-general\\git\\output\\log\\20240603\\rank_sample_seq.log", level=logging.INFO)
```

2. Run training notebook<br>
`train/20240603/rank_sample_seq.ipynb`

# Task Settings
The task is intended to rank each perceiver's subjective ratings for the five stimuli. For more information on this task, we recommend referring to Section IV-A in this [paper](https://ieeexplore.ieee.org/document/10158500).

# Dataset
This is a toy dataset. The dataset is stored in dictionary format.<br>
`data/features_dict_seq.pkl`: dataset for sequential features.<br>
`data/features_dict_nonseq.pkl`: dataset for non-sequential features.<br>
Keys of these dictionary is `[participant_id]_[stimuli_id]`. <br>
Values of these dictionary is `{'features': [list], 'score': [float]}`.

Next, thise dictionary is keys corresponding to training set and test set.<br>
`data/lopocv_dict.pkl`: keys for ranknet.<br>
`data/lopocv_dict_reg.pkl`: keys for regression.

# Training notebook
Training file is Jupyter notebook format.<br>
`train/20240603/rank_sample_seq.ipynb`: Training notebook for sequential features using ranknet.<br>
`train/20240603/rank_sample_nonseq.ipynb`: Training notebook for non-sequential features using ranknet.<br>
`train/20240603/regression_sample_seq.ipynb`: Training notebook for sequential features using regression.<br>
`train/20240603/regression_sample_nonseq.ipynb`: Training notebook for non-sequential features using regression.<br>

About `margin` parameter of ranknet, we recommend referring to Section V-E in this [paper](https://ieeexplore.ieee.org/document/10158500).