import utils
from sklearn.metrics import accuracy_score
from scipy import stats
import numpy as np

log = utils.get_logger()

def cal_acc(start, end, return_dict):
    for i in range(start, end+5, 5):
        all_golds = []
        all_preds = []
        for k, v in return_dict.items():
            all_golds.extend(v["test_golds_list"][i-1])
            all_preds.extend(v["test_preds_list"][i-1])
        acc = accuracy_score(all_preds, all_golds)
        print("Acc: {}: {}".format(i, acc))
    return acc

def cal_acc_margin(start, end, return_dict, features_dict):
    for i in range(start, end+5, 5):
        margin_acc_list = []
        print("------{}epoch------".format(i))
        all_golds, all_preds, all_margins = [], [], []
        for k, v in return_dict.items():
            all_golds.extend(v["test_golds_list"][i-1])
            all_preds.extend(v["test_preds_list"][i-1])
            for (c1, c2) in v["pair_list"][i-1]:
                all_margins.append(abs(features_dict[c1]["score"] - features_dict[c2]["score"]))

        for j in range(0, 20, 10):
            sub_golds, sub_preds = [], []
            for idx, m in enumerate(all_margins):
                if (m > j) and (m <= j+10):
                    sub_golds.append(all_golds[idx])
                    sub_preds.append(all_preds[idx])
            acc = accuracy_score(sub_preds, sub_golds)
            margin_acc_list.append(acc)
            print("{}<margin<={}: Acc: {}: len: {}".format(j, j+10, acc, len(sub_golds)))

        sub_golds, sub_preds = [], []
        for idx, m in enumerate(all_margins):
            if (m > 20):
                sub_golds.append(all_golds[idx])
                sub_preds.append(all_preds[idx])
        acc = accuracy_score(sub_preds, sub_golds)
        margin_acc_list.append(acc)
        print("{}<margin: Acc: {}: len: {}".format(20, acc, len(sub_golds)))

    return margin_acc_list

def create_id_dict_per_person(id_list):
    ranklist_dict = {}
    u_id_set = set()
    for id in id_list:
        u_id_set.add(id[:3])
    u_id_list = sorted(list(u_id_set))
    for u in u_id_list:
        ranklist_dict[u] = {"ids": []}
        for id in id_list:
            if u == id[:3]:
                ranklist_dict[u]["ids"].append(id)

    return ranklist_dict

def cal_rank(start, end, return_dict, features_dict):

    conv_set = set()
    for g in return_dict.keys():
        for p in return_dict[g]["pair_list"][-1]:
            conv_set.add(p[0])
            conv_set.add(p[1])
    conv_list = sorted(list(conv_set))

    ranklist_dict = create_id_dict_per_person(conv_list)

    for i in range(start, end+5, 5):
        pred_dict = {}
        gold_dict = {}
        for c in conv_list:
            gold_dict[c] = features_dict[c]["score"]
        for c in conv_list:
            for k, v in return_dict.items():
                for idx, p in enumerate(v["pair_list"][-1]):
                    if p[0] == c:
                        pred_dict[c] = v["score_list"][i-1][idx][0]
                    elif p[1] == c:
                        pred_dict[c] = v["score_list"][i-1][idx][1]

        for k, v in ranklist_dict.items():
            v["golds"] = [gold_dict[id] for id in v["ids"]]
            v["preds"] = [pred_dict[id] for id in v["ids"]]
        
        tau_list = []
        precision_top = []
        precision_bottom = []
        for k, v in ranklist_dict.items():
            tau, p_value = stats.kendalltau(v["preds"], v["golds"])
            tau_list.append(tau)

            rank_index_golds = np.array(v["golds"]).argsort()
            rank_index_preds = np.array(v["preds"]).argsort()

            golds_preds_and = set(rank_index_golds[-1:]) & set(rank_index_preds[-1:])
            precision_top.append(len(golds_preds_and)/1)
            golds_preds_and = set(rank_index_golds[:1]) & set(rank_index_preds[:1])
            precision_bottom.append(len(golds_preds_and)/1)

        print("------{}epoch------".format(i))
        print("tau: {}, len: {}".format(np.mean(tau_list), len(tau_list)))
        print("P@1: {}, len: {}".format(np.mean(precision_top), len(precision_top)))
        print("P@-1: {}, len {}".format(np.mean(precision_bottom), len(precision_bottom)))
        rank_list = [np.mean(tau_list), np.mean(precision_top), np.mean(precision_bottom)]
    return rank_list

def cal_acc_fold(return_dict, epoch):
    acc_list_fold = []
    for i in return_dict.keys():
        print("Fold {}: {}: {}".format(i, return_dict[str(i)]["test_acc_list"][epoch-1], return_dict[str(i)]["best_lr"]))
        acc_list_fold.append(return_dict[str(i)]["test_acc_list"][epoch-1])
    return acc_list_fold