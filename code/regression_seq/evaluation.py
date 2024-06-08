import utils
from sklearn.metrics import accuracy_score
from scipy import stats
import numpy as np

log = utils.get_logger()

def cal_acc_fold(epoch, return_dict, return_dict_rank):
    for g in return_dict.keys():
        golds_list = []
        preds_list = []
        for i in return_dict_rank[g]["pair_list"][0]:
            idx1 = return_dict[g]["conv_list"][0].index(i[0])
            idx2 = return_dict[g]["conv_list"][0].index(i[1])
            gold1 = return_dict[g]["test_golds_list"][epoch][idx1]
            gold2 = return_dict[g]["test_golds_list"][epoch][idx2]
            pred1 = return_dict[g]["test_preds_list"][epoch][idx1]
            pred2 = return_dict[g]["test_preds_list"][epoch][idx2]
            golds_list.append(1 if gold1 > gold2 else 0)
            preds_list.append(1 if pred1 > pred2 else 0)
        print("Fold{}: {}: {}".format(g, accuracy_score(preds_list, golds_list), return_dict[str(g)]["best_lr"]))

def cal_acc(start, end, return_dict, return_dict_rank):
    for e in range(start, end+1, 5):
        golds_list = []
        preds_list = []
        for g in return_dict.keys():
            for i in return_dict_rank[g]["pair_list"][0]:
                idx1 = return_dict[g]["conv_list"][0].index(i[0])
                idx2 = return_dict[g]["conv_list"][0].index(i[1])
                gold1 = return_dict[g]["test_golds_list"][e-1][idx1]
                gold2 = return_dict[g]["test_golds_list"][e-1][idx2]
                pred1 = return_dict[g]["test_preds_list"][e-1][idx1]
                pred2 = return_dict[g]["test_preds_list"][e-1][idx2]
                golds_list.append(1 if gold1 > gold2 else 0)
                preds_list.append(1 if pred1 > pred2 else 0)
        print("Acc: {}: {}".format(e, accuracy_score(preds_list, golds_list)))

# def cal_acc(start, end, return_dict):
#     for i in range(start, end+5, 5):
#         all_golds = []
#         all_preds = []
#         for k, v in return_dict.items():
#             all_golds.extend(v["test_golds_list"][i-1])
#             all_preds.extend(v["test_preds_list"][i-1])
#         acc = accuracy_score(all_preds, all_golds)
#         print("Acc: {}: {}".format(i, acc))

def cal_acc_margin(start, end, return_dict, return_dict_rank, features_dict):
    for e in range(start, end+5, 5):
        print("------{}epoch------".format(e))
        golds_list, preds_list, diff_list = [], [], []
        for g in return_dict.keys():
            for i in return_dict_rank[g]["pair_list"][0]:
                idx1 = return_dict[g]["conv_list"][0].index(i[0])
                idx2 = return_dict[g]["conv_list"][0].index(i[1])
                gold1 = return_dict[g]["test_golds_list"][e-1][idx1]
                gold2 = return_dict[g]["test_golds_list"][e-1][idx2]
                pred1 = return_dict[g]["test_preds_list"][e-1][idx1]
                pred2 = return_dict[g]["test_preds_list"][e-1][idx2]
                golds_list.append(1 if gold1 > gold2 else 0)
                preds_list.append(1 if pred1 > pred2 else 0)
                # print(features_dict[i[0]]["rapport"])
                diff = abs(features_dict[i[0]]["rapport"] - features_dict[i[1]]["rapport"])
                diff_list.append(diff)

        for j in range(0, 20, 10):
            sub_golds, sub_preds = [], []
            for idx, m in enumerate(diff_list):
                if (m > j) and (m <= j+10):
                    sub_golds.append(golds_list[idx])
                    sub_preds.append(preds_list[idx])
            acc = accuracy_score(sub_preds, sub_golds)
            print("{}<margin<={}: Acc: {}: len: {}".format(j, j+10, acc, len(sub_golds)))

        sub_golds, sub_preds = [], []
        for idx, m in enumerate(diff_list):
            if (m > 20):
                sub_golds.append(golds_list[idx])
                sub_preds.append(preds_list[idx])
        acc = accuracy_score(sub_preds, sub_golds)
        print("{}<margin: Acc: {}: len: {}".format(20, acc, len(sub_golds)))

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

def cal_rank(start, end, return_dict):
    for i in range(start, end+5, 5):
        conv_list = []
        preds_list = []
        golds_list = []
        for k, v in return_dict.items():
            conv_list.extend(v["conv_list"][i-1])
            preds_list.extend(v["test_preds_list"][i-1])
            golds_list.extend(v["test_golds_list"][i-1])
        ranklist_dict = create_id_dict_per_person(conv_list)

        for k, v in ranklist_dict.items():
            v["golds"] = [golds_list[conv_list.index(id)] for id in v["ids"]]
            v["preds"] = [preds_list[conv_list.index(id)] for id in v["ids"]]
            
        tau_list = []
        precision_top = []
        precision_bottom = []
        for k, v in ranklist_dict.items():
            tau, p_value = stats.kendalltau(v["preds"], v["golds"])
            tau_list.append(tau)

            rank_index_golds = np.array(v["golds"]).argsort()
            rank_index_preds = np.array(v["preds"]).argsort()

            golds_preds_and = set(rank_index_golds[-1:]) & set(rank_index_preds[-1:])
            precision_top.append(len(golds_preds_and))
            golds_preds_and = set(rank_index_golds[:1]) & set(rank_index_preds[:1])
            precision_bottom.append(len(golds_preds_and))

        print("------{}epoch------".format(i))
        print("tau: {}, len: {}".format(np.mean(tau_list), len(tau_list)))
        print("P@1: {}, len: {}".format(np.mean(precision_top), len(precision_top)))
        print("P@-1: {}, len {}".format(np.mean(precision_bottom), len(precision_bottom)))