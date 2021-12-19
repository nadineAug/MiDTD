#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
from ..misc import load_pickle
import logging
from tqdm import tqdm
from sklearn.metrics import average_precision_score
import numpy as np
import pickle
import torch.nn.functional as F

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)

def load_state(net, optimizer, scheduler, args, load_best=False):
    """ Loads saved model and optimizer states if exists """
    base_path = "./data/checkpoint/"
    amp_checkpoint = None

    checkpoint_path = os.path.join(base_path, "task_test_checkpoint_{}.pth.tar".format(args.model_tag))
    best_path = "./base_model/bmp_{}.pkl".format(args.model_tag)

    start_epoch, best_pred, checkpoint = 0, 0, None
    if (load_best == True) and os.path.isfile(best_path):
        checkpoint = torch.load(best_path)
        logger.info("Loaded best model.")
    elif os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        logger.info("Loaded checkpoint model.")
    if checkpoint != None:
        start_epoch = checkpoint['epoch']
        best_pred = checkpoint['best_AUC']
        net.load_state_dict(checkpoint['state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None:
            from collections import Counter
            checkpoint['scheduler']['milestones'] = Counter(checkpoint['scheduler']['milestones'])
            scheduler.load_state_dict(checkpoint['scheduler'])
        amp_checkpoint = checkpoint['amp']
        logger.info("Loaded model and optimizer.")    
    return start_epoch, best_pred, amp_checkpoint

def load_results(model_no=0):
    """ Loads saved results if exists """
    losses_path = "./data/task_test_losses_per_epoch.pkl"
    accuracy_path = "./data/task_train_accuracy_per_epoch.pkl"
    auc_path = "./data/task_test_auc_per_epoch.pkl"
    if os.path.isfile(losses_path) and os.path.isfile(accuracy_path) and os.path.isfile(auc_path):
        losses_per_epoch = load_pickle("task_test_losses_per_epoch.pkl")
        accuracy_per_epoch = load_pickle("task_train_accuracy_per_epoch.pkl")
        auc_per_epoch = load_pickle("task_test_auc_per_epoch.pkl")
        logger.info("Loaded results buffer")
    else:
        losses_per_epoch, accuracy_per_epoch, auc_per_epoch = [], [], []
    return losses_per_epoch, accuracy_per_epoch, auc_per_epoch

def evaluate_(output, labels, ignore_idx):
    idxs = (labels != ignore_idx).squeeze()
    o_labels = torch.softmax(output, dim=1).max(1)[1]
    l = labels.squeeze()[idxs]; o = o_labels[idxs]

    try:
        if len(idxs) > 1:
            acc = (l == o).sum().item()/len(idxs)
        else:
            acc = (l == o).sum().item()
    except BaseException as e:
    # except TypeError as e:
        acc = 0
    # if len(idxs) > 1:
    #     acc = (l == o).sum().item()/len(idxs)
    # else:
    #     acc = (l == o).sum().item()
    l = l.cpu().numpy().tolist() if l.is_cuda else l.numpy().tolist()
    o = o.cpu().numpy().tolist() if o.is_cuda else o.numpy().tolist()

    return acc, (o, l)

def generate_order_list(net, test_loader, pad_id, cuda, args):
    logger.info("generating order list...")
    AUC = 0; out_labels = []; true_labels = []
    num_classes = args.num_classes
    y_true = []; y_scores = []; y_prob=[]

    net.eval()
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
            x, e1_e2_start, labels, _ = data
            attention_mask = (x != pad_id).float()
            token_type_ids = torch.zeros((x.shape[0], x.shape[1])).long()
            temp = np.zeros((len(labels), num_classes)).tolist()
            cnt = 0
            for r in labels:
                temp[cnt][r.item()] = 1
                cnt += 1
            total_y = np.array(temp)
            if cuda:
                x = x.cuda()
                labels = labels.cuda()
                attention_mask = attention_mask.cuda()
                token_type_ids = token_type_ids.cuda()
                
            classification_logits = net(x, token_type_ids=token_type_ids, attention_mask=attention_mask, Q=None,\
                          e1_e2_start=e1_e2_start)

            total_logit = classification_logits.cpu().data.numpy()
            y_true.append(total_y[:, 1:])
            y_scores.append(total_logit[:, 1:])

            probability = F.softmax(classification_logits, dim=1)
            prob = probability.cpu().data.numpy()
            y_prob.append(prob[:, 1:])

    y_true = np.concatenate(y_true).reshape(-1)
    y_scores = np.concatenate(y_scores).reshape(-1)
    y_prob = np.concatenate(y_prob).reshape(-1)

    order = np.argsort(-y_scores)
    y_prob_sorted = sorted(y_prob, reverse=True)
    with open("./data/order.pkl", 'wb') as output:
        pickle.dump(order, output)
    with open("./data/y_prob_sorted.pkl", 'wb') as output:
        pickle.dump(y_prob_sorted, output)

def calc_prec_recall_f1(y_actual, y_pred, none_id=0, eps=0.0001):
    """
    Calculates precision recall and F1 score

    Parameters
    ----------
    y_actual:	Actual labels
    y_pred:		Predicted labels
    none_id:	Identifier used for denoting NA relation

    Returns
    -------
    precision:	Overall precision
    recall:		Overall recall
    f1:		Overall f1
    """
    pos_pred, pos_gt, true_pos = 0.0, 0.0, 0.0

    for i in range(len(y_actual)):
        if y_actual[i] != none_id:
            pos_gt += 1.0

    for i in range(len(y_pred)):
        if y_pred[i] != none_id:
            pos_pred += 1.0
            if y_pred[i] == y_actual[i]:
                true_pos += 1.0

    precision = true_pos / (pos_pred + eps)
    recall = true_pos / (pos_gt + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    return precision, recall, f1

def split_bag(df_data, mode):
    df_data = df_data.reset_index(drop=True)
    print("Start splitting bag for testing...")
    grouped = df_data.groupby(['entity_pairs'])
    bag_idxinbag = {}
    for i, j in tqdm(grouped):
        if mode[-7:] == 'testone':
            if len(list(j.index.values)) > 1:
                bag_idxinbag[i] = list(j.index.values)
        elif mode[-11:] == 'testalldata':
            bag_idxinbag[i] = list(j.index.values)

    print("Finished !")
    return bag_idxinbag

def evaluate_results_bag_train(df_dev, net, test_loader, pad_id, cuda, args, mode):
    AUC = 0
    num_classes = args.num_classes
    y_true = []
    y_scores = []
    if mode[-7:] == 'devone':
        if os.path.isfile('./data/bag_idxinbag_devall.pkl'):
            with open('./data/bag_idxinbag_devall.pkl', 'rb') as pkl_file:
                bag_idxinbag = pickle.load(pkl_file)
        else:
            bag_idxinbag = split_bag(df_dev, mode=mode)
            with open('./data/bag_idxinbag_devall.pkl', 'wb') as output:
                pickle.dump(bag_idxinbag, output)
    if mode[-11:] == 'testalldata':
        if os.path.isfile('./data/bag_idxinbag_devalldata.pkl'):
            with open('./data/bag_idxinbag_devalldata.pkl', 'rb') as pkl_file:
                bag_idxinbag = pickle.load(pkl_file)
        else:
            bag_idxinbag = split_bag(df_dev, mode=mode)
            with open('./data/bag_idxinbag_devalldata.pkl', 'wb') as output:
                pickle.dump(bag_idxinbag, output)

    with torch.no_grad():
        test_logit, test_onehot = {}, {}
        for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
            x, e1_e2_start, labels, index = data
            attention_mask = (x != pad_id).float()
            token_type_ids = torch.zeros((x.shape[0], x.shape[1])).long()
            temp = np.zeros((len(labels), num_classes)).tolist()
            cnt = 0
            for r in labels:
                temp[cnt][r.item()] = 1
                cnt += 1
            total_y = np.array(temp)
            if cuda:
                x = x.cuda()
                labels = labels.cuda()
                attention_mask = attention_mask.cuda()
                token_type_ids = token_type_ids.cuda()

            classification_logits = net(x, token_type_ids=token_type_ids, attention_mask=attention_mask, Q=None, \
                                        e1_e2_start=e1_e2_start)

            probability = F.softmax(classification_logits, dim=1)

            target = labels.squeeze(1)
            one_hot = torch.zeros(probability.shape).cuda().scatter(1, torch.unsqueeze(target, dim=1), 1)

            for j, idx in enumerate(index):
                logits = classification_logits[j].cpu().data.numpy().tolist()
                test_logit[idx] = logits
                one_hots = one_hot[j].cpu().data.numpy().tolist()
                test_onehot[idx] = one_hots

        bag_idx_list = list(bag_idxinbag.values())
        print("Start selecting sample with max_logit in each bag...")
        for idx_list in tqdm(bag_idx_list):
            max_logit = []
            for idx in idx_list:
                max_logit.append(max(test_logit[idx]))
            max_num = np.argmax(max_logit)
            max_idx = idx_list[max_num]
            onehot = test_onehot[max_idx]
            logit = test_logit[max_idx]

            y_true.append(onehot[1:])
            y_scores.append(logit[1:])

        y_true = np.concatenate(y_true).reshape(-1)
        y_scores = np.concatenate(y_scores).reshape(-1)

    ################################ 用于entity-level (测P@N和AUC) ########################

    if mode[-11:] == 'testalldata':
        AUC = average_precision_score(y_true, y_scores)
        print('AUC: ', AUC)

    if mode[-7:] == 'testall' or mode[-7:] == 'testone':
        order = np.argsort(-y_scores)

        def p_score(n):
            corr_num = 0.0
            for i in order[:n]:
                corr_num += 1.0 if (y_true[i] == 1) else 0
            return corr_num / n

        print('P@100: ', p_score(100))
        print('P@200: ', p_score(200))
        print('P@300: ', p_score(300))
        print('P@500: ', p_score(500))
        print('P@1000: ', p_score(1000))
        print('P@2000: ', p_score(2000))
        print('P@3000: ', p_score(3000))
        print('P@4000: ', p_score(4000))
        print('P@5000: ', p_score(5000))
        print('P@6000: ', p_score(6000))
        print('P@7000: ', p_score(7000))
        print('P@8000: ', p_score(8000))

    return AUC

def evaluate_results_bag_test(df_test, net, test_loader, pad_id, cuda, args, mode):
    AUC = 0
    num_classes = args.num_classes
    y_true = []
    y_scores = []
    '''生成两种测试集的bag_idxinbag'''
    if mode[-7:] == 'testone':
        if os.path.isfile('./data/bag_idxinbag_testall.pkl'):
            with open('./data/bag_idxinbag_testall.pkl', 'rb') as pkl_file:
                bag_idxinbag = pickle.load(pkl_file)
        else:
            bag_idxinbag = split_bag(df_test, mode=mode)
            with open('./data/bag_idxinbag_testall.pkl', 'wb') as output:
                pickle.dump(bag_idxinbag, output)
    if mode[-11:] == 'testalldata':
        if os.path.isfile('./data/bag_idxinbag_testalldata.pkl'):
            with open('./data/bag_idxinbag_testalldata.pkl', 'rb') as pkl_file:
                bag_idxinbag = pickle.load(pkl_file)
        else:
            bag_idxinbag = split_bag(df_test, mode=mode)
            with open('./data/bag_idxinbag_testalldata.pkl', 'wb') as output:
                pickle.dump(bag_idxinbag, output)

    dir_path = "./data/{mode}_tmp_save".format(mode=mode)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    true_path = "./data/{mode}_tmp_save/y_true.pkl".format(mode=mode)
    scores_path = "./data/{mode}_tmp_save/y_scores.pkl".format(mode=mode)
    test_logit_path = "./data/{mode}_tmp_save/test_logit.pkl".format(mode=mode)
    test_onehot_path = "./data/{mode}_tmp_save/test_onehot.pkl".format(mode=mode)

    if os.path.isfile(test_logit_path) and os.path.isfile(test_onehot_path):
        with open(test_logit_path, 'rb') as pkl_file:
            test_logit = pickle.load(pkl_file)
        with open(test_onehot_path, 'rb') as pkl_file:
            test_onehot = pickle.load(pkl_file)

        bag_idx_list = list(bag_idxinbag.values())
        print("Start selecting sample with max_logit in each bag...")
        for idx_list in tqdm(bag_idx_list):
            max_logit = []
            for idx in idx_list:
                max_logit.append(max(test_logit[idx]))
            max_num = np.argmax(max_logit)
            max_idx = idx_list[max_num]
            onehot = test_onehot[max_idx]
            logit = test_logit[max_idx]

            y_true.append(onehot[1:])
            y_scores.append(logit[1:])

        y_true = np.concatenate(y_true).reshape(-1)
        y_scores = np.concatenate(y_scores).reshape(-1)

        with open(true_path, 'wb') as output:
            pickle.dump(y_true, output)
        with open(scores_path, 'wb') as output:
            pickle.dump(y_scores, output)

    else:
        with torch.no_grad():
            test_logit, test_onehot = {}, {}
            for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
                x, e1_e2_start, labels, index = data
                attention_mask = (x != pad_id).float()
                token_type_ids = torch.zeros((x.shape[0], x.shape[1])).long()
                temp = np.zeros((len(labels), num_classes)).tolist()
                cnt = 0
                for r in labels:
                    temp[cnt][r.item()] = 1
                    cnt += 1
                total_y = np.array(temp)
                if cuda:
                    x = x.cuda()
                    labels = labels.cuda()
                    attention_mask = attention_mask.cuda()
                    token_type_ids = token_type_ids.cuda()

                classification_logits = net(x, token_type_ids=token_type_ids, attention_mask=attention_mask, Q=None, \
                                            e1_e2_start=e1_e2_start)

                probability = F.softmax(classification_logits, dim=1)

                target = labels.squeeze(1)
                one_hot = torch.zeros(probability.shape).cuda().scatter(1, torch.unsqueeze(target, dim=1), 1)

                for j, idx in enumerate(index):
                    logits = classification_logits[j].cpu().data.numpy().tolist()
                    test_logit[idx] = logits
                    one_hots = one_hot[j].cpu().data.numpy().tolist()
                    test_onehot[idx] = one_hots

        with open(test_logit_path, 'wb') as output:
            pickle.dump(test_logit, output)
        with open(test_onehot_path, 'wb') as output:
            pickle.dump(test_onehot, output)

        bag_idx_list = list(bag_idxinbag.values())
        print("Start selecting sample with max_logit in each bag...")
        for idx_list in tqdm(bag_idx_list):
            max_logit = []
            for idx in idx_list:
                max_logit.append(max(test_logit[idx]))
            max_num = np.argmax(max_logit)
            max_idx = idx_list[max_num]
            onehot = test_onehot[max_idx]
            logit = test_logit[max_idx]

            y_true.append(onehot[1:])
            y_scores.append(logit[1:])

        y_true = np.concatenate(y_true).reshape(-1)
        y_scores = np.concatenate(y_scores).reshape(-1)

        with open(true_path, 'wb') as output:
            pickle.dump(y_true, output)
        with open(scores_path, 'wb') as output:
            pickle.dump(y_scores, output)

        print("Finished !")
    ################################ 用于entity-level (测P@N和AUC) ########################

    if mode[-11:] == 'testalldata':
        AUC = average_precision_score(y_true, y_scores)
        print('AUC: ', AUC)

    if mode[-7:] == 'testall' or mode[-7:] == 'testone':
        order = np.argsort(-y_scores)

        def p_score(n):
            corr_num = 0.0
            for i in order[:n]:
                corr_num += 1.0 if (y_true[i] == 1) else 0
            return corr_num / n

        print('P@100: ', p_score(100))
        print('P@200: ', p_score(200))
        print('P@300: ', p_score(300))
        print('P@500: ', p_score(500))
        print('P@1000: ', p_score(1000))
        print('P@2000: ', p_score(2000))
        print('P@3000: ', p_score(3000))
        print('P@4000: ', p_score(4000))
        print('P@5000: ', p_score(5000))
        print('P@6000: ', p_score(6000))
        print('P@7000: ', p_score(7000))
        print('P@8000: ', p_score(8000))

    #######################################################################################

    return AUC


def evaluate_results_bag(df_test, net, test_loader, pad_id, cuda, args, mode, train_or_test):
    net.eval()

    """基于包的训练阶段"""
    if train_or_test == 'train':
        AUC = evaluate_results_bag_train(df_test, net, test_loader, pad_id, cuda, args, mode)

    """基于包的测试阶段"""
    if train_or_test == 'test':
        AUC = evaluate_results_bag_test(df_test, net, test_loader, pad_id, cuda, args, mode)

    return AUC

