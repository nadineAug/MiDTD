#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
from ..misc import load_pickle
import logging
from tqdm import tqdm
from sklearn.metrics import average_precision_score # add
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
    checkpoint_path = os.path.join(base_path,"task_test_checkpoint_{}.pth.tar".format(args.tag))
    start_epoch, best_pred, checkpoint = 0, 0, None
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        logger.info("Loaded checkpoint model.")
    if checkpoint != None:
        start_epoch = checkpoint['epoch']
        best_pred = checkpoint['best_AUC']
        net.load_state_dict(checkpoint['state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None:
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
    ### ignore index 0 (padding) when calculating accuracy
    idxs = (labels != ignore_idx).squeeze()
    o_labels = torch.softmax(output, dim=1).max(1)[1]
    l = labels.squeeze()[idxs]; o = o_labels[idxs]

    if len(idxs) > 1:
        acc = (l == o).sum().item()/len(idxs)
    else:
        acc = (l == o).sum().item()
    l = l.cpu().numpy().tolist() if l.is_cuda else l.numpy().tolist()
    o = o.cpu().numpy().tolist() if o.is_cuda else o.numpy().tolist()

    return acc, (o, l)

def evaluate_results(net, test_loader, pad_id, cuda, args, mode="val"):
    logger.info("Evaluating samples...")
    AUC=0
    num_classes = args.num_classes
    y_true = []; y_scores = []
    y_prob = []

    golden_label, pre_label = [], []

    net.eval()

    dir_path = "./data/{mode}_tmp_save".format(mode=mode)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    true_path = "./data/{mode}_tmp_save/y_true.pkl".format(mode=mode)
    scores_path = "./data/{mode}_tmp_save/y_scores.pkl".format(mode=mode)
    prob_path = "./data/{mode}_tmp_save/y_prob.pkl".format(mode=mode)
    golden_label_path = "./data/{mode}_tmp_save/golden_label.pkl".format(mode=mode)
    pre_label_path = "./data/{mode}_tmp_save/pre_label.pkl".format(mode=mode)

    if os.path.isfile(true_path) and os.path.isfile(scores_path) and os.path.isfile(prob_path) and os.path.isfile(golden_label_path) and os.path.isfile(pre_label_path):
        with open(true_path, 'rb') as pkl_file:
            y_true = pickle.load(pkl_file)
        with open(scores_path, 'rb') as pkl_file:
            y_scores = pickle.load(pkl_file)
        with open(prob_path, 'rb') as pkl_file:
            y_prob = pickle.load(pkl_file)
        with open(golden_label_path, 'rb') as pkl_file:
            golden_label = pickle.load(pkl_file)
        with open(pre_label_path, 'rb') as pkl_file:
            pre_label = pickle.load(pkl_file)
    else:
        with torch.no_grad():
            for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
                x, e1_e2_start, labels, _,_,_ = data
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

                golden_label += labels.cpu().data.numpy().squeeze().tolist()
                pre_label += np.argmax(total_logit, axis=1).tolist()

        y_true = np.concatenate(y_true).reshape(-1)
        y_scores = np.concatenate(y_scores).reshape(-1)

        y_prob = np.concatenate(y_prob).reshape(-1)

        with open(true_path, 'wb') as output:
            pickle.dump(y_true, output)
        with open(scores_path, 'wb') as output:
            pickle.dump(y_scores, output)
        with open(prob_path, 'wb') as output:
            pickle.dump(y_prob, output)
        with open(golden_label_path, 'wb') as output:
            pickle.dump(golden_label, output)
        with open(pre_label_path, 'wb') as output:
            pickle.dump(pre_label, output)

    if mode[-4:] == 'test':
        AUC = average_precision_score(y_true, y_scores)
        print('AUC: ', AUC)

    if mode[-7:] == 'testall':
        order = np.argsort(-y_prob)
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
