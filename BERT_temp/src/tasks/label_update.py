#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import torch.optim as optim
from .train_funcs import load_state
from ..misc import save_as_pickle, load_pickle
import logging
import torch.nn.functional as F
from tqdm import tqdm
import pickle

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)

def update_label(args, train_loader):

    if args.fp16:    
        from apex import amp
    else:
        amp = None
    
    cuda = torch.cuda.is_available()

    from ..model.BERT.modeling_bert import BertModel as Model

    model_name = 'BERT'

    net = Model.from_pretrained('./tmp/bert-base-uncased', force_download=False,
                                model_size=args.model_size,
                                task='classification',
                                n_classes_=args.num_classes)

    tokenizer = load_pickle("%s_tokenizer.pkl" % model_name)
    net.resize_token_embeddings(len(tokenizer))
    e1_id = tokenizer.convert_tokens_to_ids('[E1]')
    e2_id = tokenizer.convert_tokens_to_ids('[E2]')
    assert e1_id != e2_id != 1
    
    if cuda:
        net.cuda()
        
    logger.info("FREEZING MOST HIDDEN LAYERS...")
    if args.model_no == 0:
        unfrozen_layers = ["classifier", "pooler", "encoder.layer.11", \
                           "classification_layer", "blanks_linear", "lm_linear", "cls"]
    elif args.model_no == 1:
        unfrozen_layers = ["classifier", "pooler", "classification_layer",\
                           "blanks_linear", "lm_linear", "cls",\
                           "albert_layer_groups.0.albert_layers.0.ffn"]
        
    for name, param in net.named_parameters():
        if not any([layer in name for layer in unfrozen_layers]):
            param.requires_grad = False
        else:
            param.requires_grad = True

    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.Adam([{"params":net.parameters(), "lr": args.lr}])
    
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,4,6,8,12,15,18,20,22,\
                                                                      24,26,30], gamma=0.8)
    
    start_epoch, best_pred, amp_checkpoint = load_state(net, optimizer, scheduler, args, load_best=False)
    
    if (args.fp16) and (amp is not None):
        logger.info("Using fp16...")
        net, optimizer = amp.initialize(net, optimizer, opt_level='O2')
        if amp_checkpoint is not None:
            amp.load_state_dict(amp_checkpoint)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,4,6,8,12,15,18,20,22,\
                                                                          24,26,30], gamma=0.8)


    pad_id = tokenizer.pad_token_id
    logger.info("Starting training process...")
    pad_id = tokenizer.pad_token_id
    update_size = len(train_loader)//10

    logger.info("Loading from the best model...")
    checkpoint = torch.load('./base_model/best_model_param.pkl')
    net.load_state_dict(checkpoint['state_dict'])
    auc = checkpoint['best_AUC']

    generate_RLD(net, train_loader, pad_id, cuda, args, auc, save_softlabel_path='train_softlabel.pkl')

    logger.info("Finished updating soft labels!")

def generate_RLD(net, train_loader, pad_id, cuda, args, auc, save_softlabel_path):

    net.eval()

    logger.info("Loading soft labels...")
    if os.path.isfile("./data/" + save_softlabel_path):
        soft_label = load_pickle(save_softlabel_path)
    else:
        soft_label = {}
        save_as_pickle(save_softlabel_path, soft_label)

    with torch.no_grad():
        idx_prob = []
        soft_label, logit_label, onehot_label = {}, {}, {}
        for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            x, e1_e2_start, labels, index = data
            attention_mask = (x != pad_id).float()
            token_type_ids = torch.zeros((x.shape[0], x.shape[1])).long()
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

            golden = labels.squeeze(1)
            for j, idx in enumerate(index):
                prob = probability[j].cpu().data.numpy().tolist()
                idx_prob.append((idx, prob[golden[j]]))
                soft_label[idx] = prob
                logits = classification_logits[j].cpu().data.numpy().tolist()
                logit_label[idx] = logits
                one_hots = one_hot[j].cpu().data.numpy().tolist()
                onehot_label[idx] = one_hots

        def takeSecond(elem):
            return elem[1]
        idx_prob.sort(key=takeSecond, reverse=True)

        with open('./data/idx_prob.pkl', 'wb') as output:
            pickle.dump(idx_prob, output)

        with open('./data/train_logit.pkl', 'wb') as output:
            pickle.dump(logit_label, output)

        with open('./data/train_onehot.pkl', 'wb') as output:
            pickle.dump(onehot_label, output)

        with open('./data/train_softlabel.pkl', 'wb') as output:
            pickle.dump(soft_label, output)

def generate_new_logit(args, train_logit, df_train):
    num_classes = args.num_classes
    rou = args.rou
    import numpy as np
    new_logit = {}
    if os.path.isfile('./data/bag_idxinbag.pkl'):
        with open('./data/bag_idxinbag.pkl', 'rb') as pkl_file:
            bag_idxinbag = pickle.load(pkl_file)
    else:
        bag_idxinbag = split_bag(df_train)
    bag_idx_list = list(bag_idxinbag.values())
    print("Start generating new logit consists of bag info...")
    for idx_list in tqdm(bag_idx_list):
        bag_logit_tmp = np.zeros(num_classes)
        for idx in idx_list:
            bag_logit_tmp += np.array(train_logit[idx])
        bag_logit = bag_logit_tmp / len(idx_list)
        for idx in idx_list:
            tmp_logit = rou * bag_logit + (1-rou) * np.array(train_logit[idx])
            new_logit[idx] = list(tmp_logit)
    new_logit_path = './data/new_logit_rou{}.pkl'.format(args.rou)
    # print(new_logit_path)
    with open(new_logit_path, 'wb') as output:
        pickle.dump(new_logit, output)
    print("Finished !")
    return new_logit

def split_bag(df_data):
    df_data = df_data.reset_index(drop=True)
    print("Start splitting bag for training...")
    relation_fact_list = []
    entity_pair_list = []
    relations = df_data['relations'].tolist()
    sents = df_data['sents'].tolist()
    for idx, sent in enumerate(tqdm(sents)):
        e1 = [x for x in sent.split() if '[E1]' in x][0][4:-5]
        e2 = [x for x in sent.split() if '[E2]' in x][0][4:-5]
        entity_pair = e1+'###'+e2
        entity_pair_list.append(entity_pair)
        relation = relations[idx]
        relation_fact_list.append(entity_pair+'###'+relation)
    df_data['relation_fact'] = relation_fact_list
    grouped = df_data.groupby(['relation_fact'])
    bag_idxinbag = {}
    for i, j in tqdm(grouped):
        bag_idxinbag[i] = list(j.index.values)
    with open('./data/bag_idxinbag.pkl', 'wb') as output:
        pickle.dump(bag_idxinbag, output)
    print("Finished !")
    return bag_idxinbag
