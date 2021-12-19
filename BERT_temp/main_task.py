#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from argparse import ArgumentParser
from src.tasks.preprocessing_funcs import load_dataloaders
from src.tasks.trainer_EM_mix import train_with_softlabel
import os
import pickle
import torch.nn.functional as F
import torch
from tqdm import tqdm
from src.tasks.label_update import update_label, generate_new_logit


logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--task", type=str, default='NYT')
    parser.add_argument("--train_data", type=str, default='./data/NYT/train.txt', help="training data .txt file path")
    parser.add_argument("--test_data", type=str, default='./data/NYT/test.txt', help="test data .txt file path")
    parser.add_argument("--rel2id", type=str, default='./data/NYT/relation2id.txt', help="rel2id .txt file path")
    parser.add_argument("--max_length", type=int, default=512, help="max length of input_ids")
    parser.add_argument("--use_pretrained_blanks", type=int, default=0,
                        help="0: Don't use pre-trained blanks model, 1: use pre-trained blanks model")
    parser.add_argument("--num_classes", type=int, default=53, help='number of relation classes')
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--gradient_acc_steps", type=int, default=2, help="No. of steps of gradient accumulation")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipped gradient norm")
    parser.add_argument("--fp16", type=int, default=0, help="1: use mixed precision ; 0: use floating point 32")
    parser.add_argument("--num_epochs", type=int, default=20, help="No of epochs")
    parser.add_argument("--lr", type=float, default=7e-5, help="learning rate")
    parser.add_argument("--model_no", type=int, default=0, help='Model ID: 0 - BERT')
    parser.add_argument("--model_size", type=str, default='bert-base-uncased')
    parser.add_argument("--model_tag", type=str, default='berttemp_bs16_lr7e-5_mu10.7_mu2-0.45_k300_rou0.5')
    parser.add_argument("--mu1", type=float, default=0.7)
    parser.add_argument("--mu2", type=float, default=0.45)
    parser.add_argument("--k", type=int, default=300)
    parser.add_argument("--rou", type=float, default=0.5)
    args = parser.parse_args()
    print(args)

    df_train, df_test, df_dev, train_loader, test_loader, dev_loader, train_len = load_dataloaders(args)

    if not os.path.isfile('./data/idx_prob.pkl'):
        update_label(args, train_loader)

    if os.path.isfile('./data/idx_prob.pkl'):
        with open('./data/idx_prob.pkl', 'rb') as pkl_file:
            idx_prob = pickle.load(pkl_file)

    if os.path.isfile('./data/train_logit.pkl'):
        with open('./data/train_logit.pkl', 'rb') as pkl_file:
            train_logit = pickle.load(pkl_file)

    if os.path.isfile('./data/train_onehot.pkl'):
        with open('./data/train_onehot.pkl', 'rb') as pkl_file:
            onehot = pickle.load(pkl_file)

    new_logit_path = './data/new_logit_rou{rou}.pkl'.format(rou=args.rou)
    print(new_logit_path)
    if os.path.isfile(new_logit_path):
        with open(new_logit_path, 'rb') as pkl_file:
            new_logit = pickle.load(pkl_file)
    else:
        new_logit = generate_new_logit(args, train_logit, df_train)

    file_path = './data/T_dict_mu1{mu1}_mu2{mu2}_k{k}_rou{rou}.pkl'.format(mu1=args.mu1, mu2=args.mu2, k=args.k, rou=args.rou)
    print(file_path)
    if not os.path.isfile(file_path):
        entropy_list = []
        T_dict = {}
        mu1 = args.mu1
        mu2 = args.mu2
        k = args.rou
        for i in tqdm(range(len(idx_prob))):
            prob_gold = F.softmax(torch.tensor(new_logit[i]), 0)
            entropy = - torch.sum(prob_gold * torch.log(prob_gold))
            T1 = 1. + mu1 * torch.exp(-k*entropy)
            T2 = 1. + mu2 * torch.exp(-k*entropy)
            logits = torch.tensor(new_logit[i]) / T1
            teach_prob = F.softmax(logits, 0)
            T_dict[i] = [teach_prob, 1.0 / T2]
        with open(file_path, 'wb') as output:
            pickle.dump(T_dict, output)

    train_with_softlabel(args, df_dev, train_loader, dev_loader, train_len)
