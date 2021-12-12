#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from argparse import ArgumentParser
from plot import pr_curve
from src.tasks.tester import test
from src.tasks.preprocessing_funcs import load_dataloaders

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
    parser.add_argument("--use_pretrained_blanks", type=int, default=0, \
                        help="0: Don't use pre-trained blanks model, 1: use pre-trained blanks model")
    parser.add_argument("--num_classes", type=int, default=53, help='number of relation classes')
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--gradient_acc_steps", type=int, default=2, help="No. of steps of gradient accumulation")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipped gradient norm")
    parser.add_argument("--fp16", type=int, default=0, help="1: use mixed precision ; 0: use floating point 32")
    parser.add_argument("--num_epochs", type=int, default=10, help="No of epochs")
    parser.add_argument("--lr", type=float, default=0.00007, help="learning rate")
    parser.add_argument("--model_no", type=int, default=0, help='''Model ID: 0 - BERT\n
                                                                            1 - ALBERT''')
    parser.add_argument("--model_size", type=str, default='bert-base-uncased', help="For BERT: 'bert-base-uncased', \
                                                                                                'bert-large-uncased',\
                                                                                    For ALBERT: 'albert-base-v2',\
                                                                                                'albert-large-v2'")
    parser.add_argument("--model_tag", type=str, default='berttemp_bs16_lr7e-5_mu10.7_mu2-0.45_k300_rou0.5')
    args = parser.parse_args()

    df_train, train_loader, test_alldata_loader, train_len, df_test_alldata = load_dataloaders(args)

    test(args, test_alldata_loader, df_test_alldata)


