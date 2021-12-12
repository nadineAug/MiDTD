#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from src.tasks.trainer import train_and_dev
from src.tasks.tester import test
import logging
from argparse import ArgumentParser
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
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--gradient_acc_steps", type=int, default=2, help="No. of steps of gradient accumulation")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipped gradient norm")
    parser.add_argument("--fp16", type=int, default=0, help="1: use mixed precision ; 0: use floating point 32") # mixed precision doesn't seem to train well
    parser.add_argument("--num_epochs", type=int, default=20, help="No of epochs")
    parser.add_argument("--lr", type=float, default=0.00007, help="learning rate")
    parser.add_argument("--model_no", type=int, default=0, help='''Model ID: 0 - BERT''')
    parser.add_argument("--model_size", type=str, default='bert-base-uncased')
    parser.add_argument("--tag", type=str, default='bert_base')
    args = parser.parse_args()

    train_loader, dev_loader, test_loader, train_len, dev_len, test_length = load_dataloaders(args)

    train_and_dev(args, train_loader, dev_loader, test_loader, train_len, dev_len)

