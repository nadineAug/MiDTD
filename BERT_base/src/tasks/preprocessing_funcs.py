#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import random
import copy
import pandas as pd
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from ..misc import save_as_pickle, load_pickle
from tqdm import tqdm
import logging
from random import shuffle

tqdm.pandas(desc="prog_bar")
logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

def process_NYT_text(text, args):
    rm = Relations_Mapper(args)
    rel_list = list(rm.rel2idx)
    sents, relations = [], []
    entity_pairs = []
    for i, line in enumerate(text):
        items = line.split('\t')
        e1 = items[2]
        e2 = items[3]
        entity_pair = e1+'###'+e2
        rel = items[4]
        if rel not in rel_list:
            rel = 'NA'

        if items[5][-11:] == ' ###END###\n':
            sent = items[5][:-11]
        else:
            sent = items[5]

        for e in (e1, e2):
            sen_tmp = re.sub("_", " ", sent)
            e_tmp = re.sub("_", " ", e)
            if len(re.findall("_", e)) > 1 and len(re.findall(e_tmp, sen_tmp)) != 0:
                pos = sen_tmp.find(e_tmp)
                if e == e1:
                    sent = sent[:pos] + e1 + sent[(pos + len(e1)):]
                else:
                    sent = sent[:pos] + e2 + sent[(pos + len(e2)):]
                break

        for e in (e1, e2):
            if len(re.findall(" " + e + " ", sent)) != 0:
                p = sent.find(" " + e + " ")
                if e == e1:
                    sent = sent[:p] + " [E1]" + e1 + "[/E1]" + sent[(p + len(e1) + 1):]
                else:
                    sent = sent[:p] + " [E2]" + e2 + "[/E2]" + sent[(p + len(e2) + 1):]
            elif items[5][:len(e)] == e:
                if e == e1:
                    sent = "[E1]" + e + "[/E1]" + sent[len(e):]
                else:
                    sent = "[E2]" + e + "[/E2]" + sent[len(e):]
            else:
                print("--------- ERROR:e1????????????????????????????????? {}".format(i+1))
                print(e1)
                print(e2)
                print(items[5])
                break
            if e == e2:
                sents.append(sent)
                relations.append(rel)
                entity_pairs.append(entity_pair)

    return sents, relations, entity_pairs

def data_split(full_list, ratio, random_seed, shuffle=False):
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle:
        random.seed(random_seed)
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1, sublist_2

def preprocess_NYT(args):
    '''
    Data preprocessing for NYT dataset
    '''
    data_path = args.train_data
    logger.info("Reading training file %s..." % data_path)
    with open(data_path, 'r', encoding='utf8') as f:
        text = f.readlines()

    dev_text, train_text = data_split(text, ratio=0.3, random_seed=1, shuffle=True)

    sents, relations, _ = process_NYT_text(train_text, args)
    df_train = pd.DataFrame(data={'sents': sents, 'relations': relations})

    sents, relations, _ = process_NYT_text(dev_text, args)
    df_dev = pd.DataFrame(data={'sents': sents, 'relations': relations})

    data_path = args.test_data
    logger.info("Reading test file %s..." % data_path)
    with open(data_path, 'r', encoding='utf8') as f:
        text = f.readlines()
    sents, relations, entity_pairs = process_NYT_text(text, args)
    df_test = pd.DataFrame(data={'sents': sents, 'relations': relations, 'entity_pairs': entity_pairs})

    rm = Relations_Mapper(args)
    save_as_pickle('relations.pkl', rm)

    df_train['relations_id'] = df_train.progress_apply(lambda x: rm.rel2idx[x['relations']], axis=1)
    df_dev['relations_id'] = df_dev.progress_apply(lambda x: rm.rel2idx[x['relations']], axis=1)
    df_test['relations_id'] = df_test.progress_apply(lambda x: rm.rel2idx[x['relations']], axis=1)

    save_as_pickle('df_train.pkl', df_train)
    save_as_pickle('df_dev.pkl', df_dev)
    save_as_pickle('df_test.pkl', df_test)

    logger.info("Finished and saved!")

    return df_train, df_dev, df_test, rm


class Relations_Mapper(object):
    def __init__(self, args):
        self.rel2idx = {}
        self.idx2rel = {}
        
        logger.info("Mapping relations to IDs...")

        rel2id_path = args.rel2id
        logger.info("Reading rel2id file %s..." % rel2id_path)
        with open(rel2id_path, 'r', encoding='utf8') as f:
            text = f.readlines()

        self.n_classes = 0
        for line in text:
            item = line.split()
            rel = item[0]
            index = item[1]
            self.rel2idx[rel] = int(index)
            self.n_classes += 1

        for key, value in self.rel2idx.items():
            self.idx2rel[value] = key


class Pad_Sequence():
    """
    collate_fn for dataloader to collate sequences of different lengths into a fixed length batch
    Returns padded x sequence, y sequence, x lengths and y lengths of batch
    """
    def __init__(self, seq_pad_value, label_pad_value=-1, label2_pad_value=-1,\
                 ):
        self.seq_pad_value = seq_pad_value
        self.label_pad_value = label_pad_value
        self.label2_pad_value = label2_pad_value
        
    def __call__(self, batch):
        sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
        seqs = [x[0] for x in sorted_batch]
        seqs_padded = pad_sequence(seqs, batch_first=True, padding_value=self.seq_pad_value)
        x_lengths = torch.LongTensor([len(x) for x in seqs])
        
        labels = list(map(lambda x: x[1], sorted_batch))
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=self.label_pad_value)
        y_lengths = torch.LongTensor([len(x) for x in labels])
        
        labels2 = list(map(lambda x: x[2], sorted_batch))
        labels2_padded = pad_sequence(labels2, batch_first=True, padding_value=self.label2_pad_value)
        y2_lengths = torch.LongTensor([len(x) for x in labels2])
        
        return seqs_padded, labels_padded, labels2_padded, \
                x_lengths, y_lengths, y2_lengths

def get_e1e2_start(x, e1_id, e2_id):
    try:
        e1_e2_start = ([i for i, e in enumerate(x) if e == e1_id][0],\
                        [i for i, e in enumerate(x) if e == e2_id][0])
    except Exception as e:
        e1_e2_start = None
        print("-----------INFO: Drop out this sample(???????????????????????????????????????????????????")
    return e1_e2_start

# add
class NYT_dataset(Dataset):
    def __init__(self, df, tokenizer, e1_id, e2_id, args, save_input_path, save_e1e2_path):
        self.e1_id = e1_id
        self.e2_id = e2_id
        self.df = df
        logger.info("Tokenizing data...")

        if os.path.isfile("./data/" + save_input_path):
            save_input = load_pickle(save_input_path)
            self.df['input'] = save_input
            logger.info("Loaded data save_input for training.")
        else:
            self.df['input'] = self.df.progress_apply(lambda x: tokenizer.encode(x['sents'], max_length=args.max_length), axis=1)
            save_as_pickle(save_input_path, self.df['input'])

        if os.path.isfile("./data/" + save_e1e2_path):
            save_e1e2 = load_pickle(save_e1e2_path)
            self.df['e1_e2_start'] = save_e1e2
            logger.info("Loaded data save_e1e2 for training.")
        else:
            self.df['e1_e2_start'] = self.df.progress_apply(lambda x: get_e1e2_start(x['input'], e1_id=self.e1_id, \
                                                                                 e2_id=self.e2_id), axis=1)
            save_as_pickle(save_e1e2_path, self.df['e1_e2_start'])
            logger.info("Saved data for training.")

        print("\nInvalid rows/total: %d/%d" % (df['e1_e2_start'].isnull().sum(), len(df)))
        self.df.dropna(axis=0, inplace=True)

    def __len__(self,):
        return len(self.df)

    def __getitem__(self, idx):
        return torch.LongTensor(self.df.iloc[idx]['input']),\
                torch.LongTensor(self.df.iloc[idx]['e1_e2_start']),\
                torch.LongTensor([self.df.iloc[idx]['relations_id']])


def load_dataloaders(args):
    if args.model_no == 0:
        from ..model.BERT.tokenization_bert import BertTokenizer as Tokenizer
        model = args.model_size
        lower_case = True
        model_name = 'BERT'

        
    if os.path.isfile("./data/%s_tokenizer.pkl" % model_name):
        tokenizer = load_pickle("%s_tokenizer.pkl" % model_name)
        logger.info("Loaded tokenizer from pre-trained blanks model")
    else:
        logger.info("Pre-trained blanks tokenizer not found, initializing new tokenizer...")
        tokenizer = Tokenizer.from_pretrained(model, do_lower_case=False)
        tokenizer.add_tokens(['[E1]', '[/E1]', '[E2]', '[/E2]', '[BLANK]'])

        save_as_pickle("%s_tokenizer.pkl" % model_name, tokenizer)
        logger.info("Saved %s tokenizer at ./data/%s_tokenizer.pkl" %(model_name, model_name))

    e1_id = tokenizer.convert_tokens_to_ids('[E1]')
    e2_id = tokenizer.convert_tokens_to_ids('[E2]')
    assert e1_id != e2_id != 1

    if args.task == 'NYT':
        relations_path = './data/relations.pkl'
        train_path = './data/df_train.pkl'
        dev_path = './data/df_dev.pkl'
        test_path = './data/df_test.pkl'

        if os.path.isfile(relations_path) and os.path.isfile(train_path) and os.path.isfile(dev_path) and os.path.isfile(test_path):
            rm = load_pickle('relations.pkl')
            df_train = load_pickle('df_train.pkl')
            df_dev = load_pickle('df_dev.pkl')
            df_test = load_pickle('df_test.pkl')
        else:
            df_train, df_dev, df_test, rm = preprocess_NYT(args)

        train_set = NYT_dataset(df_train, tokenizer=tokenizer, e1_id=e1_id, e2_id=e2_id, args=args, \
                                save_input_path='save_train_input.pkl', save_e1e2_path='save_train_e1e2.pkl')

        dev_set = NYT_dataset(df_dev, tokenizer=tokenizer, e1_id=e1_id, e2_id=e2_id, args=args, \
                                save_input_path='save_dev_input.pkl', save_e1e2_path='save_dev_e1e2.pkl')

        test_set = NYT_dataset(df_test, tokenizer=tokenizer, e1_id=e1_id, e2_id=e2_id, args=args, \
                               save_input_path='save_test_input.pkl', save_e1e2_path='save_test_e1e2.pkl')

        train_length = len(train_set)
        dev_length = len(dev_set)
        test_length = len(test_set)

        PS = Pad_Sequence(seq_pad_value=tokenizer.pad_token_id, \
                          label_pad_value=tokenizer.pad_token_id, \
                          label2_pad_value=-1)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, \
                                  num_workers=0, collate_fn=PS, pin_memory=False)
        dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=True, \
                                  num_workers=0, collate_fn=PS, pin_memory=False)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, \
                                 num_workers=0, collate_fn=PS, pin_memory=False)

    return train_loader, dev_loader, test_loader, train_length, dev_length, test_length

