#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from .train_funcs import load_state, evaluate_results_bag
from ..misc import load_pickle
import logging


logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)

def test(args, test_alldata_loader, df_test_alldata):
    if args.fp16:    
        from apex import amp
    else:
        amp = None
    
    cuda = torch.cuda.is_available()

    if args.model_no == 0:
        from ..model.BERT.modeling_bert import BertModel as Model
        model = args.model_size
        lower_case = True
        model_name = 'BERT'

    net = Model.from_pretrained('./tmp/bert-base-uncased', force_download=False, \
                                model_size=args.model_size,
                                task='classification' if args.task != 'fewrel' else 'fewrel', \
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
            print("[FROZE]: %s" % name)
            param.requires_grad = False
        else:
            print("[FREE]: %s" % name)
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
    mask_id = tokenizer.mask_token_id

    print('#'*10)
    logger.info("Starting testing process...")

    ckp_path = './model/bmp_CTD_Sdown_Tup_entropynew_mu10.7_mu20.45_k300_rou0.5_lr7e-5.pkl'
    checkpoint = torch.load(ckp_path)
    print('checkpoint path:', ckp_path)

    net.load_state_dict(checkpoint['state_dict'])
    logger.info("Loaded from the best model!")

    logger.info("Evaluating P@N...")
    _ = evaluate_results_bag(df_test_alldata, net, test_alldata_loader, pad_id, cuda, args, mode='max_base_bert_testone', train_or_test='test')

    logger.info("Evaluating AUC...")
    _ = evaluate_results_bag(df_test_alldata, net, test_alldata_loader, pad_id, cuda, args, mode='max_mu10.7_mu20.45_k300_rou0.5_lr7e-5_testalldata', train_or_test='test')




