#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from .train_funcs import load_state, load_results, evaluate_, evaluate_results_bag
from ..misc import save_as_pickle, load_pickle
import time
import logging
import torch.nn.functional as F
import pickle

from torch.autograd import Variable

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)

def train_with_softlabel(args, train_loader, test_alldata_loader, train_len, df_test_alldata):

    if args.fp16:    
        from apex import amp
    else:
        amp = None
    
    cuda = torch.cuda.is_available()

    logger.info("Loaded %d Training samples." % train_len)

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
    
    losses_per_epoch, accuracy_per_epoch, auc_per_epoch = load_results(args.model_no)
    
    pad_id = tokenizer.pad_token_id
    mask_id = tokenizer.mask_token_id
    update_size = len(train_loader)//10

    file_path = './data/T_dict_mu1{mu1}_mu2{mu2}_k{k}_rou{rou}.pkl'.format(mu1=args.mu1, mu2=args.mu2, k=args.k,
                                                                           rou=args.rou)
    if os.path.isfile(file_path):
        with open(file_path, 'rb') as pkl_file:
            soft_label = pickle.load(pkl_file)

    logger.info("Starting training process...")
    for epoch in range(start_epoch, args.num_epochs):
        start_time = time.time()
        net.train(); total_loss = 0.0; losses_per_batch = []; total_acc = 0.0; accuracy_per_batch = []
        for i, data in enumerate(train_loader, 0):
            x, e1_e2_start, labels, index = data
            attention_mask = (x != pad_id).float()
            token_type_ids = torch.zeros((x.shape[0], x.shape[1])).long()

            if cuda:
                x = x.cuda()
                labels = labels.cuda()
                attention_mask = attention_mask.cuda()
                token_type_ids = token_type_ids.cuda()
                
            classification_logits = net(x, token_type_ids=token_type_ids, attention_mask=attention_mask, Q=None,\
                          e1_e2_start=e1_e2_start)


            """CTD_Sdown/_Sdown_Tup """
            target = labels.squeeze(1)
            probability = F.softmax(classification_logits, dim=1)
            one_hot = torch.zeros(probability.shape).cuda().scatter(1, torch.unsqueeze(target, dim=1), 1)
            one_hot = Variable(one_hot, requires_grad=False)
            p_log = torch.log(probability)
            tmp = one_hot * p_log
            loss_1 = - torch.sum(tmp, (0, 1)) / tmp.size(0)

            kplusone_SLD = []
            temp = []
            for idx in index:
                kplusone_SLD.append(soft_label[idx][0])
                temp.append(soft_label[idx][1])

            kplusone_SLD_t = torch.stack(kplusone_SLD, dim=0).cuda()
            temp = torch.tensor(temp).cuda()

            logit_diag = torch.matmul(torch.diag(temp), classification_logits)
            p_student_T = F.softmax(logit_diag, dim=1)
            p_student_T_log = torch.log(p_student_T)
            tmp = kplusone_SLD_t * p_student_T_log
            loss_2 = - torch.sum(tmp, (0, 1)) / tmp.size(0)

            alpha = 0.8
            loss = (1 - alpha) * loss_1 + alpha * loss_2


            loss = loss/args.gradient_acc_steps
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            
            if args.fp16:
                grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_norm)
            else:
                grad_norm = clip_grad_norm_(net.parameters(), args.max_norm)
            
            if (i % args.gradient_acc_steps) == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item()
            total_acc += evaluate_(classification_logits, labels, \
                                   ignore_idx=-1)[0]
            
            if (i % update_size) == (update_size - 1):
                losses_per_batch.append(args.gradient_acc_steps*total_loss/update_size)
                accuracy_per_batch.append(total_acc/update_size)
                print('[Epoch: %d, %5d/ %d points] total loss, accuracy per batch: %.3f, %.3f' %
                      (epoch + 1, (i + 1)*args.batch_size, train_len, losses_per_batch[-1], accuracy_per_batch[-1]))
                total_loss = 0.0; total_acc = 0.0
        
        scheduler.step()

        if (epoch % 1) == 0:

            print("###################################")
            print("#            TEST STAGE           #")
            print("###################################")

            logger.info("Evaluating AUC...")
            AUC = evaluate_results_bag(df_test_alldata, net, test_alldata_loader, pad_id, cuda, args, mode='testalldata', train_or_test='train')

            auc_per_epoch.append(AUC)
            losses_per_epoch.append(sum(losses_per_batch)/len(losses_per_batch))
            accuracy_per_epoch.append(sum(accuracy_per_batch)/len(accuracy_per_batch))
            print("Epoch finished, took %.2f seconds." % (time.time() - start_time))
            print("Losses at Epoch %d: %.7f" % (epoch + 1, losses_per_epoch[-1]))
            print("Train accuracy at Epoch %d: %.7f" % (epoch + 1, accuracy_per_epoch[-1]))
            print("Dev AUC at Epoch %d: %.7f" % (epoch + 1, auc_per_epoch[-1]))

            save_as_pickle("task_test_losses_per_epoch.pkl", losses_per_epoch)
            save_as_pickle("task_train_accuracy_per_epoch.pkl", accuracy_per_epoch)
            save_as_pickle("task_test_auc_per_epoch.pkl", auc_per_epoch)
            torch.save({
                    'epoch': epoch + 1,
                    'state_dict': net.state_dict(),
                    'best_AUC': auc_per_epoch[-1],
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'amp': amp.state_dict() if amp is not None else amp
                }, os.path.join("./data/checkpoint", "task_test_checkpoint_{}.pth.tar".format(args.model_tag)))
            logger.info("Finished Storing Checkpoint!")

            if auc_per_epoch[-1] > best_pred:
                best_pred = auc_per_epoch[-1]
                best_epoch = epoch + 1
                logger.info("Storing the better model...")
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': net.state_dict(),
                    'best_AUC': auc_per_epoch[-1],
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'amp': amp.state_dict() if amp is not None else amp
                }, './model/bmp_{}.pkl'.format(args.model_tag))
                logger.info("Finished Storing Better Model Parameters From Epoch {num}!".format(num=epoch+1))

    print('best auc:', best_pred, 'best epoch:', best_epoch)
    logger.info("Finished Training!")

