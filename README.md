# MiDTD: A Simple and Effective Distillation Framework for Distantly Supervised Relation Extraction

## Overview
A PyTorch implementation of the models for the paper "MiDTD: A Simple and Effective Distillation Framework for Distantly Supervised Relation Extraction". We only show the implementation corresponding to dataset NYT-10.

## Requirements
Requirements: Python (3.6+), PyTorch (1.2.0+)
Pre-trained BERT courtesy of HuggingFace.co (https://huggingface.co)

## Training & Testing Teacher Network
Run main_task.py in ./BERT_base with arguments below to train the teacher network.

```python 
python main_task.py [--train_data TRAIN_DATA]
                    [--test_data TEST_DATA]
                    [--rel2id REL_PATH]
                    [--max_length MAX_LEN]
                    [--num_classes NUM_CLASSES] 
                    [--batch_size BATCH_SIZE]
                    [--num_epochs NUM_EPOCHS]
                    [--lr LR] 
                    [--model_tag TAG]                    
```

Run test_task.py in ./BERT_base to train the teacher network.

```python 
python test_task.py
```


## Training & Testing Student Network
Run main_task.py in ./BERT_temp with arguments below to train the student network.

```python
python main_task.py [--train_data TRAIN_DATA]
                    [--test_data TEST_DATA]
                    [--rel2id REL_PATH]
                    [--max_length MAX_LEN]
                    [--num_classes NUM_CLASSES] 
                    [--batch_size BATCH_SIZE]
                    [--num_epochs NUM_EPOCHS]
                    [--lr LR] 
                    [--model_tag TAG]
                    [--mu1 MU1_VALUE]
                    [--mu2 MU2_VALUE]
                    [--k K_VALUE]
                    [--rou ROU_VALUE]                    
```

Run test_task.py in ./BERT_temp to train the student network.

```python 
python test_task.py
```
