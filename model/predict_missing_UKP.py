"""
Author: Jan Wira Gotama Putra

Prediction of sentence links, given a partial gold tree (in UKP data), we try to predict "missing links".
Definition: non-AC nodes (in gold standard annotation of UKP) that could be AC nodes in our scheme, and get the links
"""
from typing import *
from tqdm import tqdm
import time
import argparse
import ast
import itertools
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.iterators import DataIterator

from model.datasetreader import * 
from model.model_functions import *
from model.Nets.BiaffineMTL import *

from sklearn.metrics import classification_report, confusion_matrix
from model.predictors import *
import pickle
from preprocessing.common_functions import list_files_in_dir, open_essays
from model.bert_labelling.labelling_helper import BERTRelLabeller
from preprocessing.discourseunit import NO_REL_SYMBOL, NO_TARGET_CONSTANT

flatten_list = lambda l: [item for sublist in l for item in sublist]


def list_directory(path) -> List[str]:
    """
    List directory existing in path
    """
    return [ os.path.join(path, name) for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)) ]


def save_output(save_dir, filename, content):
    """
    Args:
        save_dir (str): path to directory
        filename (str)
        content (Any)
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(save_dir+filename, 'w+') as f:
        f.write(str(content))
        # to read, use ast.literal_eval()
    f.close()
    print("Output successfully saved to", save_dir+filename)


def convert_linking_prediction_to_heuristic_baseline(test_preds):
    """
    Heuristic baseline prediction for linking
    """
    for i in range(len(test_preds)):
        for j in range(len(test_preds[i])):
            if j == 0:
                test_preds[i][j] = 0
            else:
                test_preds[i][j] = -1


if __name__ == "__main__":
    # user arguments
    parser = argparse.ArgumentParser(description='Prediction: linking experiment')
    parser.add_argument(
        '-test_dir', '--test_dir', type=str, help='dataset directory (test data)', required=True)
    parser.add_argument(
        '-tsv_dir', '--tsv_dir', type=str, help='dataset directory (tsv data)', required=True)
    parser.add_argument(
        '-model_dir', '--model_dir', type=str, help='model directory (containing many models)', required=True)
    # parser.add_argument(
    #     '-pred_dir', '--pred_dir', type=str, help='directory to save the prediction result', required=True)
    parser.add_argument(
        '-save_new_tsv_dir', '--save_new_tsv_dir', type=str, help='directory to save the prediction result in tsv', required=True)
    args = parser.parse_args()

    # device
    cuda_device = 0 if torch.cuda.is_available() else -1
    torch_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("CUDA device = "+str(cuda_device)+" (running on "+str(torch_device)+")")
    available_gpus = np.arange(torch.cuda.device_count()).tolist()
    print("Available GPUs "+str(available_gpus))

    # model
    model_dirs = list_directory(args.model_dir)
    print("%d models to test" % (len(model_dirs)))
    model_dirs.sort()

    # Fixed configuration
    general_config = Config(
        use_percentage=1.0, # how much data to use
        batch_size=1, # for prediction, this does not matter
    )

    # dataset reader
    reader = SeqDatasetReader(
        use_percentage=general_config.use_percentage,
    )

    # load test data (CPU only)
    if torch_device.type=="cpu":
        test_ds = reader.read(args.test_dir)
        print("# Test data size", len(test_ds))
        print()


    # tsv data
    files = list_files_in_dir(args.tsv_dir)
    files.sort()
    gold_essays = open_essays(files)

    # relation labelling model
    rel_labeller = BERTRelLabeller("model/bert_labelling/run-11/", torch_device, cuda_device)

    # iterate over models
    for model_dir in model_dirs:
        # model configuration
        model_config = get_model_config(model_dir+"/")
        print("Model architecture:", model_config.architecture)
        prediction_namespaces = ["rel_distances_labels"] # + model_config.aux_tasks; only rel distance

        # loading test data inside the loop. I am not sure why but this is more memory-friendly for GPU (when run)
        if torch_device.type=="cuda":
            test_ds = reader.read(args.test_dir)
            print("# Test data size", len(test_ds))
            print()

        # loading model
        model = load_model(model_dir+"/", torch_device)
        print("Model config", model.config)
        model.to(torch_device)

        # iterator
        iterator = BasicIterator(batch_size=general_config.batch_size)
        iterator.index_with(model.vocab) # this is a must for consistency reason

        # prediction
        predictor = PredictorUKPMissingLink(model, iterator, cuda_device=cuda_device)
        for task in prediction_namespaces:
            print("Task:", task)
            test_preds, gold_preds = predictor.predict(test_ds, namespace=task, coerce_tree=True)

            # to judge the model's quality
            if task == "rel_distances_labels":
                print("The ratio of tree from argmax prediction only %.3lf" % (predictor.tree_from_argmax / len(test_ds)) )
                predictor._reset_tree_argmax_count()
        
            # subdir = model_dir.split("/")[-1]
            # save_output(args.pred_dir + subdir + "/", task + "_pred.txt", test_preds)
            # save_output(args.pred_dir + subdir + "/", task + "_gold.txt", gold_preds)

            # save the tsv with augmented link
            for it in range(len(gold_essays)):
                print(gold_essays[it].essay_code)
                # print("gold links:", gold_preds[it])
                # print("pred links:", test_preds[it])
                for s in range(len(gold_preds[it])): 
                    if gold_preds[it][s] != 0: # if AC in gold, should be AC in test_preds
                        if test_preds[it][s]!=gold_preds[it][s]:
                            print("need investigation", gold_essays[it].essay_code)
                            input()

                    if gold_preds[it][s] != test_preds[it][s]: # this particular non-AC in gold has some possible connection
                        # predict the relation label from this particular node
                        source_text = [gold_essays[it].units[s].text]
                        target_text = [gold_essays[it].units[s + int(test_preds[it][s])].text]
                        pred_rel_label = rel_labeller.predict(source_text, target_text)[0]

                        # if detail, then we accept; else just leave it as non-AC
                        if pred_rel_label == "det": 
                            gold_essays[it].units[s].rel_name = "recovered"
                            gold_essays[it].units[s].targetID = gold_essays[it].units[s + int(test_preds[it][s])].ID
                            gold_essays[it].units[s].dropping = False

                # further check; if the target node of a particular node is non-AC; then it is non-AC as well
                relation_removal_exist = True
                while relation_removal_exist:
                    relation_removal_exist = False
                    for s in range(len(gold_essays[it].units)): 
                        if gold_essays[it].units[s].rel_name!=NO_REL_SYMBOL: # outgoing relation exist
                            target_node = int(gold_essays[it].units[s].targetID)
                            if gold_essays[it].units[target_node-1].dropping == True: # remove relation to ensure tree structure; must add -1 here
                                # print("relation removal", gold_essays[it].units[s].ID," to ", gold_essays[it].units[s].targetID)
                                gold_essays[it].units[s].rel_name = NO_REL_SYMBOL
                                gold_essays[it].units[s].targetID = NO_TARGET_CONSTANT
                                gold_essays[it].units[s].dropping = True
                                relation_removal_exist = True

                # save
                save_path = args.save_new_tsv_dir + gold_essays[it].essay_code + "_recovered_links.tsv"
                with open(save_path, "w+") as f:
                    f.write(gold_essays[it].to_tsv())

            


