"""
Author: Jan Wira Gotama Putra

Prediction of sentence links, using greedy decoding (NOT enforcing the output to form a tree)
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
        '-model_dir', '--model_dir', type=str, help='model directory (containing many models)', required=True)
    parser.add_argument(
        '-pred_dir', '--pred_dir', type=str, help='directory to save the prediction result', required=True)
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

    # iterate over models
    for model_dir in model_dirs:
        # model configuration
        model_config = get_model_config(model_dir+"/")
        print("Model architecture:", model_config.architecture)
        prediction_namespaces = ["rel_distances_labels"] + model_config.aux_tasks

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
        predictor = PredictorBiaffine(model, iterator, cuda_device=cuda_device)
        for task in prediction_namespaces:
            print("Task:", task)
            test_preds, gold_preds = predictor.predict(test_ds, namespace=task, coerce_tree=True)

            # to judge the model's quality
            if task == "rel_distances_labels":
                print("The ratio of tree from argmax prediction only %.3lf" % (predictor.tree_from_argmax / len(test_ds)) )
                predictor._reset_tree_argmax_count()
        
            subdir = model_dir.split("/")[-1]
            save_output(args.pred_dir + subdir + "/", task + "_pred.txt", test_preds)
            save_output(args.pred_dir + subdir + "/", task + "_gold.txt", gold_preds)

            # test_preds_flat = flatten_list(test_preds)
            # gold_preds_flat = flatten_list(gold_preds)

            # print(classification_report(y_true=gold_preds_flat, y_pred=test_preds_flat, digits=3))



