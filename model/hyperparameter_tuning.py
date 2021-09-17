"""
Author: Jan Wira Gotama Putra
"""
from typing import *
from tqdm import tqdm
import time
import numpy as np
import argparse
import ast
import itertools
import numpy as np
import os
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.trainer import Trainer
from allennlp.data.iterators import DataIterator

from model.datasetreader import * 
from model.model_functions import *
from model.Nets.BiaffineMTL import *
from model.predictors import * 
from model.predict import flatten_list

from preprocessing.common_functions import remove_unwanted_files, list_files_in_dir

from sklearn.metrics import classification_report, confusion_matrix


def param_grid_combinations(args):
    """
    Create a combination of parameter grid
    
    Args:
        args (argparse)

    Returns:
        list[dict]
    """
    dropout_rate = ast.literal_eval(args.dropout_rate)
    batch_size = ast.literal_eval(args.batch_size) 

    param_grid = dict(
        dropout_rate=dropout_rate,
        batch_size=batch_size
    )

    keys = param_grid.keys()
    values = (param_grid[key] for key in keys)
    combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]

    return combinations


def logging(s: str, log_file, log_file_only: bool=False):
    """
    A helper function

    Args:
        s (str): the message you want to log
        log_file: log file object
        log_file_only (bool): flag if you want to print this in log file only
    """
    if not log_file_only:
        print(s)
    log_file.write(s+"\n")
    log_file.flush()


if __name__ == "__main__":
    # user arguments
    parser = argparse.ArgumentParser(description='Hyperparameter tuning using cross-validation')
    parser.add_argument(
        '-mode', '--mode', type=str, help='"test_run" or "real_run"', required=True)
    parser.add_argument(
        '-architecture', '--architecture', type=str, help='{"BiaffineMTL"}', required=True)
    parser.add_argument(
        '-dir', '--dir', type=str, help='dataset directory (CV data)', required=True)
    parser.add_argument(
        '-dropout_rate', '--dropout_rate', type=str, help='list of dropout_rate', required=True)
    parser.add_argument(
        '-batch_size', '--batch_size', type=str, help='list of batch_size', required=True)
    parser.add_argument(
        '-epochs', '--epochs', type=int, help='the number of max epoch', required=True)
    parser.add_argument(
        '-evaluate_every', '--evaluate_every', type=int, help='evaluate model every X epoch', required=True)
    parser.add_argument(
        '-aux_tasks', '--aux_tasks', type=str, help='the list of auxiliary tasks', required=True)
    parser.add_argument(
        '-log', '--log', type=str, help='path to save the running log', required=True)
    args = parser.parse_args()

    # check if architecture is known
    accepted_architectures = set(["BiaffineMTL"])
    if not (args.architecture in accepted_architectures):
        raise Exception("Unknown Architecture!")

    # safety measure
    if os.path.exists(args.log):
        raise Exception("Existing log file with the same name exists! Use the different name!")

    # log file
    log_file = open(args.log,"w+")
    csv_separator = "\t"

    # arguments
    logging("args: "+str(args)+"\n", log_file)

    # cuda device
    cuda_device = 0 if torch.cuda.is_available() else -1
    torch_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging("CUDA device = "+str(cuda_device)+" (running on "+str(torch_device)+")", log_file)
    available_gpus = np.arange(torch.cuda.device_count()).tolist()
    logging("Available GPUs "+str(available_gpus), log_file)

    # hyperparameter combinations
    hyperparams = param_grid_combinations(args)
    logging("Param combinations: "+str(len(hyperparams)), log_file)

    # run configuration
    run_config = Config(
        architecture=args.architecture,
        use_percentage=1.0, # how much data to use
        emb_dim=768,
        reduc_dim=512,
        lstm_u=256, 
        n_stack=3,
        fc_u=256,
        epochs=args.epochs,
        evaluate_every=args.evaluate_every,
        learning_rate=0.001,
        aux_tasks=ast.literal_eval(args.aux_tasks),
        aux_features=[], # we do not set auxiliary features for HPT; should be much or less the same
    )
    # change the epoch for test run
    if args.mode == "test_run": 
        run_config.epochs = 1
    logging("run_config: "+str(run_config)+"\n", log_file)
    
    # HPT data directory
    train_data_dir = args.dir + "train/"
    dev_data_dir = args.dir + "dev/"
    
    # dataset reader
    dataset_reader = SeqDatasetReader(
        use_percentage=run_config.use_percentage,
    )

    # Training using train and dev
    header = [k for k, v in hyperparams[0].items()] + ["epoch", "F1-macro (linking)", "Accuracy (linking)"]
    logging("\t".join(header), log_file, True)
    for hyperparam in hyperparams:
        # load train dev set
        train_ds = dataset_reader.read(train_data_dir)
        valid_ds = dataset_reader.read(dev_data_dir)
        print("# train data: %d    # dev data: %d\n" % (len(train_ds), len(valid_ds)))
        print("Hyperparam:", hyperparam)

        # prepare batch for training
        combined_dataset = train_ds + valid_ds
        vocab = Vocabulary.from_instances(combined_dataset) # a workaround so the model can predict unseen label in the training data
        del combined_dataset # saving memory
        
        iterator = BasicIterator(batch_size=hyperparam["batch_size"])
        iterator.index_with(vocab) # this is a must for consistency reason

        # batch for prediction
        predict_iterator = BasicIterator(batch_size=1) # not to blow up GPU memory
        predict_iterator.index_with(vocab)

        # model
        model_config = deepcopy(run_config)
        print(model_config)
        model_config.set("batch_size", hyperparam["batch_size"])
        model_config.set("dropout_rate", hyperparam["dropout_rate"])
        model = get_model(model_config, vocab, torch_device)
        model.to(torch_device)

        # optimizer
        optimizer = optim.Adam(model.parameters(), lr=run_config.learning_rate)

        # train for until max epochs, with checkpoint-ing
        t_iteration = 0
        while t_iteration < run_config.epochs:

            # how many epochs the model has been trained on
            t_iteration += run_config.evaluate_every

            # train
            train_start_time = time.time()
            trainer = Trainer(
                model=model,
                optimizer=optimizer,
                iterator=iterator,
                train_dataset=train_ds,
                shuffle=True, # to shuffle the batch
                cuda_device=cuda_device,
                should_log_parameter_statistics=False,
                num_epochs=run_config.evaluate_every,
            )
            metrics = trainer.train()
            train_end_time = time.time()
            print('Finished training for epochs=%d, time %.3fmins' % (run_config.evaluate_every, (train_end_time-train_start_time)/60.0))

            # prediction (evaluation on dev set)
            predictor = PredictorBiaffine(model, predict_iterator, cuda_device=cuda_device)
            # only predict rel distances for hyperparameter tuning
            link_preds, link_golds = predictor.predict(valid_ds, namespace="rel_distances_labels", coerce_tree=True)
            link_preds_flat = flatten_list(link_preds)
            link_golds_flat = flatten_list(link_golds)

            # performance report linking
            print(classification_report(y_true=link_golds_flat, y_pred=link_preds_flat, digits=3))
            linking_report = classification_report(y_true=link_golds_flat, y_pred=link_preds_flat, output_dict=True)

            logging_row = [hyperparam["dropout_rate"], hyperparam["batch_size"], t_iteration, linking_report['macro avg']['f1-score'], linking_report['accuracy']]
            logging("\t".join([str(x) for x in logging_row]), log_file)

            # conserve memory
            del predictor
            del trainer
            del link_preds_flat
            del link_golds_flat
            del link_preds
            del link_golds
            if torch_device.type == "cuda":
                torch.cuda.empty_cache()

        # conserve memory
        del iterator
        del predict_iterator
        del optimizer
        del model
        del vocab
        del train_ds
        del valid_ds
        if torch_device.type == "cuda":
            torch.cuda.empty_cache()

    # end for hyperparams
    log_file.close()

