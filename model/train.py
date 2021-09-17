"""
Author: Jan Wira Gotama Putra
"""
from typing import *
import time
import argparse
import numpy as np
import ast

import torch
import torch.nn as nn
import torch.optim as optim

from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.trainer import Trainer
from allennlp.data.iterators import DataIterator

from model.datasetreader import * 
from model.model_functions import *
from model.Nets.BiaffineMTL import *


POSSIBLE_AUX_TASKS = ["component_labels", "node_depths_labels"] # the list of supported auxiliary task
POSSIBLE_AUX_FEATURES = ["sentence_position"]


def load_training_data(args, config):
    """
    A helper function to load training data
    """
    # load training data
    train_ds = dataset_reader.read(args.dir)
    print("# Training Instances", len(train_ds))

    # prepare batch
    vocab = Vocabulary.from_instances(train_ds)
    iterator = BasicIterator(batch_size=config.batch_size)
    iterator.index_with(vocab) # this is a must for consistency reason

    return train_ds, vocab, iterator


if __name__ == "__main__":
    # user arguments
    parser = argparse.ArgumentParser(description='Training linking model')
    parser.add_argument(
        '-mode', '--mode', type=str, help='"test_run" or "real_run"', required=True)
    parser.add_argument(
        '-architecture', '--architecture', type=str, help='{"BiaffineMTL"}', required=True)
    parser.add_argument(
        '-dir', '--dir', type=str, help='dataset directory (train data)', required=True)
    parser.add_argument(
        '-reduc_dim', '-reduc_dim', type=int, help='dimension reduction layer size', required=False, default=512)
    parser.add_argument(
        '-lstm_u', '-lstm_u', type=int, help='LSTM unit size', required=False, default=256)
    parser.add_argument(
        '-n_stack', '-n_stack', type=int, help='number of BiLSTM stack', required=False, default=3)
    parser.add_argument(
        '-fc_u', '-fc_u', type=int, help='dense layer size (after BiLSTM)', required=False, default=256)
    parser.add_argument(
        '-dropout_rate', '--dropout_rate', type=float, help='dropout_rate', required=False, default=0.5)
    parser.add_argument(
        '-batch_size', '--batch_size', type=int, help='batch_size', required=False, default=4)
    parser.add_argument(
        '-epochs', '--epochs', type=int, help='epochs', required=True)
    parser.add_argument(
        '-n_run', '--n_run', type=int, help='the number of run', required=True)
    parser.add_argument(
        '-aux_tasks', '--aux_tasks', type=str, help='the list of auxiliary tasks', required=True)
    parser.add_argument(
        '-aux_features', '--aux_features', type=str, help='the list of auxiliary features', required=True)
    parser.add_argument(
        '-save_dir', '--save_dir', type=str, help='directory to save trained models', required=True)
    parser.add_argument(
        '-save_start_no', '--save_start_no', type=int, help='save start number', required=False, default=1) # when we save the model, the script will create subfolders run-1, run-2, etc.. This is to specify which number to start from
    args = parser.parse_args()

    # check if architecture is known
    accepted_architectures = set(["BiaffineMTL"])
    if not (args.architecture in accepted_architectures):
        raise Exception("Unknown Architecture!")

    # validate aux tasks
    aux_tasks = ast.literal_eval(args.aux_tasks)
    for x in aux_tasks:
        if x not in POSSIBLE_AUX_TASKS:
            raise Exception(x, "is not supported as an auxiliary task")

    # validate aux features
    aux_features = ast.literal_eval(args.aux_features)
    for x in aux_features:
        if x not in POSSIBLE_AUX_FEATURES:
            raise Exception(x, "is not supported as an auxiliary feature")

    # arguments
    print(str(args)+"\n")

    # cuda device
    cuda_device = 0 if torch.cuda.is_available() else -1
    torch_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("CUDA device = "+str(cuda_device)+" (running on "+str(torch_device)+")")
    available_gpus = np.arange(torch.cuda.device_count()).tolist()
    print("Available GPUs "+str(available_gpus))

    # epochs
    if args.mode == "test_run":
        print("Max epoch=1 for test run")
        args.epochs = 1
    print("# Epochs =", args.epochs)
    
    # train config
    config = Config(
        architecture=args.architecture,
        use_percentage=1.0, # how much data to use
        emb_dim=768,
        reduc_dim=args.reduc_dim,
        lstm_u=args.lstm_u, 
        n_stack=args.n_stack,
        fc_u=args.fc_u,
        dropout_rate=args.dropout_rate,
        epochs=args.epochs,
        batch_size=args.batch_size, # number of essays 
        learning_rate=0.001,
        aux_tasks=aux_tasks,
        aux_features=aux_features,
    )
    print("Config: ", config)

    # dataset reader
    dataset_reader = SeqDatasetReader(
        use_percentage=config.use_percentage,
    )

    # load training data and prepare batching (CPU only)
    if torch_device.type == "cpu": 
        train_ds, vocab, iterator = load_training_data(args, config)

    # train model
    for i in range(args.n_run):
        # loading train data inside the loop. I am not sure why but this is more memory-friendly for GPU (when run)
        if torch_device.type == "cuda":
            train_ds, vocab, iterator = load_training_data(args, config)

        # model
        model = get_model(config, vocab, torch_device)
        model.to(torch_device)

        # optimizer
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        # learning rate scheduler
        lr_scheduler = LearningRateScheduler.from_params(optimizer, Params({"type" : "step", "gamma": 0.1, "step_size": config.epochs-10})) # this step size is rather experimental science rather than real science

        train_start_time = time.time()
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            iterator=iterator,
            train_dataset=train_ds,
            shuffle=True, # to shuffle the batch
            cuda_device=cuda_device,
            should_log_parameter_statistics=False,
            learning_rate_scheduler=lr_scheduler,
            num_epochs=config.epochs,
        )
        metrics = trainer.train()
        train_end_time = time.time()
        print('Finished Training, run=%d, epochs=%d, time %.3fmins' % (i+args.save_start_no, config.epochs, (train_end_time-train_start_time)/60.0))

        # save model
        save_model(model, args.save_dir+"run-"+str(i+args.save_start_no)+"/")
        print()

        # conserve memory
        del trainer
        del optimizer
        del model
        if torch_device.type == "cuda":
            del iterator
            del train_ds # since the train_ds of the previous iteration has been moved to gpu, we need to (force) delete it (because otherweise, it won't get detached but may persists) and then reload again to conserve memory
            torch.cuda.empty_cache()
        

