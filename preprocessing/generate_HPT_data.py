"""
by  Jan Wira Gotama Putra

This script is used to split training and dev sets (for hyperparameter tuning) from existing preprocessed training set

Previously, I thought to generate 5-cross-validation train and dev, but this might be overkill for a hyperparameter tuning
    - Maybe just generate one fold will be enough

Basically, just copy and paste files
"""
import argparse
import numpy as np
import csv
from preprocessing.common_functions import list_files_in_dir
from preprocessing.common_functions import open_essays
from sklearn.model_selection import KFold
import os
import shutil


if __name__ == "__main__":
    # user arguments
    parser = argparse.ArgumentParser(description='Split training data to train and dev')
    parser.add_argument(
        '-in_dir', '--in_dir', type=str, help='relative directory of training data', required=True)
    parser.add_argument(
        '-out_dir', '--out_dir', type=str, help='relative directory for output', required=True)
    parser.add_argument(
        '-N_fold', '--N_fold', type=int, help='number of folds', required=False, default=5)
    args = parser.parse_args()

    # open files
    files = list_files_in_dir(args.in_dir)

    # get unique essay codes
    essay_codes = set()
    for file in files:
        essay_code = file.split("/")[-1].split(".")[0]
        essay_codes.add(essay_code)
    print("# Essays", len(essay_codes))
    essay_codes = list(essay_codes)
    essay_codes.sort()

    # generate splits
    CV = KFold(n_splits=args.N_fold, shuffle=True, random_state=868) # do not change the random split for reproducibility
    x_idx = np.arange(len(essay_codes))
    y_idx = np.arange(len(essay_codes))
    it = 1
    for train_index, dev_index in CV.split(x_idx, y_idx):

        # directory to hold the fold information
        train_dir = args.out_dir + "train/"
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)
        dev_dir = args.out_dir + "dev/"
        if not os.path.exists(dev_dir):
            os.mkdir(dev_dir)

        print("Preparing split; train: %d  dev=%d" % (len(train_index), len(dev_index)))
        
        # copy train data
        for x in train_index:
            preprocessed_files = [s for s in files if essay_codes[x] in s]
            for e in preprocessed_files:
                shutil.copy2(e, train_dir + e.split("/")[-1])

        # copy dev data
        for x in dev_index:
            preprocessed_files = [s for s in files if essay_codes[x] in s]
            for e in preprocessed_files:
                shutil.copy2(e, dev_dir + e.split("/")[-1])

        it+=1
        break # we only care about one fold

