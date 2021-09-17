"""
Split train and test files

Basically, just copy and paste files
"""
import argparse
import numpy as np
import csv
from preprocessing.common_functions import list_files_in_dir
from preprocessing.common_functions import open_essays
from preprocessing.tsv_to_vector import check_train_or_test
import os
import shutil


if __name__ == "__main__":
    # user arguments
    parser = argparse.ArgumentParser(description='Split train and test set')
    parser.add_argument(
        '-in_dir', '--in_dir', type=str, help='relative directory of dataset', required=True)
    parser.add_argument(
        '-out_dir', '--out_dir', type=str, help='relative directory for output', required=True)
    parser.add_argument(
        '-split', '--split', type=str, help='train_test_split information file', required=True)
    args = parser.parse_args()

    # open files
    files = list_files_in_dir(args.in_dir)
    files.sort()
    essays = open_essays(files)
    print("Data:",len(essays), "essays")

    # train test split
    if args.split:
        print ("Split provided")
        with open(args.split, 'r') as f:
            split_info = [row for row in csv.reader(f.read().splitlines(), delimiter=',')]
        for i in range(len(split_info)):
            split_info[i][0] = split_info[i][0].split(".")[0] # delete the file extension
            split_info[i][1] = split_info[i][1].replace('\"','').strip()


    # convert to vector form
    for it in range(len(essays)):
        essay = essays[it]
        print("Processing",essay.essay_code)

        # determine where to save the file
        if args.split:
            split_folder = check_train_or_test(split_info, essay.essay_code)
            assert (split_folder != None)
            split_folder = split_folder + "/"
        else: 
            split_folder = "" # no split information provided

        # directory to hold the fold information
        out_path = args.out_dir + split_folder
        
        # copy data
        shutil.copy2(files[it], out_path + essay.essay_code + ".tsv")
