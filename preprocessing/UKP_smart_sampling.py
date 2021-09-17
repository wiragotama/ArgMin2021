"""
Split train and test files

Sample UKP dataset that satisfy the following requirements:
1. max length 17 sentences (avg. 14 in ICNALE + 3 stdev)
2. Max 2 non-AC per essay (avg 0.5 in ICNALE + 9 stdev)

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
    parser = argparse.ArgumentParser(description='Smart sampling of UKP dataset')
    parser.add_argument(
        '-in_dir', '--in_dir', type=str, help='relative directory of dataset', required=True)
    parser.add_argument(
        '-out_dir', '--out_dir', type=str, help='relative directory for output', required=True)
    args = parser.parse_args()

    # open files
    files = list_files_in_dir(args.in_dir)
    files.sort()
    essays = open_essays(files)
    print("Data:",len(essays), "essays")

    count = 0
    for it in range(len(essays)):
        essay = essays[it]
        print("Processing",essay.essay_code)

        if len(essay.units) <= 17 and essay.n_non_ACs() <= 2:
            # copy data
            shutil.copy2(files[it], args.out_dir + essay.essay_code + ".tsv")
            count += 1

    print(count)
