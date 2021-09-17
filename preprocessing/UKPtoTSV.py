"""
Script: convert UKP dataset into our dataset format
"""

import os
from os import listdir
from os.path import isfile, join
import argparse
import numpy as np
import csv
import re
import sys
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import collections
from deprecated import deprecated


class Unit:
    def __init__(self, code=None, order=None, text=None, target_id=None, target_rel=None, dropping=None):
        self.code = code
        self.id = order # original text order
        self.text = text
        self.target_id = target_id
        self.target_rel = target_rel
        self.dropping = dropping

    def setArgCode(self, argComponent):
        """
        Copy info from argComponent
        """
        self.code = argComponent.code
        self.dropping = False

    def __str__(self):
        return "[" + str(self.code) + "]"+ " (" + str(self.id) + ", " + str(self.target_id) + ", " + str(self.target_rel) + ", " + str(self.dropping) + ") " + str(self.text)


class ArgComponent:
    def __init__(self, code, label, sentence):
        x = label.split(" ")
        self.code = code
        self.label = x[0] # component class
        self.sentence = sentence
        self.alignment = None # for textual alignment
        self.start_idx = int(x[1]) # the start index of this component in the actual text
        self.end_idx = int(x[2])
        self.local_idx = None # for non-arg units segmentation

    def __str__(self):
        return "[" + self.code + "-" + self.label + "] " + self.sentence + " (start, end=" + str(self.start_idx) + "," + str(self.end_idx) + "; alignment="+ str(self.alignment)+")"


class Relation:
    def __init__(self, rel_info=None):
        if rel_info == None:
            self.rel_name = None
            self.source = None
            self.target = None
        else:
            x = rel_info.split(" ")
            self.rel_name = x[0]
            self.source = x[1].split(":")[1]
            self.target = x[2].split(":")[1]

    def __str__(self):
        return self.source + " -> " + self.target + " [" + self.rel_name + "]"


class Stance:
    def __init__(self, info):
        x = info.split(" ")
        self.source = x[1]
        self.stance = x[2].lower()

    def __str__(self):
        return "[Stance-" + self.source + " : " + self.stance + "]"


def open_ann(path):
    """
    Open annotation data
    """
    with open(path, 'r') as f:
        data = [row.split("\t") for row in f.read().splitlines()]
    return data


def open_csv(path):
    with open(path, 'r') as f:
        data = [row for row in csv.reader(f.read().splitlines(), delimiter=';')]
    del data[0] # delete header
    return data


def open_txt(path):
    """
    Open text file and then discard the prompt
    """
    with open(path, 'r') as f:
        data = [row for row in csv.reader(f.read().splitlines(), delimiter='\t', quotechar=None)]
    prompt_len = len(data[0][0]) + 1 # count newline
    del data[0] # do not need the prompt
    if len(data[0]) != 0:  # supposed to be empty but not empty
        prompt_len += len(data[0][0]) + 1 # count spaces and newline
    del data[0] # do not need the newline
    output = ""
    for x in data:
        if len(x) > 0: # not empty
            output += (x[0] + " ") 
    return output.strip(), prompt_len


def print_element(o):
    for e in o:
        print (e)


def get_stance(stances, claimCode):
    for s in stances:
        if s.source == claimCode:
            return s.stance
    return None


def get_unit_idx(units, code):
    for i in range(len(units)):
        if units[i].code == code:
            return i
    return None


def get_rel(rels, sourceCode):
    for r in rels:
        if s.source == sourceCode:
            return r
    return None


def augment_relation(argComponents, rels, stances):
    """
    Augment relations from claim to the first major claim and other major claim(s) to the first major claim
    Also, change the relation label to suit our scheme (for consistency purpose)
    """
    firstMC = None
    for i in range(len(argComponents)):
        if argComponents[i].label == "MajorClaim":
            if firstMC == None:
                firstMC = i
            else:
                augRel = Relation()
                augRel.rel_name = "augmentation" # augmentation
                augRel.source = argComponents[i].code
                augRel.target = argComponents[firstMC].code
                rels.append(augRel)

    # connect claim to firstMC, this is done with another loop since MajorClaim possible not located in the beginning           
    for i in range(len(argComponents)):
        if argComponents[i].label == "Claim":
            augRel = Relation()
            augRel.rel_name = get_stance(stances, argComponents[i].code)
            if augRel.rel_name == "for":
                augRel.rel_name = "supports"
            elif augRel.rel_name == "against":
                augRel.rel_name = "attacks"
            augRel.source = argComponents[i].code
            augRel.target = argComponents[firstMC].code
            rels.append(augRel)
    
    # change the relation naming to suit our scheme
    for i in range(len(rels)):
        if rels[i].rel_name == "supports":
            rels[i].rel_name = 'sup'
        elif rels[i].rel_name == "attacks":
            rels[i].rel_name = "att"
        elif rels[i].rel_name == "augmentation":
            rels[i].rel_name = "=" # restatement


def read_annotation_data(dir_path, essay_name):
    """
    Read annotation data from the UKP dataset
    """
    data = open_ann(dir_path+essay_name+".ann")
    argComponents = []
    rels = []
    stances = []
    for entry in data:
        if 'T' in entry[0]:
            newArg = ArgComponent(entry[0], entry[1], entry[2])
            argComponents.append(newArg)
        elif 'R' in entry[0]:
            newRel = Relation(entry[1])
            rels.append(newRel)
        elif 'A' in entry[0]:
            newStance = Stance(entry[1])
            stances.append(newStance)
    augment_relation(argComponents, rels, stances) # augment stances into real relations
    return argComponents, rels


def check_AC_within_sentence(AC, sentence, text, prompt_len):
    """
    Check if the AC is within the current sentence
    """
    flag = False
    flag = AC.sentence in sentence

    if flag:
        # preventing false positive
        sentence_start_idx = text.index(sentence) + prompt_len # +1 for newline
        sentence_end_idx = sentence_start_idx + len(sentence) +1 # + 1 because the indexing in UKP's dataset seems to be off +1
        return AC.start_idx >= sentence_start_idx and AC.end_idx <= sentence_end_idx
    else:
        return flag


def output_to_tsv(save_path, essay_code, units):
    content = []
    content.append(['essay code', 'sentence ID', 'text', 'target', 'relation', 'drop flag']) # header
    for unit in units:
        content.append([essay_code, str(unit.id), unit.text.strip(), str(unit.target_id), unit.target_rel, str(unit.dropping)])

    # flush
    f = open(save_path, 'w+')
    writer = csv.writer(f, delimiter='\t')
    for row in content:
        writer.writerow(row)
    f.close()


if __name__ == "__main__":
    dir_path = "data/UKP/UKP-2.0/brat-project-final/"
    save_dir = "data/UKP/tsv/"

    # get essay
    source_files = [join(dir_path, f) for f in listdir(dir_path) if isfile(join(dir_path, f))]
    essay_codes = []
    for s in source_files:
        if "essay" in s:
            code = s.split("/")[-1].split(".")[0]
            if not code in essay_codes:
                essay_codes.append(code)
    
    # open essays
    essay_codes.sort()
    n_sentences = 0
    for e in essay_codes:
        print ("Processing ", e)

        # read the original text
        ori_text, prompt_len = open_txt(dir_path+e+".txt")
        ori_sentences = sent_tokenize(ori_text)
        # print(ori_sentences)

        # read annotation data
        argComponents, rels = read_annotation_data(dir_path, e)
        argComponents.sort(key=lambda x: x.start_idx) # according to original order in text

        # alignment
        realignment = True
        while (realignment):
            # initialization
            AC_alignment_flag = [-1] * len(argComponents) # which ori_sentences we align the argument components to
            ori_sent_alignment_flag = [] # which argcomponent the sentence belongs to; one sentence can contain multiple arg components
            for i in range(len(ori_sentences)):
                ori_sent_alignment_flag.append([])

            # alignment between argComponents to the ori_sentences
            for i in range(len(ori_sentences)):
                for j in range(len(argComponents)):
                    # if argComponents[j].sentence in ori_sentences[i]:
                    if check_AC_within_sentence(argComponents[j], ori_sentences[i], ori_text, prompt_len):
                        AC_alignment_flag[j] = i
                        ori_sent_alignment_flag[i].append(j)
            
            # check completeness of alignment
            non_aligned = 0
            for x in AC_alignment_flag:
                if x == -1:
                    non_aligned += 1

            # not complete, most probably caused by wrong segmentation in nltk, especialy when "etc. ... " or "e.g.," presents in the text
            if non_aligned!=0: 
                if e == "essay399": # a very special case
                    ori_sentences[9] = ori_sentences[9] + " " +ori_sentences[10]
                    del ori_sentences[10]
                    realignment = True
                else:
                    for i in range(len(argComponents)):
                        if AC_alignment_flag[i] == -1:
                            for j in range(len(ori_sentences)):
                                if ori_sentences[j] in argComponents[i].sentence:
                                    ori_sentences[j] = ori_sentences[j] + " " + ori_sentences[j+1]
                                    del ori_sentences[j+1]
                                    realignment = True
                                    break
            else:
                realignment = False

        # creating units
        units = []
        unit_id = 1
        for i in range(len(ori_sentences)):
            if len(ori_sent_alignment_flag[i]) == 0: # non-AC
                newUnit = Unit(code="non-AC", order=unit_id, text=ori_sentences[i], target_id="", target_rel="", dropping=True)
                units.append(newUnit)
                unit_id += 1
            elif len(ori_sent_alignment_flag[i]) == 1: # AC
                newUnit = Unit(code=argComponents[ori_sent_alignment_flag[i][0]].code, order=unit_id, text=ori_sentences[i], target_id="", target_rel="", dropping=False)
                units.append(newUnit)
                unit_id += 1
            else: # multiple ACs in one sentence
                prev_end_idx = 0
                for j in range(len(ori_sent_alignment_flag[i])):
                    AC = argComponents[ori_sent_alignment_flag[i][j]]
                    begin_idx = ori_sentences[i].index(AC.sentence)

                    # slice the text
                    if j == len(ori_sent_alignment_flag[i]):
                        sub_text = ori_sentences[i][prev_end_idx:]
                    else:
                        sub_text = ori_sentences[i][prev_end_idx: begin_idx+len(AC.sentence)]
                        prev_end_idx = begin_idx+len(AC.sentence)+1

                    newUnit = Unit(code=AC.code, order=unit_id, text=sub_text, target_id="", target_rel="", dropping=False)
                    units.append(newUnit)
                    unit_id += 1

        # establish connections between units
        for rel in rels:
            source_id = get_unit_idx(units, rel.source) 
            target_id = get_unit_idx(units, rel.target)
            units[source_id].target_id = str(target_id+1) # need to add +1 because the csv starts from 1
            units[source_id].target_rel = rel.rel_name
        # print_element(rels)
        # input()

        # final check: all ACs should have connections, except for 1 node (major claim)
        AC_with_outgoing_connections = 0
        AC_count = 0
        for i in range(len(units)):
            if units[i].code != "non-AC":
                AC_count += 1
                if units[i].target_id != "":
                    AC_with_outgoing_connections += 1

        if (AC_count != AC_with_outgoing_connections + 1):
            print_element(units)
            input()


        # convert to TSV and save
        save_path = save_dir + "UKP_"+e + ".tsv"
        output_to_tsv(save_path, "UKP_"+e, units)
        
        