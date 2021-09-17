"""
by  Jan Wira Gotama Putra

This is the data structure to represent annotated essays 
"""
import os
import numpy as np
import csv
import re
import sys
import codecs
from bs4 import BeautifulSoup
from os import listdir
from os.path import isfile, join
from copy import deepcopy
from copy import deepcopy
from nltk.tokenize import word_tokenize
import random
import itertools

"""
Global constants
"""
NO_TARGET_CONSTANT = -1 #-1 means the discourse unit in question has no target (no outgoing link)
DELIMITER = "\t"
NO_REL_SYMBOL = '' # empty string 
RESTATEMENT_SYMBOL = '='
str_to_bool = lambda s: s.lower() in ["true", "t", "yes", "1"] # convert boolean string to boolean
SOURCE_BEFORE_TARGET = 1
SOURCE_AFTER_TARGET = 0



class DiscourseUnit:
    """
    DiscourseUnit data structure
    """

    def __init__(self):
        global DELIMITER
        self.ID = "" # unit ID (original order)
        self.text = ""
        self.targetID = ""
        self.rel_name = ""
        self.dropping = ""

    def __display_targetID(self):
        return str(self.targetID) if self.targetID!=-1 else ""

    def __str__(self):
        return str(self.ID) + DELIMITER + self.text + DELIMITER + self.__display_targetID() + DELIMITER + str(self.rel_name) + DELIMITER + str(self.dropping)



class Essay:
    """
    Essay data structure
    
    Args:
        filepath (str): file that contains the essay annotation

    Attributes:
        essay_code (str): the essay code of the current object
        units (:obj:`list` of :obj:`DiscourseUnit`): list of discourse untits 
        scores (:obj:`list` of :obj:`int`): essay score(s)
        score_types (:obj:`list` of :obj:`str`): singleton variable of score types

    """

    score_types = ["Content (/12)", "Organization (/12)", "Vocabulary (/12)", "Language Use (/12)", "Mechanics (/12)", "Total 1 (%)", "Total 2 (Weighted %)"]

    def __init__(self, filepath):
        """
        :param string filepath: file that contains the essay annotation
        """
        global NO_TARGET_CONSTANT
        self.essay_code, self.units = self.__process_file(filepath) #  automatically detects html or tsv and then process them
        self.scores =  [] # essay scores, there are many as defined by score_types (singleton variable)


    def n_ACs(self):
        """
        Returns: 
            the number of argumentative components (non-dropped) in the essay
        """
        total = 0
        for unit in self.units:
            if unit.dropping == False:
                total += 1
        return total


    def n_non_ACs(self):
        """
        Returns:
            the number of non-argumentative components (non-dropped) in the essay
        """
        total = 0
        for unit in self.units:
            if unit.dropping == True:
                total += 1
        return total


    def n_sentences(self):
        """
        Returns:
            the number of sentences in the essay
        """
        return len(self.units)


    def n_rel(self, label):
        """
        Args:
            label (str): relation label

        Returns:
            the number of relations with corresponding label existing in the essay
        """
        total = 0
        for unit in self.units:
            if unit.rel_name == label:
                total += 1
        return total


    def n_tokens(self):
        """
        Returns:
            the number of total tokens in the essay
        """
        total = 0
        for unit in self.units:
            total += len(word_tokenize(unit.text))
        return total


    def n_tokens_per_sentence(self):
        """
        Returns:
            the number of total tokens in the essay per sentence
        """
        output = []
        for unit in self.units:
            output.append(len(word_tokenize(unit.text)))
        return output

    
    def get_pairwise_ordering_data(self, arg_links_only=False, encode_sentences=False, encoder=None, normalised=True, experiment="STL"):
        """
        Given a pair of (source, target), determine whether the source appears before the target (using final reordered information)
        
        Args:
            arg_links_only (bool): True if we want to use only pair of source and target sentences that appear in argumentative links
            encode_sentences (bool): whether to encode sentences in the essay 
            encoder (obj): any kind of encoder object, it needs to have "text_to_vec(str)" function
            normalised (bool): True if we want to use the original text without repair, False if use repaired text
            experiment (str): "STL" or "MTL"

        Returns:
            {
                list of tuple (essay_code, source unit_id, target unit_id, source_text, target_text, label)
            }
        """
        # assertion
        assert experiment in {"STL", "MTL"}

        # output
        source_target_label = []

        # sentences
        sentences = self.get_sentences(order="original", normalised=normalised) # must be in original order to be aligned with the adj_matrix

        # embeddings
        if encode_sentences:
            sent_embs = encoder.text_to_vec(texts)

        # text order & dropping status 
        order_annotated = [] # reordered order
        dropping_status = [False] * len(self.units) # aligned with the original order
        for unit in self.units:
            order_annotated.append(int(unit.ID)-1) # -1 so we can start from 0
            dropping_status[int(unit.ID)-1] = bool(unit.dropping) # aligned with the original order

        # output
        adj_matrix = self.adj_matrix() # adj matrix uses the unit ID (so it corresponds to original order) for ease of debug

        # for combinations of AllPairs
        if not arg_links_only:
            # create combinations of source and target
            IDs = []
            for i in range(len(adj_matrix)):
                if not dropping_status[i]:
                    IDs.append(i)
            combination_pairs = list(itertools.combinations(IDs, 2)) # for AllPairs setting

            # keep the seed for reproducibility, do not edit this number
            random.seed(42) 

            # randomly swap the order, so we can have a balanced dataset
            for i in range(len(combination_pairs)):
                verdict = random.randint(0, 1)
                if verdict == 1:
                    combination_pairs[i] = (combination_pairs[i][1], combination_pairs[i][0]) # swap

        # create instances
        for i in range(len(adj_matrix)):
            for j in range(len(adj_matrix)):
                # check filter
                if arg_links_only:
                    if adj_matrix[i, j] != NO_REL_SYMBOL:
                        use = True
                    else:
                        use = False
                else:
                    if (i!=j) and (not dropping_status[i] and not dropping_status[j]): # not self loop, and both source and target are not non-ACs
                        try: # check if the combination has been used before
                            combination_pairs.index((i,j))
                            use = True
                        except:
                            use = False
                    else: 
                        use = False
                
                if use:
                    # check if source appears before the target
                    if order_annotated.index(i) < order_annotated.index(j):
                        ordering = SOURCE_BEFORE_TARGET # before
                    else:
                        ordering = SOURCE_AFTER_TARGET # after

                    # add to output variable
                    if encode_sentences:
                        if experiment == "STL":
                            source_target_label.append([self.essay_code, i+1, j+1, sent_embs[i].tolist(), sent_embs[j].tolist(), ordering])
                        elif experiment == "MTL":
                            source_target_label.append([self.essay_code, i+1, j+1, sent_embs[i].tolist(), sent_embs[j].tolist(), ordering, adj_matrix[i,j]])
                    else:
                        if experiment == "STL":
                            source_target_label.append([ self.essay_code, i+1, j+1, sentences[i], sentences[j], ordering ])
                        elif experiment == "MTL":
                            source_target_label.append([ self.essay_code, i+1, j+1, sentences[i], sentences[j], ordering, adj_matrix[i,j]])

        return source_target_label


    def get_rel_distances(self, mode="reordering", include_non_arg_units=False):
        """
        Get the list of relation distance (+ relation labels, this is useful for stats) for all sentences, according to mode
            if mode=="reordering", use the ordering as annotated, while sorted if "original"
            minus sign means pointing backward (to something appeared before), while plus means forward

        Args:
            mode (str): {reordering, original}
            include_non_arg_units (bool): True of False

        Returns:
            {
                distances from each sentence to its target,
                labels from each sentence to its target
            }
        """
        units_copy = deepcopy(self.units)
        assert mode in {"reordering", "original"}
        if mode == "original":
            units_copy = sorted(units_copy, key=lambda unit: int(unit.ID))

        order = []
        for unit in units_copy:
            order.append(int(unit.ID))

        distances = []
        rels = []
        curr_pos = 0
        for unit in units_copy:
            if unit.dropping == False:
                if  unit.targetID != NO_TARGET_CONSTANT: # connected
                    target_pos = order.index(int(unit.targetID))
                    dist = target_pos - curr_pos # minus means pointing backward, plus means forward
                    rels.append(unit.rel_name)
                else: # not connected
                    dist = 0 # major claim points to itself
                    rels.append("major claim")
                distances.append(dist)
            else: # dropped = non-AC 
                if include_non_arg_units:
                    dist = 0
                    rels.append("non-AC")
                    distances.append(dist)
            curr_pos += 1

        return distances, rels


    def get_texts(self, order="reordering", normalised=False):
        """
        get the list of sentences in the essay

        Args:
            order (str, optional, defaults to 'reordering'): {"reordering", "original"}
            normalised(bool, optional, defaults to False): True if we use the original version of the text without repair, False if we use the text repair
        
        Returns:
            list of sentences in the essay
        """
        units_copy = deepcopy(self.units)
        if order == "original":
            units_copy = sorted(units_copy, key=lambda unit: int(unit.ID))

        output = []
        for x in units_copy:
            if normalised:
                output.append(Essay.discard_text_repair(x.text))
            else:
                output.append(Essay.use_text_repair(x.text))
        return output


    def get_ACI_label(self, mode):
        """
        Get AC vs. Non-AC labels

        Args:
            mode (str): {reordering, original}

        Returns:
            {
                component labels
            }
        """
        units_copy = deepcopy(self.units)
        assert mode in {"reordering", "original"}
        if mode == "original":
            units_copy = sorted(units_copy, key=lambda unit: int(unit.ID))

        # order = []
        # for unit in units_copy:
        #     order.append(int(unit.ID))
        # print(order)

        labels = []
        for unit in units_copy:
            if unit.dropping == False: # AC
                labels.append("AC")
            else: # dropped = non-AC 
                labels.append("non-AC")

        return labels


    def get_final_order(self, exclude_non_ac=True):
        """
        Get the final order (reordered) of sentences
        """
        order = []
        for unit in self.units:
            if exclude_non_ac and unit.dropping == True:
                pass
            else:
                order.append(int(unit.ID))
        return order


    def get_restatement_pairs(self):
        """
        Get the pair of source and target sentences that are connected by restatement label
        """
        adj_matrix = self.adj_matrix() # adj matrix uses the unit ID (so it corresponds to original order) for ease of debug
        pairs = []
        for i in range(len(adj_matrix)):
            for j in range(len(adj_matrix)):
                if adj_matrix[i, j] != NO_REL_SYMBOL and adj_matrix[i, j] == '=':
                    pairs.append( (i+1, j+1) ) # plus one it corresponds to the ID appeared in text
        return pairs


    def get_sentences(self, order="reordering", normalised=False):
        """
        get the list of sentences in the essay

        Args:
            order (str, optional, defaults to 'reordering'): {"reordering", "original"}
            normalised(bool, optional, defaults to False): True if we use the original version of the text without repair, False if we use the text repair
        
        Returns:
            list of sentences in the essay
        """
        units_copy = deepcopy(self.units)
        if order == "original":
            units_copy = sorted(units_copy, key=lambda unit: int(unit.ID))

        output = []
        for x in units_copy:
            if normalised:
                output.append(Essay.discard_text_repair(x.text))
            else:
                output.append(Essay.use_text_repair(x.text))
        return output


    def get_ACs_and_rhetorical_categories(self, order="reordering", normalised=True):
        """
        Get the list of ACs and their rhetorical categories

        Since in our dataset, sentences' rhetorical category was not annoated, we use the relation label as an approximation
        If there is a link a -> b with relation X, then the rhetorical category for a is X
        We have 5 categories here: major claim, support, detail, attack, restatement

        Args:
            order (str, optional, defaults to 'reordering'): {"reordering", "original"}
            normalised(bool, optional, defaults to False): True if we use the original version of the text without repair, False if we use the text repair
        
        Returns:
            list of sentences in the essay
        """
        units_copy = deepcopy(self.units)
        if order == "original":
            units_copy = sorted(units_copy, key=lambda unit: int(unit.ID))

        output = []
        ret_labels = []
        for unit in units_copy:
            if unit.dropping == False:
                if normalised:
                    unit.text = Essay.discard_text_repair(unit.text)
                else:
                    unit.text = Essay.use_text_repair(unit.text)

                output.append(unit)
                if unit.rel_name == NO_REL_SYMBOL: # major claim
                    ret_labels.append("major claim")
                else:
                    ret_labels.append(unit.rel_name)
        return output, ret_labels


    def get_rhetorical_categories(self, order="reordering"):
        """
        Get the list of rhetorical categories for each sentence

        Since in our dataset, sentences' rhetorical category was not annoated, we use the relation label as an approximation
        If there is a link a -> b with relation X, then the rhetorical category for a is X
        We have 6 categories in this function: major claim, support, detail, attack, restatement and non-AC

        Args:
            order (str, optional, defaults to 'reordering'): {"reordering", "original"}

        Returns:
            list of sentences in the essay
        """
        units_copy = deepcopy(self.units)
        if order == "original":
            units_copy = sorted(units_copy, key=lambda unit: int(unit.ID))

        ret_labels = []
        for unit in units_copy:
            if unit.dropping == False:
                if unit.rel_name == NO_REL_SYMBOL: # major claim
                    ret_labels.append("major claim")
                else:
                    ret_labels.append(unit.rel_name)
            else:
                ret_labels.append("non-AC")
        return ret_labels


    def get_content(self, order, normalised, include_non_AC):
        """
        concat the list of sentences in the essay

        Args:
            order (str): {"reordering", "original"}
            normalised (bool): True if we use the original version of the text without repair, False if we use the text repair
            include_non_AC (bool): include non-ACs or not
        
        Returns:
            str
        """
        units_copy = deepcopy(self.units)
        if order == "original":
            units_copy = sorted(units_copy, key=lambda unit: int(unit.ID))

        output = []
        IDs = []
        for x in units_copy:
            if not include_non_AC and x.dropping == True:
                # do not include non-AC and the current node is non-AC
                pass
            else:
                if normalised:
                    output.append(Essay.discard_text_repair(x.text))
                else:
                    output.append(Essay.use_text_repair(x.text))
                IDs.append(x.ID)

        return " ".join(output), IDs


    def get_units_with_order(self, sequence):
        """
        Get the list of units in particular ordering sequence

        Args:
            sequence (list): start from 1

        Returns:
            list of DiscourseUnits
        """
        units_copy = deepcopy(self.units)
        units_copy = sorted(units_copy, key=lambda unit: int(unit.ID))

        retval = []
        for i in range(len(sequence)):
            retval.append(units_copy[sequence[i]-1])
        return retval


    def min_edit_distance(self):
        """
        Calculate the min_edit_distance between original ordering and annotated ordering of sentences

        Returns:
            {
                minimum edit distance score,
                sentence original order (sentence IDs)
                sentence reordered (sentence IDs)
            }
        """
        units_copy = deepcopy(self.units)
        units_copy = sorted(units_copy, key=lambda unit: int(unit.ID))

        order_annotated = []
        for unit in self.units:
            order_annotated.append(int(unit.ID))

        order_ori = []
        for unit in units_copy:
            order_ori.append(int(unit.ID))

        # Create a table to store results of subproblems
        n = len(order_annotated)
        dp = [[0 for x in range(n+1)] for x in range(n+1)] 
      
        # Fill d[][] in bottom up manner 
        for i in range(n+1): 
            for j in range(n+1): 
                if i == 0: 
                    dp[i][j] = j    # Min. operations = j 
                elif j == 0: 
                    dp[i][j] = i    # Min. operations = i 
                elif order_ori[i-1] == order_annotated[j-1]: 
                    dp[i][j] = dp[i-1][j-1] 
                else: # if different
                    dp[i][j] = 1 + min(dp[i][j-1],  # Insert 
                                       dp[i-1][j],  # Remove 
                                       dp[i-1][j-1]) # Replace 

        return dp[n][n], order_ori, order_annotated


    def projectivity_test(self, linearization="original"):
        """
        Test if the argumentative structure linearization contains only projective links (TRUE) or also non-projective links (FALSE, crossing dependencies) 
        
        Args:
            linearization (str): original or reordering

        Returns:
            (
                flag: TRUE (projective) or FALSE (non-projective),
                verdict (projective or non-projective),
                the proportion of non-projective links 
            )
        """
        def print_board(board):
            """
            for debugging
            """
            for x in board:
                print(x)


        assert linearization in {"reordering", "original"}
        units_copy = deepcopy(self.units)
        if linearization == "original":
            units_copy = sorted(units_copy, key=lambda unit: int(unit.ID))

        IDs, targets = [], []
        for unit in units_copy:
            if unit.dropping == False: # AC
                IDs.append(unit.ID)
                targets.append(unit.targetID)
            else: # dropped = non-AC 
                pass
        # print(IDs)
        # print(targets)
        
        # link info
        link_info = []
        for i in range(len(targets)):
            if targets[i] != -1:
                target_pos = IDs.index(targets[i])
                source_pos = i 
                dist = abs(source_pos-target_pos)
                link_info.append((dist, source_pos, target_pos))
        link_info.sort() # need to be sort, we draw short-distanced links first

        # projectivity test, simulating the drawing of the thing
        board = []
        for i in range(len(IDs)):
            board.append(['.'] * len(IDs))
        
        # establishing relations: we need to start from adjacent relations, then move to faraway relations, not based on IDs
        count = 0
        flag = True # projective
        n_non_projective_links = 0
        for link in link_info:
            # print(link)
            dist, source_pos, target_pos = link[0], link[1], link[2]
            curr_verdict = True # the current link is projective

            # drawing horizontal line
            for j in range(dist+1):
                if board[source_pos][j] == '|' or board[target_pos][j] == '|' or board[source_pos][j] == '@' or board[target_pos][j] == '@':
                    flag = False
                    board[source_pos][j] = "@"
                    if curr_verdict: # to prevent multiple counting
                        n_non_projective_links += 1
                    curr_verdict = False
                else:
                    board[source_pos][j] = "*"
                    board[target_pos][j] = "*"

            # drawing vertical line
            A = min(source_pos, target_pos)
            B = max(source_pos, target_pos)
            for j in range(A+1, B):
                if board[j][dist] == '*' or board[j][dist] == '@':
                    flag = False
                    board[source_pos][j] = "@"
                    if curr_verdict: # to prevent multiple counting
                        n_non_projective_links += 1
                    curr_verdict = False
                else:
                    board[j][dist] = "|"

            # print_board(board)
            # input()

        if flag:
            verdict = "projective"
        else:
            verdict = "non-projective"
        return flag, verdict, float(n_non_projective_links) / float(len(link_info)) # minus one because major claim is not counted



    def get_score(self, score_name):
        """
        Args:
            score_name (str): 

        Returns:
            the corresponding score for score_name
        """
        try:
            return self.scores[self.score_types.index(score_name)]
        except:
            return np.nan


    @staticmethod
    def open_html(path):
        """
        Read tsv from external file

        Args:
            path (str): path to file

        Returns:
            BeautifulSoup
        """
        f = codecs.open(path, 'r', 'utf-8')
        soup= BeautifulSoup(f.read(), 'html.parser')
        return soup


    def __process_html(self, soup):
        """
        Process html file (annotated with TIARA annotation tool) to internal data structure

        Args:
            soup (BeautifulSoup): html document object
        """
        try:
            essay_code = soup.find("h4", id="essay_code").get_text().strip() # current format
        except:
            essay_code = soup.find("h4", id="essay_code_ICNALE").get_text().strip() # old format
        units = []

        unit_annotation = soup.find_all("div", class_="flex-item") # sentences, clause, or clause-like segments (discourse unit)
        for x in unit_annotation:
            unit = DiscourseUnit() # initialization            
            unit.ID = int(x.find('span', class_="sentence-id-number").get_text().strip())
            unit.text = x.find("textarea").get_text().strip()

            target = x.find("span", id="target"+str(unit.ID)).get_text().strip()
            if target != "":
                unit.targetID = int(re.sub("[^0-9]", "", target))
            else:
                unit.targetID = NO_TARGET_CONSTANT 

            unit.rel_name = x.find("span", id="relation"+str(unit.ID)).get_text().strip()

            dropping_flag = x.find("input", id="dropping"+str(unit.ID))["value"]
            if dropping_flag == "non-drop":
                unit.dropping = False
            else:
                unit.dropping = True

            units.append(unit)

        return essay_code, units


    def __process_tsv(self, filepath):
        """
        Process tsv file into internal data structure

        Args:
            filepath (str): tsv filepath
        """
        # use filename as essay_code
        filename, file_extension = os.path.splitext(filepath)
        essay_code = filename.split("/")[-1]

        # open file
        with open(filepath, 'r') as f:
            data = [row for row in csv.reader(f.read().splitlines(), delimiter='\t')]
        del data[0] # delete header
        
        # process
        n_sentences = len(data)
        units = [] # sentences
        for i in range(n_sentences):
            row = data[i]
            unit = DiscourseUnit()
            unit.ID = int(row[1])
            unit.text = row[2]
            unit.targetID = int(row[3]) if row[3]!='' else NO_TARGET_CONSTANT 
            unit.rel_name = row[4]
            unit.dropping = str_to_bool(row[5])
        
            units.append(unit) 
        return essay_code, units


    def __process_file(self, filepath):
        """
        Load a file into internal data structure

        Args:
            filepath (str): 
        """
        filename, file_extension = os.path.splitext(filepath)
        if file_extension == ".html":
            return self.__process_html(Essay.open_html(filepath))
        elif file_extension == ".tsv":
            return self.__process_tsv(filepath)
        else:
            raise Exception('unsupported file', filepath)


    def to_tsv(self):
        """
        Convert the essay to tsv

        Returns:
            str
        """
        header = "essay code" + DELIMITER + "unit id" + DELIMITER + "text" + DELIMITER + "target" + DELIMITER + "relation" + DELIMITER + "drop_flag" + "\n"
        tsv = header
        for i in range(len(self.units)):
            tsv = tsv + self.essay_code + DELIMITER + str(self.units[i]) + "\n"
        return tsv


    def to_tsv_sorted(self):
        """
        Convert the essay to tsv, sorted according to original unit ID

        Returns:
            str
        """
        header = "essay code" + DELIMITER + "unit id" + DELIMITER + "text" + DELIMITER + "target" + DELIMITER + "relation" + DELIMITER + "drop_flag" + "\n"
        tsv = header
        units_copy = deepcopy(self.units)
        units_copy.sort(key=lambda x: x.ID)
        for i in range(len(units_copy)):
            tsv = tsv + self.essay_code + DELIMITER + str(units_copy[i]) + "\n"
        return tsv


    def units_to_tsv(self, essay_code, units):
        """
        Concert the essay to tsv

        Args:
            essay_code (str)
            units: list of DiscourseUnits with a particular ordering sequence

        Returns:
            str
        """
        header = "essay code" + DELIMITER + "unit id" + DELIMITER + "text" + DELIMITER + "target" + DELIMITER + "relation" + DELIMITER + "drop_flag" + "\n"
        tsv = header
        for i in range(len(units)):
            tsv = tsv + essay_code + DELIMITER + str(units[i]) + "\n"
        return tsv


    def get_dropping_sorted(self):
        """
        Get dropping info sorted according to unitID

        Returns:
            list of bool
        """
        drop_flags = []        
        units_copy = deepcopy(self.units)
        units_copy.sort(key=lambda x: x.ID)
        for i in range(len(units_copy)):
            drop_flags.append(units_copy[i].dropping)
        return drop_flags


    def adj_matrix(self):
        """
        Convert the relations in essay into adj matrix

        Returns:
            numpy.ndarray
        """
        n = len(self.units)
        adj_matrix = np.zeros((n, n), dtype="<U5")
        for i in range(n):
            source = self.units[i].ID - 1
            target = self.units[i].targetID - 1
            if target != NO_TARGET_CONSTANT - 1:
                adj_matrix[source][target] = self.units[i].rel_name

        return adj_matrix


    @staticmethod
    def discard_text_repair(sentence):
        """
        Use the original version of the repaired part

        Args:
            sentence (str)

        Return:
            sentence normalised without text repair
        """
        while sentence.find("[") != -1:
            left = sentence.find("[")
            mid = sentence.find("|")
            right = sentence.find("]")
            old = sentence[left+1:mid].strip()
            want_to_replace = sentence[left:right+1]
            sentence = sentence.replace(want_to_replace, " "+old+" ") # give space before and after to avoid tokenization problem
        # remove multiple spaces
        sentence = re.sub(' +', ' ', sentence)
        return sentence.strip()


    @staticmethod
    def use_text_repair(sentence):
        """
        Use the repaired version of the repaired part

        Args:
            sentence (str)

        Return:
            sentence normalised without text repair
        """
        while sentence.find("[") != -1:
            left = sentence.find("[")
            mid = sentence.find("|")
            right = sentence.find("]")
            use = sentence[mid+1:right].strip()
            want_to_replace = sentence[left:right+1]
            sentence = sentence.replace(want_to_replace, " "+use+" ") # give space before and after to avoid tokenization problem
        # remove multiple spaces
        sentence = re.sub(' +', ' ', sentence)
        return sentence.strip()


    @staticmethod
    def list_paths(adj_matrix):
        """
        List all paths in the graph
        :param numpy.ndarray adj_matrix
        :return: list[list(int, str)]
        """
        n_nodes = len(adj_matrix)
        pointed = [False] * n_nodes

        # pointed flag --> no need this due to restatement complication, any kind of node can be a leaf if we take into account restatements
        # for i in range(n_nodes):
        #     for j in range(n_nodes):
        #         if adj_matrix[i][j] != NO_REL_SYMBOL:
        #             pointed[j] = True

        # list paths from any node to its root
        all_paths = []
        for i in range(n_nodes):
            if not pointed[i]:
                visited = [False] * n_nodes
                Essay.parent_traversal(i, visited, n_nodes, adj_matrix, [], all_paths)

        return all_paths


    @staticmethod
    def list_subpaths(adj_matrix):
        """
        List all paths in the graph (including subpath), from leaf to root node
        Discard the subpaths that contain only one element
        :param numpy.ndarray adj_matrix
        :return: list[list[int]]
        """
        paths = Essay.list_paths(adj_matrix)
        retval = []
        for i in range(len(paths)):
            retval.append(paths[i])
            n = len(paths[i])
            for j in range(1, n):
                start = 0
                end = start + j
                while end <= n:
                    retval.append(paths[i][start:end])
                    start += 1
                    end = start + j

        # only take the node ID
        retval_clean = []
        for path in retval:
            path_clean = []
            for n, rel_name in path:
                path_clean.append(n)
            path_clean = tuple(path_clean)
            retval_clean.append(path_clean)

        # take only unique paths
        retval_clean = set(retval_clean)

        # delete subpaths that consist of only one element
        retval_filter = set()
        for e in retval_clean:
            if len(e) > 1:
                retval_filter.add(e)

        return retval_filter


    @staticmethod
    def depth(adj_matrix):
        """
        Return the depth of the structure (adj_matrix)
        
        Args:
            adj_matrix (numpy.ndarray)
        """
        paths = Essay.list_paths(adj_matrix)
        max_depth = -1
        for path in paths:
            if len(path) > max_depth:
                max_depth = len(path)
        return max_depth-1 # because root is 0


    @staticmethod
    def parent_traversal(current_node, visited, n_nodes, adj_matrix, curr_path, all_paths):
        """
        DFS traversal from a node until reaching root, we discard dropped nodes (no incoming and outgoing connection)

        Args:
            current_node (int)
            visited (:obj:`list` of :obj:`bool`)
            n_nodes (int)
            adj_matrix (numpy.ndarray)
            curr_path (:obj:`list` of :obj:`(int, str)`)
            all_paths (:obj:`list` of :obj:`list` of :obj:`(int, str)`): variable to save all found paths, this is MUTABLE
        """
        vis_copy = deepcopy(visited)
        if not vis_copy[current_node]:
            vis_copy[current_node] = True
            outgoing = adj_matrix[current_node]
            flag = False
            for i in range(n_nodes):
                if outgoing[i] != NO_REL_SYMBOL:
                    flag = True
                    path = deepcopy(curr_path)
                    path.append((current_node, outgoing[i]))
                    Essay.parent_traversal(i, vis_copy, n_nodes, adj_matrix, path, all_paths)
            if not flag:
                curr_path.append((current_node, ""))
                if len(curr_path) > 1: # not a dropped node
                    all_paths.append(curr_path)
        else: # cycle detected, this is useful for graph after restatement inference
            if len(curr_path) > 1: # not a dropped node
                all_paths.append(curr_path)


    @staticmethod
    def list_substructures(adj_matrix):
        """
        Generate substructure information (set of nodes: own + descendant) for each node in the matrix
        :param numpy.ndarray adj_matrix
        :return: list[list[int]]
        """
        n_nodes = len(adj_matrix)
        pointed = [False] * n_nodes

        # pointed flag
        for i in range(n_nodes):
            for j in range(n_nodes):
                if adj_matrix[i][j] != NO_REL_SYMBOL:
                    pointed[j] = True

        # list paths from leaf to root
        subtrees = [set()] * n_nodes
        for i in range(n_nodes):
            if not pointed[i]:
                visited = [False] * n_nodes
                Essay.substructure_parsing(i, visited, n_nodes, adj_matrix, set(), subtrees)
        return subtrees


    @staticmethod
    def substructure_parsing(current_node, visited, n_nodes, adj_matrix, curr_subtree, subtrees):
        """
        DFS traversal from a node until reaching root, we USE dropped nodes (no incoming and outgoing connection)
        While traversing, pass the information of own descendants to parent
        :param int current_node
        :param list[bool] visited
        :param int n_nodes
        :param numpy.ndarray adj_matrix
        :param set curr_subtree
        :param list[set[int]] all_paths: variable to save all found paths, this is (MUTABLE)
        """
        vis_copy = deepcopy(visited)
        if not vis_copy[current_node]:
            vis_copy[current_node] = True
            outgoing = adj_matrix[current_node]
            flag = False
            for i in range(n_nodes):
                if outgoing[i] != NO_REL_SYMBOL:
                    flag = True
                    local_subtree = deepcopy(curr_subtree)
                    local_subtree.add(current_node)
                    subtrees[current_node] = subtrees[current_node].union(local_subtree)
                    Essay.substructure_parsing(i, vis_copy, n_nodes, adj_matrix, subtrees[current_node], subtrees)
            if not flag:
                local_subtree = deepcopy(curr_subtree)
                local_subtree.add(current_node)
                if len(local_subtree) > 1: # not a dropped node
                    subtrees[current_node] = subtrees[current_node].union(local_subtree)
        else: # cycle detected, this is useful for graph after restatement inference
            if len(curr_subtree) > 1: # not a dropped node
                subtrees[current_node] = subtrees[current_node].union(curr_subtree)


if __name__ == "__main__":
    essay = Essay("data/test_reordered_random_structure/W_CHN_PTJ0_041_A2_0_EDIT_r.tsv")
    print(essay.essay_code)
    # data = essay.get_pairwise_ordering_data(arg_links_only=True, experiment="MTL") # all pairs
    # data = essay.get_pairwise_ordering_data(arg_links_only=False, experiment="STL") # only pairs connected by arg links
    # print(essay.get_restatement_pairs())
    # print("# instances", len(data))
    # print(data[0])
    # for x in data:
        # print(x[1], x[2])

    # text = "[aku|dia] anak [baik|Nakal]"
    # print(Essay.discard_text_repair(text))
    # print(Essay.use_text_repair(text))

    units, ret_labels = essay.get_ACs_and_rhetorical_categories(order="reordering")
    print(ret_labels)
    for x in units:
        print(x)

