"""
Author: Jan Wira Gotama Putra
"""
from typing import *
from tqdm import tqdm
import time
import numpy as np
import os
import json
import codecs

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from allennlp.models import Model
from allennlp.data import Instance
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.iterators import DataIterator
from allennlp.common.util import namespace_match
from allennlp.nn import util as nn_util

from model.Nets.BiaffineMTL import *
from model.model_functions import *
from model.datasetreader import tonp
from model.EdmondMST import MSTT
from preprocessing.treebuilder import TreeBuilder


def get_rank_order(attn_matrix):
    """
    Args:
        attn_matrix (numpy.ndarray): a matrix where element [i,j] denotes the probability (or score) of sentence-i points to sentence-j (j as a target)

    Returns:
        rank_order (list[list]): we have the rank of each node per row, meaning how likely sentence-i points to sentence-j (in terms of ranking)
    """

    idx = np.argsort((-attn_matrix), axis=-1) # gives the index of the largest element, e.g., idx[0] is the index of the 0-th largest element 
    # adjust the idx matrix, so we now have the rank of each node (per row), this is easier for debugging
    rank_order = np.zeros((len(idx), len(idx)), dtype=int)
    for i in range(len(idx)):
        for j in range(len(idx[i])):
            rank_order[i, idx[i,j]] = int(j)

    return rank_order


def run_MST(rank_order, weight_matrix, verdict="min"):
    """
    Perform an MST algorithm on the attn_matrix (directed graph)

    Args:
        rank_order (list[list]): how likely sentence-i points to sentence-j (in temrs of ranking)
        weight_matrix (numpy.ndarray): matrix containing the weight of edges 
        verdict (str): minimum ("min") or maximum ("max") spanning tree

    Returns:
        dist_interpretation (list): the distance between each node to its target sentence
    """
    diag = np.diag(rank_order)
    root_candidate = np.argmin(diag) # the most probable node that points to itself
    min_val = min(diag) # the weight (rank order) of a node points to itself
    non_ACs = set()
    if min_val == 0: # there might be multiple plausible root candidates, but we only use the first node (the node that appears the first in the text) and regards the rest as non-ACS
        for i in range(len(diag)):
            if diag[i] == min_val and i!=root_candidate:
                non_ACs.add(i) 

    # list of edges for MST
    MST = MSTT()
    for i in range(len(weight_matrix)):
        for j in range(len(weight_matrix)):
            if not(i in non_ACs) and not(j in non_ACs):
                MST.add_edge(i, weight_matrix[i,j], j) # source, weight, target

    # run MST algorithm
    if verdict == "min":
        mst_arcs = MST.min_spanning_arborescence(root_candidate)
    else:
        mst_arcs = MST.max_spanning_arborescence(root_candidate)

    # distance interpretation
    dist_interpretation = [0] * len(rank_order)
    for arc in mst_arcs:
        dist_interpretation[arc.tail] = arc.head - arc.tail

    # tree_rep = TreeBuilder(dist_interpretation)
    # assert (tree_rep.is_tree() == True)
    # print('not tree after post-processing!', dist_interpretation) 
    
    return dist_interpretation



class PredictorBiaffine:
    def __init__(self, 
                model: Model, 
                iterator: DataIterator,
                cuda_device: int=-1) -> None:
        self.model = model
        self.iterator = iterator
        self.cuda_device = cuda_device
        self.tree_from_argmax = 0 # for a reference only, how many outputs form a tree by using argmax (without MST)


    def namespace_idx_to_true_label(self, namespace, idx):
        """
        Args:
            namespace (str): the task being predicted
            idx (int): denoting the index of the highest-scored element (most probable output from the logits or softmax prediction vector)
        """
        if namespace == "rel_distances_labels":
            return int(self.model.vocab.get_token_from_index(int(idx), namespace=namespace))
        else:
            return self.model.vocab.get_token_from_index(int(idx), namespace=namespace)


    def _unpack_model_batch_prediction(self, batch, namespace, coerce_tree=False) -> np.ndarray:
        """
        Interpret prediction result per batch
        coerce_tree = True if you want to ensure that the output forms a tree

        Args:
            batch (torch.Tensor)
            namespace (str): the task being predicted
            coerce_tree (boolean): true if we want to coerce the output to form a tree, this is for linking only

        Returns:
            np.ndarray
        """
        out_dict = self.model(**batch)
        if namespace=="rel_distances_labels":
            return self.__predict_linking(out_dict, coerce_tree=True)
        else:
            return self.__predict_aux_task(batch, out_dict, namespace)


    def __predict_linking(self, out_dict, coerce_tree=True):
        """
        Linking prediction

        Args:
            out_dict (dictionary): output from the model
            coerce_tree (boolean): true if we want to coerce the output to form a tree

        Returns:
            np.ndarray
        """
        pred_matrix = out_dict["pred_linking"]
        batch_interpretation = []
        for es in range(len(pred_matrix)):
            essay_pred = tonp(pred_matrix[es])
           
            # decoding using simple argmax
            essay_pred = np.argmax(essay_pred, axis=-1)
            dist_interpretation = []
            for i in range(len(essay_pred)):
                dist_interpretation.append(essay_pred[i]-i)

            # check if the output is a tree
            rep = TreeBuilder(dist_interpretation)
            # build a tree if the output from argmax does not form a tree
            if (not rep.is_tree()) and (coerce_tree==True): 
                # run MINIMUM spanning tree
                attn_matrix = tonp(pred_matrix[es])
                attn_matrix = np.array(attn_matrix)
                rank_order = get_rank_order(attn_matrix)
                dist_interpretation = run_MST(rank_order, rank_order, verdict="min") # --> use rank as the weight, "minimum" spanning tree, lower_rank number in rank is better
            else:
                self.tree_from_argmax += 1 # for a reference only, how many outputs form a tree by using argmax (without MST)

            # add the decoding result to the batch result
            batch_interpretation.append(dist_interpretation)

        return batch_interpretation


    def _reset_tree_argmax_count(self):
        self.tree_from_argmax = 0


    def __predict_aux_task(self, batch, out_dict, namespace):
        """
        Prediction for aux task

        Args:
            batch (dictionary): input to the model
            out_dict (dictionary): output from the model
            namespace (str): the task being predicted

        Returns:
            np.ndarray
        """
        if namespace == "component_labels":
            pred_softmax = out_dict["pred_component_labels_softmax"]
        elif namespace == "node_depths_labels":
            pred_softmax = out_dict["pred_node_depths_softmax"]
        else:
            raise Exception(namespace, "as aux task is not implemented or does not exist")

        retval = []
        for es in range(len(pred_softmax)):
            essay_lvl_result = []
            max_seq_len = batch["seq_len"][es]

            # iterate each sentence in the essay, s is the index of the current sentence
            for s in range(max_seq_len):
                # simple decoding using argmax for aux task
                curr_label_softmax = pred_softmax[es][s] # essay es, sentence s
                pred_idx = np.argmax(curr_label_softmax)
                pred_label = self.namespace_idx_to_true_label(namespace, pred_idx)

                # essay-level result
                essay_lvl_result.append(pred_label)

            # batch lvl result
            retval.append(essay_lvl_result)
        return retval


    def _unpack_gold_batch_prediction(self, batch: np.ndarray, namespace: str) -> List:
        """
        Only use predictions without padding

        Args:
            batch (np.ndarray): input batch
            namespace (str): the task being predicted

        Returns:
            List
        """
        output = []
        if namespace == "rel_distances_labels":
            key = "rel_distances"
        elif namespace == "component_labels":
            key = namespace
        elif namespace == "node_depths_labels":
            key = "node_depths"
        else:
            raise Exception(namespace, "as aux task is not implemented or does not exist")

        batch_pred = tonp(batch[key])
        seq_len = batch["seq_len"]

        for b in range(len(batch_pred)):
            non_padded_pred = batch_pred[b][:seq_len[b]].tolist()
            non_padded_pred = [self.namespace_idx_to_true_label(namespace, x) for x in non_padded_pred] 
            output.append(non_padded_pred)
        return output
    

    def predict(self, ds: Iterable[Instance], namespace: str, coerce_tree=True) -> np.ndarray:
        """
        Generate prediction result
        Args:
            ds (Iterable): test set
            namespace (str): the task we are currently predicted
            coerce_tree (boolean): true if we want to coerce the output to form a tree, this is for linking only

        Returns:
            np.ndarray
        """
        pred_generator = self.iterator(ds, num_epochs=1, shuffle=False)
        self.model.eval()
        pred_generator_tqdm = tqdm(pred_generator,
                                   total=self.iterator.get_num_batches(ds)) # what if the the valid/test data contain label that does not exist in the training data --> workaround for the vocab
        preds = []
        golds = []
        with torch.no_grad():
            for batch in pred_generator_tqdm:
                batch = nn_util.move_to_device(batch, self.cuda_device)

                # prediction result
                preds.extend(self._unpack_model_batch_prediction(batch, namespace, coerce_tree=coerce_tree))
                
                # gold data
                golds.extend(self._unpack_gold_batch_prediction(batch, namespace))

        return preds, golds







class PredictorUKPMissingLink:
    """
    Predictor for "recovering UKP missing links" experiment
    """
    def __init__(self, 
                model: Model, 
                iterator: DataIterator,
                cuda_device: int=-1) -> None:
        self.model = model
        self.iterator = iterator
        self.cuda_device = cuda_device
        self.tree_from_argmax = 0 # for a reference only, how many outputs form a tree by using argmax (without MST)


    def namespace_idx_to_true_label(self, namespace, idx):
        """
        Args:
            namespace (str): the task being predicted
            idx (int): denoting the index of the highest-scored element (most probable output from the logits or softmax prediction vector)
        """
        if namespace == "rel_distances_labels":
            return int(self.model.vocab.get_token_from_index(int(idx), namespace=namespace))
        else:
            return self.model.vocab.get_token_from_index(int(idx), namespace=namespace)


    def _unpack_model_batch_prediction(self, batch, namespace, batch_gold_links, coerce_tree=True) -> np.ndarray:
        """
        Interpret prediction result per batch
        coerce_tree = True if you want to ensure that the output forms a tree

        Args:
            batch (torch.Tensor)
            batch_gold_links (torch.Tensor)
            namespace (str): the task being predicted
            coerce_tree (boolean): true if we want to coerce the output to form a tree, this is for linking only

        Returns:
            np.ndarray
        """
        out_dict = self.model(**batch)
        if namespace=="rel_distances_labels":
            return self.__predict_linking(out_dict, batch_gold_links, coerce_tree)
        else:
            return self.__predict_aux_task(batch, out_dict, namespace)


    def __predict_linking(self, out_dict, batch_gold_links, coerce_tree=True):
        """
        Linking prediction

        Args:
            out_dict (dictionary): output from the model
            batch_gold_links: gold links of the current batch
            coerce_tree (boolean): true if we want to coerce the output to form a tree

        Returns:
            np.ndarray
        """
        pred_matrix = out_dict["pred_linking"]
        batch_interpretation = []
        for es in range(len(pred_matrix)):
            
            # gold result
            gold_link = batch_gold_links[es]
            rep_gold = TreeBuilder(gold_link)
            gold_component_labels = rep_gold.auto_component_labels()
            # print("gold link:", gold_link)
            # print("gold component label:", gold_component_labels)
            # print()

            # essay pred
            essay_pred = tonp(pred_matrix[es])
            # modify essay pred using the links in gold standard 
            for i in range(len(gold_component_labels)):
                if gold_component_labels[i] != 'non-AC':
                    essay_pred[i][i+gold_link[i]] = 1000000000 # max score so we can have partial gold answer
            # print(essay_pred)
            # print()

            # decoding using simple argmax
            essay_pred_argmax = np.argmax(essay_pred, axis=-1)
            dist_interpretation = []
            for i in range(len(essay_pred_argmax)):
                dist_interpretation.append(essay_pred_argmax[i]-i)
            # print("dist interpretation:", dist_interpretation)
            # print()

            # check if the output is a tree
            rep = TreeBuilder(dist_interpretation)
            # build a tree if the output from argmax does not form a tree
            if (not rep.is_tree()) and (coerce_tree==True): 
                # run MINIMUM spanning tree
                attn_matrix = essay_pred # use the modified prediction result
                attn_matrix = np.array(attn_matrix)
                rank_order = get_rank_order(attn_matrix)
                dist_interpretation_mst = run_MST(rank_order, rank_order, verdict="min") # --> use rank as the weight, "minimum" spanning tree, lower_rank number in rank is better
            
                # modified; we enforce the gold links here if it is different
                for i in range(len(gold_component_labels)):
                    if gold_component_labels[i] != 'non-AC':
                        dist_interpretation_mst[i] = gold_link[i]

                # check if the forced inference currently forms a tree; we only need the MST results for non-AC part (of gold)
                rep_mst = TreeBuilder(dist_interpretation_mst)
                if rep_mst.is_tree():
                    dist_interpretation = dist_interpretation_mst
                else:
                    dist_interpretation = gold_link # forget about the inference, use the gold links
            else:
                self.tree_from_argmax += 1 # for a reference only, how many outputs form a tree by using argmax (without MST)

            # add the decoding result to the batch result
            # print("after MST interpretation:", dist_interpretation)
            # print()
            batch_interpretation.append(dist_interpretation)

            # just checking
            # rep = TreeBuilder(dist_interpretation)
            # print("result:", dist_interpretation)
            # print("pred component label:", rep.auto_component_labels())
            # print()

        return batch_interpretation


    def _reset_tree_argmax_count(self):
        self.tree_from_argmax = 0


    def __predict_aux_task(self, batch, out_dict, namespace):
        """
        Prediction for aux task

        Args:
            batch (dictionary): input to the model
            out_dict (dictionary): output from the model
            namespace (str): the task being predicted

        Returns:
            np.ndarray
        """
        if namespace == "component_labels":
            pred_softmax = out_dict["pred_component_labels_softmax"]
        elif namespace == "node_depths_labels":
            pred_softmax = out_dict["pred_node_depths_softmax"]
        else:
            raise Exception(namespace, "as aux task is not implemented or does not exist")

        retval = []
        for es in range(len(pred_softmax)):
            essay_lvl_result = []
            max_seq_len = batch["seq_len"][es]

            # iterate each sentence in the essay, s is the index of the current sentence
            for s in range(max_seq_len):
                # simple decoding using argmax for aux task
                curr_label_softmax = pred_softmax[es][s] # essay es, sentence s
                pred_idx = np.argmax(curr_label_softmax)
                pred_label = self.namespace_idx_to_true_label(namespace, pred_idx)

                # essay-level result
                essay_lvl_result.append(pred_label)

            # batch lvl result
            retval.append(essay_lvl_result)
        return retval


    def _unpack_gold_batch_prediction(self, batch: np.ndarray, namespace: str) -> List:
        """
        Only use predictions without padding

        Args:
            batch (np.ndarray): input batch
            namespace (str): the task being predicted

        Returns:
            List
        """
        output = []
        if namespace == "rel_distances_labels":
            key = "rel_distances"
        elif namespace == "component_labels":
            key = namespace
        elif namespace == "node_depths_labels":
            key = "node_depths"
        else:
            raise Exception(namespace, "as aux task is not implemented or does not exist")

        batch_pred = tonp(batch[key])
        seq_len = batch["seq_len"]

        for b in range(len(batch_pred)):
            non_padded_pred = batch_pred[b][:seq_len[b]].tolist()
            non_padded_pred = [self.namespace_idx_to_true_label(namespace, x) for x in non_padded_pred] 
            output.append(non_padded_pred)
        return output
    

    def predict(self, ds: Iterable[Instance], namespace: str, coerce_tree=True) -> np.ndarray:
        """
        Generate prediction result
        Args:
            ds (Iterable): test set
            namespace (str): the task we are currently predicted
            coerce_tree (boolean): true if we want to coerce the output to form a tree, this is for linking only

        Returns:
            np.ndarray
        """
        assert (namespace=="rel_distances_labels")
        pred_generator = self.iterator(ds, num_epochs=1, shuffle=False)
        self.model.eval()
        pred_generator_tqdm = tqdm(pred_generator,
                                   total=self.iterator.get_num_batches(ds)) # what if the the valid/test data contain label that does not exist in the training data --> workaround for the vocab
        preds = []
        golds = []
        with torch.no_grad():
            for batch in pred_generator_tqdm:
                batch = nn_util.move_to_device(batch, self.cuda_device)

                # gold data
                batch_gold_links = self._unpack_gold_batch_prediction(batch, namespace)
                golds.extend(batch_gold_links)

                # prediction result
                preds.extend(self._unpack_model_batch_prediction(batch, namespace, batch_gold_links, coerce_tree=coerce_tree))

        return preds, golds



