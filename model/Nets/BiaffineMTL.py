"""
Author: Jan Wira Gotama Putra

We reference this implementation based on implementation by yzhangcs https://github.com/yzhangcs/biaffine-parser
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
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.trainer import Trainer
from allennlp.data.iterators import DataIterator
from allennlp.common.util import namespace_match
from allennlp.nn.util import sequence_cross_entropy_with_logits
from allennlp.common import Params

from model.datasetreader import *
from model.Nets.BiaffineModules.mlp import MLP
from model.Nets.BiaffineModules.biaffine import Biaffine
from model.Nets.BiaffineModules.bilstm import BiLSTM
from model.Nets.BiaffineModules.dropout import SharedDropout

from preprocessing.treebuilder import TreeBuilder


class BiaffineMTL(Model): 
    """
    Reimplementation of Biaffine BiLSTM model for dependency parsing
    """
    def __init__(self, vocab: Vocabulary, emb_dim: int, reduc_dim: int, lstm_u: int, n_stack: int, fc_u: int, dropout_rate: float, aux_tasks: List[str], torch_device: torch.device, aux_features: List[str] = []) -> None:
        """
        vocab (Vocabulary): AllenNLP vocabulary
        emb_dim (int): sentence embedding dimension
        reduc_dim (int): the number of hidden layer for a dense layer to reduce the embedding dimension
        lstm_u (int): the number of lstm units
        n_stack(int): the number of BiLSTM stack
        fc_u (int): the number of hidden layer for the next dense layer after BiLSTM
        dropout_rate (float): used for all dropouts: (1) sequence dropout, (2) dropout rate for between {BiLSTM and fc_u} and (3) between {fc_u and prediction}
        aux_tasks (List[str]): the list of auxiliary task to train this model
        torch_device (torch.device): where this model supposed to run
        aux_features (List[str]): the list of auxiliary features to train this model, default=[] for compatibility purpose
        """
        super().__init__(vocab)
        # aux task confirmation
        self.POSSIBLE_AUX_TASKS = ["component_labels", "node_depths_labels"] # the list of supported auxiliary task
        for x in aux_tasks:
            if x not in self.POSSIBLE_AUX_TASKS:
                raise Exception(x, "is not supported as an auxiliary task")
        self.user_aux_task = aux_tasks # supplied auxiliary tasks from the user

        # aux features
        self.POSSIBLE_AUX_FEATURES = ["sentence_position"] # the list of supported auxiliary features
        for x in aux_features:
            if x not in self.POSSIBLE_AUX_FEATURES:
                raise Exception(x, "is not supported as an auxiliary feature")
        self.user_aux_features = aux_features

        # number of labels for each aux task
        self.AUX_TASKS_N_LABELS = dict()
        for x in self.POSSIBLE_AUX_TASKS:
            self.AUX_TASKS_N_LABELS[x] = vocab.get_vocab_size(namespace=x)
        
        # the number of linking distances
        self.n_dists = vocab.get_vocab_size(namespace="rel_distances_labels") # the number of output distances in linking

        # input dimensionality reduction
        self.reduc_dim = nn.Linear(emb_dim+len(self.user_aux_features), reduc_dim) # we concat embedding and aux features 
        self.emb_dropout = nn.Dropout(p=dropout_rate)

        # bilstm layer
        self.bilstm = BiLSTM(input_size=reduc_dim,
                            hidden_size=lstm_u,
                            num_layers=n_stack,
                            dropout=dropout_rate)
        self.bilstm_dropout = SharedDropout(p=dropout_rate)

        # MLP layers, representing head (target) and dependent (source)
        self.mlp_arc_h = MLP(n_in=lstm_u*2,
                            n_hidden=fc_u,
                            dropout=dropout_rate)
        self.mlp_arc_d = MLP(n_in=lstm_u*2,
                            n_hidden=fc_u,
                            dropout=dropout_rate)

        # [MAIN TASK] Biaffine transformation, this is the linking prediction
        self.arc_attn = Biaffine(n_in=fc_u,
                                bias_x=True,
                                bias_y=False)

        # [AUX TASK] component labels
        if "component_labels" in self.user_aux_task:
            self.mlp_component_label = nn.Linear(lstm_u * 2, fc_u) # transformation for node labelling
            self.prediction_component_label = nn.Linear(fc_u, self.AUX_TASKS_N_LABELS["component_labels"]) # component labels
            torch.nn.init.xavier_uniform_(self.mlp_component_label.weight)
            torch.nn.init.xavier_uniform_(self.prediction_component_label.weight)

        # [AUX TASK] node depths
        if "node_depths_labels" in self.user_aux_task:
            self.mlp_node_depths = nn.Linear(lstm_u * 2, fc_u) # transformation for node depths
            self.prediction_node_depths = nn.Linear(fc_u, self.AUX_TASKS_N_LABELS["node_depths_labels"]) # node depths
            torch.nn.init.xavier_uniform_(self.mlp_node_depths.weight)
            torch.nn.init.xavier_uniform_(self.prediction_node_depths.weight)

        # linking loss function
        # each element in the prediction (s_arc), can be considered as a multi-class classification logits
        self.linking_loss_function = nn.MultiMarginLoss(reduction='sum') # the original biaffine bilstm paper did not specify the loss function they used, and we follow Kipperwasser (2016) dependency parser to train using Max-Margin criterion

        # for dynamically weight the loss in MTL
        if len(self.user_aux_task) > 0:
            self.etas = [nn.Parameter(torch.zeros(1))] # for linking
            for i in range(len(self.user_aux_task)):
                self.etas.append(nn.Parameter(torch.zeros(1)))
            # defined as 2*log(sigma) https://discuss.pytorch.org/t/how-to-learn-the-weights-between-two-losses/39681/12
            # sigma is initialized as 1, that's why 2*log(sigma)=0

        # for saving model
        self.config = {
            "architecture": "BiaffineMTL",
            "emb_dim": emb_dim,
            "reduc_dim": reduc_dim,
            "lstm_u": lstm_u,
            "n_stack": n_stack,
            "fc_u": fc_u,
            "dropout_rate": dropout_rate,
            "aux_tasks": aux_tasks,
            "aux_features": aux_features
        }
        self.vocab = vocab
        self.torch_device = torch_device


    def __compute_linking_loss(self, s_arc, rel_dists, seq_len): 
        """
        Compute linking loss (average of essay-level loss)

        Args:
            s_arc (torch.Tensor)
            rel_dists (torch.Tensor)
            seq_len (Any)

        Returns:
            (torch.Tensor, torch.Tensor)
        """
        def dist_idx_to_dist(idx):
            return int(self.vocab.get_token_from_index(int(idx), namespace="rel_distances_labels"))
        batch_size = len(rel_dists)

        # gold ans
        gold_ans = []
        for b in range(batch_size):
            non_padded_pred = rel_dists[b][:seq_len[b]].tolist()
            non_padded_pred = [dist_idx_to_dist(x) for x in non_padded_pred] 
            gold_matrix = torch.Tensor(TreeBuilder(non_padded_pred).adj_matrix)
            target = torch.argmax(gold_matrix, dim=-1) # index of the correct label
            if self.torch_device.type=="cuda": # move to device
                target = target.cuda()
            gold_ans.append(target)
            
        # pred ans
        pred_ans = []
        for b in range(batch_size):
            non_padded_pred = s_arc[b, :seq_len[b], :seq_len[b]]
            pred_ans.append(non_padded_pred) 

        # loss
        avg_loss = []
        for b in range(batch_size): # batch_size
            loss = self.linking_loss_function(pred_ans[b], gold_ans[b]) # linking loss per essay
            avg_loss.append(loss) # loss per batch
        avg_loss = torch.mean(torch.stack(avg_loss))
        
        return pred_ans, avg_loss


    def __compute_aux_task_loss(self, pred_logits, gold_labels, seq_len):
        """
        Compute loss for auxiliary task

        Args:
            pred_logits (torch.Tensor)
            gold_labels (torch.Tensor)
            seq_len (Any)

        Returns:
            torch.Tensor
        """
        mask = SeqDatasetReader.get_batch_seq_mask(seq_len)
        if self.torch_device.type=="cuda": # move to device
            mask = mask.cuda()   
        loss = sequence_cross_entropy_with_logits(pred_logits, gold_labels, mask)
        return loss


    def __multi_task_dynamic_loss(self, losses):
        """
        Compute dynamic weighting of task-specific losses during training process, based on homoscedastic uncertainty of tasks
        proposed by Kendall et al. (2018): multi task learning using uncertainty to weigh losses for scene geometry and semantics
        
        References:
        - https://arxiv.org/abs/1705.07115
        - https://github.com/ranandalon/mtl
        - https://discuss.pytorch.org/t/how-to-learn-the-weights-between-two-losses/39681
        - https://github.com/CubiCasa/CubiCasa5k/blob/master/floortrans/losses/uncertainty_loss.py

        Args:
            losses (the list of losses)

        Returns:
            combined loss (torch.Tensor)
        """
        def weighted_loss(loss, eta):
            # print(loss.shape)
            # input()
            return torch.exp(-eta) * loss + torch.log(1+torch.exp(eta))

        assert (len(losses) == len(self.etas))
        for i in range(len(losses)):
            if i==0:
                combined_loss = weighted_loss(losses[i], self.etas[i])
            else:
                combined_loss = combined_loss + weighted_loss(losses[i], self.etas[i])

        return torch.mean(combined_loss)



    def forward(self, 
                essay_code: Any,
                vectors: torch.Tensor,
                rel_distances: torch.Tensor,
                rel_labels: torch.Tensor,
                component_labels: torch.Tensor,
                node_depths: torch.Tensor,
                seq_len: Any) -> Dict:
        """
        Forward pass
        
        Args:
            essay_code (Any)
            vectors (torch.Tensor): of size (batch_size, seq_len, emb_dim)
            rel_distances (torch.Tensor): of size (batch_size, seq_len, output_labels)
            rel_labels (torch.Tensor): of size (batch_size, seq_len, output_labels)
            component_labels (torch.Tensor): of size (batch_size, seq_len, output_labels)
            node_depths (torch.Tensor): of size (batch_size, seq_len, output_labels)
            seq_len (Any)

        Returns:
            Dict
        """
        inp_shape = vectors.shape # (batch_size, seq_len, embeddings)
        # print("sequence sizes", seq_len)

        # sentence vector embedding
        flattened_embeddings = vectors.view(inp_shape[0]*inp_shape[1], -1) # (batch_size * seq_len, embeddings)

        # if auxiliary feature exist
        if "sentence_position" in self.user_aux_features:
            mask = SeqDatasetReader.get_batch_seq_mask(seq_len)
            positional_feature = deepcopy(mask)
            for e in range(len(seq_len)):
                for s in range(seq_len[e]):
                    positional_feature[e][s] = (s+1) / seq_len[e] # we start the positional encoding from "1"
            
            flattened_pos_feature = positional_feature.view(inp_shape[0]*inp_shape[1], -1) # (batch_size * seq_len, embedding size)
            flattened_embeddings = torch.cat((flattened_embeddings, flattened_pos_feature), dim=-1) # concat sentence vector and auxiliary features

        # feature transformation and dimensionality reduction
        reduc_emb = F.leaky_relu(self.reduc_dim(flattened_embeddings), negative_slope=0.1) 
        reduc_emb = self.emb_dropout(reduc_emb) # we try to follow biaffine paper as much as possible, even though this part might be questionable. Is it good to have dropout on pre-trained embeddings?

        # prepare input for LSTM
        bilstm_inp = reduc_emb.view(inp_shape[0], inp_shape[1], self.config["reduc_dim"]) # (batch_size * seq_len, embeddings)
        bilstm_inp = pack_padded_sequence(bilstm_inp, torch.Tensor(seq_len), batch_first=True, enforce_sorted=False) # relu(0) = 0, so we can use padded_sequence here

        # BiLSTM
        bilstm_out, (hn, cn) = self.bilstm(bilstm_inp)
        bilstm_out, _ = pad_packed_sequence(bilstm_out, batch_first=True)

        # dropout after bilstm
        bilstm_out = self.bilstm_dropout(bilstm_out)

        # apply MLPs to BiLSTM output states
        arc_h = self.mlp_arc_h(bilstm_out) # head
        arc_d = self.mlp_arc_d(bilstm_out) # dependent

        # get arc scores from the bilinear attention
        s_arc = self.arc_attn(arc_d, arc_h)
    
        # [MAIN TASK] linking
        pred_linking, linking_loss = self.__compute_linking_loss(s_arc, rel_distances, seq_len)
        losses = [linking_loss]
        
        # return value to user
        output = {  "seq_len": seq_len,
                    "pred_linking": pred_linking}

        # [AUX TASK] component labels
        if "component_labels" in self.user_aux_task:
            flattened_bilstm_out = bilstm_out.contiguous().view(inp_shape[0]*inp_shape[1], -1) # (batch_size * seq_len, hidden units)
            dense_out = F.relu(self.mlp_component_label(flattened_bilstm_out))
            pred_component_labels_logits = self.prediction_component_label(dense_out)
            pred_component_labels_softmax = F.softmax(pred_component_labels_logits, dim=-1)

            # reshape prediction to compute loss and output
            pred_component_labels_logits = pred_component_labels_logits.view(inp_shape[0], inp_shape[1], self.AUX_TASKS_N_LABELS["component_labels"])
            pred_component_labels_softmax = pred_component_labels_softmax.view(inp_shape[0], inp_shape[1], self.AUX_TASKS_N_LABELS["component_labels"])
            # add to output
            output["pred_component_labels_softmax"] = pred_component_labels_softmax

            # compute loss
            component_label_loss = self.__compute_aux_task_loss(pred_component_labels_logits, component_labels, seq_len)
            losses.append(component_label_loss)

        # [AUX TASK] node depths
        if "node_depths_labels" in self.user_aux_task:
            flattened_bilstm_out = bilstm_out.contiguous().view(inp_shape[0]*inp_shape[1], -1) # (batch_size * seq_len, hidden units)
            dense_out = F.relu(self.mlp_node_depths(flattened_bilstm_out))
            pred_node_depths_logits = self.prediction_node_depths(dense_out)
            pred_node_depths_softmax = F.softmax(pred_node_depths_logits, dim=-1)

             # reshape prediction to compute loss and output
            pred_node_depths_logits = pred_node_depths_logits.view(inp_shape[0], inp_shape[1], self.AUX_TASKS_N_LABELS["node_depths_labels"])
            pred_node_depths_softmax = pred_node_depths_softmax.view(inp_shape[0], inp_shape[1], self.AUX_TASKS_N_LABELS["node_depths_labels"])
            # add to output
            output["pred_node_depths_softmax"] = pred_node_depths_softmax

            # compute loss
            node_depth_loss = self.__compute_aux_task_loss(pred_node_depths_logits, node_depths, seq_len)
            losses.append(node_depth_loss)

        # output to user
        if len(losses) == 1: # linking only
            output["loss"] = linking_loss
        else:
            output["loss"] = self.__multi_task_dynamic_loss(losses)

        return output


