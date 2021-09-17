"""
by  Jan Wira Gotama Putra

Given a sequence of sentences, find the distance from source to target sentence
"""
import ast
import numpy as np
from os import listdir
from os.path import isfile, join
from abc import ABC, abstractmethod
from copy import deepcopy
import random
import torch
from torch import tensor
import scipy
from typing import *
from functools import partial
from overrides import overrides

from allennlp.data import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.data.tokenizers import Token
from allennlp.nn import util as nn_util

from allennlp.data.fields import TextField, MetadataField, ArrayField, LabelField, SequenceLabelField, ListField, MultiLabelField
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.iterators import BasicIterator, BucketIterator

from preprocessing.pretrained_transformer_tokenizer import PretrainedTransformerTokenizer
from preprocessing.pretrained_transformer_indexer import PretrainedTransformerIndexer


def tonp(tsr): return tsr.detach().cpu().numpy() # for seamless interaction between gpu and cpu


class SeqDatasetReader(DatasetReader):
    """
    Dataset structure for linking experimental setting
    An instance represents an essay
    """

    def __init__(self, use_percentage: float=1.0) -> None: 
        """
        Args:
            use_percentage (float): how many (percent) of the essays you want to use (this is useful for cutting the number of samples for test run); must be between 0 and 1
        """
        super().__init__(lazy=False)
        if use_percentage > 1 or use_percentage < 0:
            raise Exception("[ERROR] use_percentage variable must be between [0,1]")
        self.use_percentage = use_percentage


    @overrides
    def text_to_instance(self,
                        essay_code: str,
                        vectors: np.ndarray, 
                        rel_distances: np.ndarray,
                        rel_labels: np.ndarray,
                        component_labels: np.ndarray,
                        node_depths: np.ndarray,
                        seq_len: int) -> Instance:
        """
        Args:
            essay_code (str): essay code of the corresponding Instance
            vectors (np.ndarray): target sentence (encoded)
            rel_distances (np.ndarray): target sentence distance (pre-processed)
            rel_labels (np.ndarray): relation label (as a source)
            component_labels (np.ndarray): component label (as a source)
            node_depths (np.ndarray): the position of sentence in the hierarchical structure
            seq_len (int): sequence length of the current instance

        Returns:
            Instance
        """
        # meta-data
        essay_code_field = MetadataField(essay_code)
        fields = {"essay_code": essay_code_field}

        # to be used for sequence mask
        # during training, we will have (batch_size, seq_len (# sentences), emb_dim)
        # since the number of sentences are different for each essay, we need some masking when feeding it into the network
        seq_len_field = MetadataField(seq_len)
        fields["seq_len"] = seq_len_field
        
        # use extracted embedding
        list_emb_field = []
        for emb in vectors:
            list_emb_field.append(ArrayField(emb))
        list_emb_field = ListField(list_emb_field)
        fields["vectors"] = list_emb_field
        ref_seq = list_emb_field # required for SequenceLabelField

        # distances (main task)
        # AllenNLP: We recommend you use a namespace ending with 'labels' or 'tags'
        rel_dist_field = SequenceLabelField(rel_distances, label_namespace="rel_distances_labels", sequence_field=ref_seq) # automatic padding uses "0" as padding
        fields["rel_distances"] = rel_dist_field

        # relation label (aux task)
        rel_label_field = SequenceLabelField(rel_labels, label_namespace="rel_labels", sequence_field=ref_seq) # automatic padding uses "0" as padding
        fields["rel_labels"] = rel_label_field

        # component label (aux task)
        component_label_field = SequenceLabelField(component_labels, label_namespace="component_labels", sequence_field=ref_seq) # automatic padding uses "0" as padding
        fields["component_labels"] = component_label_field

        # node depths (aus task)
        node_depth_field = SequenceLabelField(node_depths, label_namespace="node_depths_labels", sequence_field=ref_seq) # automatic padding uses "0" as padding
        fields["node_depths"] = node_depth_field

        return Instance(fields)
    

    @overrides
    def _read(self, directory: str) -> Iterator[Instance]:
        """
        Args:
            directory (str): containing the dataset
            use (float): how many (percent) of the essays you want to use (this is useful for cutting the number of samples for test run); must be between 0 and 1
        
        Returns:
            Iterator[Instance]
        """
        # file checking
        source_files = [join(directory, f) for f in listdir(directory) if isfile(join(directory, f))]
        source_files.sort()
        flag, essay_codes = self.__source_files_checking(source_files)

        # open files if passed the checking
        if flag:
            if self.use_percentage < 1:
                essay_codes = random.sample(essay_codes, int(len(essay_codes) * self.use_percentage))
                essay_codes.sort()

            # read all files in the directory
            for essay_code in essay_codes:
                vectors, rel_distances, rel_labels, component_labels, node_depths = self.__open_essay(directory, essay_code)
                yield self.text_to_instance(
                    essay_code,
                    vectors, 
                    rel_distances, 
                    rel_labels, 
                    component_labels, 
                    node_depths,
                    len(node_depths)
                )


    def __source_files_checking(self, source_files: List[str]) -> (bool, List[str]):
        """
        Check whether the source files is complete
        Definition: each essay is represented by two files 
            - ".source_target_sentences"
            - ".source_target_sentences_embedding" (source and target texts already in embedding form)
            - ".source_target_rels" (relation between source -> target)

        Check if each unique essay (according to filename) has those three files

        Args:
            source_files (:obj:`list` of :obj:`str`)
        
        Returns:
            bool (True or False)
            List[str], unique filenames
        """
        # get all unique essay codes and existing files
        unique_names = set()
        filecodes = []
        for x in source_files:
            if (".DS_Store" not in x) and (".gitignore" not in x):
                filecode = x.split("/")[-1]
                essay_code = filecode.split(".")[0]

                unique_names.add(essay_code)
                filecodes.append(filecode)

        # check if for each essay code, there are three corresponding files 
        flag = True
        for x in unique_names:
            if not ((x + ".rel_distances" in filecodes) and
                    (x + ".rel_labels" in filecodes) and
                    (x + ".component_labels" in filecodes) and
                    (x + ".sentences" in filecodes) and
                    (x + ".vectors" in filecodes) and
                    (x + ".node_depths" in filecodes)):
                flag = False
                raise Exception("[Error] essay", x, "has incomplete files")

        # for ease of debugging
        unique_names = list(unique_names)
        unique_names.sort()

        return flag, unique_names


    def __open_essay(self, directory: str, essay_code: str) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        """
        Open essay information

        Args:
            directory (str)
            essay_code (str)

        Returns:
            {
                numpy.ndarray,
                numpy.ndarray,
                numpy.ndarray,
                numpy.ndarray,
                numpy.ndarray,
            }
        """
        text_file = directory + essay_code + ".sentences"
        vector_file = directory + essay_code + ".vectors"
        rel_dist_file = directory + essay_code + ".rel_distances"
        rel_label_file = directory + essay_code + ".rel_labels"
        component_file = directory + essay_code + ".component_labels"
        node_depth_file = directory + essay_code + ".node_depths"
        
        
        with open(vector_file, 'r') as f:
            vectors = np.array(ast.literal_eval(f.readline()), dtype=float)
        with open(rel_dist_file, 'r') as f:
            rel_distances = np.array(ast.literal_eval(f.readline()), dtype=str)
        with open(rel_label_file, 'r') as f:
            rel_labels = np.array(ast.literal_eval(f.readline()), dtype=str)
        with open(component_file, 'r') as f:
            component_labels = np.array(ast.literal_eval(f.readline()), dtype=str)
        with open(node_depth_file, 'r') as f:
            node_depths = np.array(ast.literal_eval(f.readline()), dtype=str)
        
        # checking
        assert(len(vectors) == len(rel_distances))
        assert(len(rel_distances) == len(rel_labels))
        assert(len(rel_distances) == len(component_labels))
        assert(len(rel_distances) == len(node_depths))
       
        return vectors, rel_distances, rel_labels, component_labels, node_depths


    @staticmethod
    def get_batch_seq_mask(seq_len: torch.Tensor) -> torch.Tensor:
        """
        Create a mask for feeding batch of sequences into LSTM

        Args:
            seq_len (Tensor): of (batch_size); containing the correct sequence length (without padding)
        """
        max_len = max(seq_len)
        batch_mask = []
        for x in seq_len:
            one = [1] * x
            pads = [0] * (max_len-x)
            seq_mask = one + pads
            batch_mask.append(seq_mask)

        return torch.FloatTensor(batch_mask)



"""
Example
"""
if __name__ == "__main__": 
    working_dir = "data/ICNALE-SBERT/test/"
    
    # reading dataset
    reader = SeqDatasetReader(
        use_percentage=1.0,
    )

    train_ds = reader.read(working_dir)
    print(len(train_ds))
    print(vars(train_ds[0].fields["essay_code"]))
    print(vars(train_ds[0].fields["vectors"]))
    print(vars(train_ds[0].fields["rel_distances"]))
    print(vars(train_ds[0].fields["rel_labels"]))
    print(vars(train_ds[0].fields["component_labels"]))
    print(vars(train_ds[0].fields["node_depths"]))
    print(vars(train_ds[0].fields["seq_len"]))

    # prepare vocabulary
    vocab = Vocabulary.from_instances(train_ds) # somehow, this is a must

    # batch-ing
    iterator = BasicIterator(batch_size=8)
    iterator.index_with(vocab) # this is a must for consistency reason
    batch = next(iter(iterator(train_ds, shuffle=False))) # shuffle = False just for checking
    print(batch)
    print("vectors", batch['vectors'].shape) # shape should be (batch_size * max_seq_len in the batch)
    print("rel_distances", batch['rel_distances'].shape)
    print("rel_labels", batch['rel_labels'].shape)
    print("component_labels", batch['component_labels'].shape)
    print("node_depths", batch['node_depths'].shape)
    print("seq_len", batch['seq_len'])

    # sequence mask
    sequence_mask = SeqDatasetReader.get_batch_seq_mask(batch['seq_len'])
    print("sequence mask", sequence_mask)
    print("sequence mask", sequence_mask.shape)

    # mapping label back
    print("MAPPING TRIAL")
    print(batch['essay_code'][0])
    rel_distance = batch['rel_distances'][0]
    rel_label = batch['rel_labels'][0]
    component_label = batch['component_labels'][0]
    node_depth = batch['node_depths'][0]

    for i in range(batch['vectors'].shape[1]):
        if sequence_mask[0][i]==1:
            print("unit id:", i+1)
            print("dist:", vocab.get_token_from_index(int(rel_distance[i]), namespace="rel_distances_labels"))
            print("rel label:", vocab.get_token_from_index(int(rel_label[i]), namespace="rel_labels"))
            print("comp label:", vocab.get_token_from_index(int(component_label[i]), namespace="component_labels"))
            print("depth:", vocab.get_token_from_index(int(node_depth[i]), namespace="node_depths_labels"))
            print()

    # test save
    vocab.save_to_files("vocabulary/")
    print(vocab.get_vocab_size(namespace="rel_labels"))

