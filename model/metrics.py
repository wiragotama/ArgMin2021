"""
Structure-based inter-annotator agreement metrics
"""
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing.discourseunit import NO_REL_SYMBOL
import math


"""
Mean agreement recall score (our original metrics)
"""
class EdgeRecall:
    @staticmethod
    def calc_normal(g1, g2, consider_label):
        """
        Calculate EdgeRecall score between two graphs
        :param numpy.ndarray g1: reflecting the first graph
        :param numpy.ndarray g2: reflecting the second graph
        :param bool consider_label: True or False. True means we consider the relation label when computing EdgeRecall
        :return: float
        """
        edges_g1 = np.count_nonzero(g1)
        edges_g2 = np.count_nonzero(g2)

        overlap = 0
        n_nodes = len(g1)
        for i in range(n_nodes):
            for j in range(n_nodes):
                if consider_label:
                    if (g1[i][j] != NO_REL_SYMBOL and g2[i][j]!= NO_REL_SYMBOL) and (g1[i][j] == g2[i][j]):
                        overlap += 1
                else:
                    if (g1[i][j] != NO_REL_SYMBOL and g2[i][j]!= NO_REL_SYMBOL):
                        overlap += 1

        r1 = float(overlap) / float(edges_g1)
        r2 = float(overlap) / float(edges_g2)
        return (r1 + r2) / float(2)


    @staticmethod
    def calc_w_inference(g1, inf_g1, g2, inf_g2, consider_label):
        """
        Calculate EdgeRecall score between two graphs, considering the result of inference (restatement and/or double_att)
        :param numpy.ndarray g1: reflecting the first graph
        :param numpy.ndarray inf_g1: the first graph with additional relations as the result of inference
        :param numpy.ndarray g2: reflecting the second graph
        :param numpy.ndarray inf_g1: the second graph with additional relations as the result of inference
        :param bool consider_label: True or False. True means we consider the relation label when computing EdgeRecall
        :return: float
        """
        edges_g1 = np.count_nonzero(g1)
        edges_g2 = np.count_nonzero(g2)

        overlap_r1 = 0
        overlap_r2 = 0
        n_nodes = len(g1)
        for i in range(n_nodes):
            for j in range(n_nodes):
                if consider_label:
                    if (g1[i][j] != NO_REL_SYMBOL and inf_g2[i][j]!= NO_REL_SYMBOL) and (g1[i][j] == inf_g2[i][j]):
                        overlap_r1 += 1 # how much g1 recalls "populated"-g2
                    if (inf_g1[i][j] != NO_REL_SYMBOL and g2[i][j]!= NO_REL_SYMBOL) and (inf_g1[i][j] == g2[i][j]):
                        overlap_r2 += 1 # how much g2 recalls "populated"-g2
                else:
                    if (g1[i][j] != NO_REL_SYMBOL and inf_g2[i][j]!= NO_REL_SYMBOL):
                        overlap_r1 += 1
                    if (inf_g1[i][j] != NO_REL_SYMBOL and g2[i][j]!= NO_REL_SYMBOL):
                        overlap_r2 += 1

        r1 = float(overlap_r1) / float(edges_g1)
        r2 = float(overlap_r2) / float(edges_g2)
        return (r1 + r2) / float(2)


"""
Substructure agreement
"""
class SubstructureAgreement:
    @staticmethod
    def save_div(n, d):
        return float(n) / float(d) if d else 0.0


    @staticmethod
    def subpath_sim(subpaths_1, subpaths_2):
        """
        Substructure similarity based on subpaths
        :param list[list[int]] subpaths_1: subpaths in annotation 1
        :param list[list[int]] subpaths_2: subpaths in annotation 2
        :return: float
        """
        u = subpaths_1.union(subpaths_2)
        f1 = np.zeros(len(u))
        f2 = np.zeros(len(u))
        u = list(u)

        # convert graph into one-hot-vector (based on the precense of subpaths)
        for i in range(len(u)):
            if u[i] in subpaths_1:
                f1[i] = 1
            if u[i] in subpaths_2:
                f2[i] = 1

        score = np.dot(f1, f2) * (np.count_nonzero(f1) + np.count_nonzero(f2)) / (2 * (np.count_nonzero(f1) * np.count_nonzero(f2)))

        if math.isnan(score): # in case of empty set
            return 0.0
        else:
            return score


    @staticmethod
    def substructure_sim_partial(subtrees_1, subtrees_2):
        """
        Subtructure similarity based on descendant nodes (measure on grouping, defined as node + its descendant), PARTIAL match
        :param list[set[int]] subtrees_1: subtrees in annotation 1
        :param list[set[int]] subtrees_2: subtrees in annotation 2
        :return: float
        """
        assert(len(subtrees_1) == len(subtrees_2))
        n = len(subtrees_1)
        f1 = np.zeros(n)
        f2 = np.zeros(n)
        for i in range(n):
            if (subtrees_1[i] == subtrees_2[i]): # exact match, or both are dropped
                f1[i] = 1.0
                f2[i] = 1.0
            else: # partial match
                f1[i] = SubstructureAgreement.save_div( len(subtrees_1[i].intersection(subtrees_2[i])), float(len(subtrees_2[i])))
                f2[i] = SubstructureAgreement.save_div( len(subtrees_2[i].intersection(subtrees_1[i])), float(len(subtrees_1[i])))

        return (np.sum(f1) + np.sum(f2)) / (2.0 * float(n)) # average of average recall


    @staticmethod
    def substructure_sim_exact(subtrees_1, subtrees_2):
        """
        Subtructure similarity based on descendant nodes (measure on grouping, defined as node + its descendant), EXACT match
        :param list[set[int]] subtrees_1: subtrees in annotation 1
        :param list[set[int]] subtrees_2: subtrees in annotation 2
        :return: float
        """
        assert(len(subtrees_1) == len(subtrees_2))
        n = len(subtrees_1)
        f1 = np.zeros(n)
        for i in range(n):
            f1[i] = subtrees_1[i] == subtrees_2[i] # calculate the number of matching pairs

        return float(np.count_nonzero(f1)) / float(len(f1))


    @staticmethod
    def siblinghood_sim_partial(siblinghood_1, siblinghood_2):
        """
        Substructure similarity based on siblinghood, PARTIAL match
        May use the subtree calculation as the metric is basically similar (just change the input)
        :param list[set[int]] siblinghood_1: annotation 1
        :param list[set[int]] siblinghood_2: annotation 2
        :return: float
        """
        return SubstructureAgreement.substructure_sim_partial(siblinghood_1, siblinghood_2)


    @staticmethod
    def siblinghood_sim_exact(siblinghood_1, siblinghood_2):
        """
        Substructure similarity based on siblinghood, EXACT match
        May use the subtree calculation as the metric is basically similar (just change the input)
        :param list[set[int]] siblinghood_1: annotation 1
        :param list[set[int]] siblinghood_2: annotation 2
        :return: float
        """
        return SubstructureAgreement.substructure_sim_exact(siblinghood_1, siblinghood_2)


"""
Graph-based IAA Measure by Kirschner et al. 2015. Linking the Thoughts: Analysis of Argumentation Structures in Scientific Publications. 2nd Workshop on Argumentation Mining.
"""
class KirschnerGraph:
    @staticmethod
    def graph_iaa(adj_matrix1, shortest_path_dist1, adj_matrix2, shortest_path_dist2, mode):
        """
        Calculate graph-based IAA measure
        :param numpy.ndarray adj_matrix1: adjacency matrix of annotation 1
        :param numpy.ndarray shortest_path_dist1: adjacency matrix, containing shortest path distance between all pairs of nodes in annotation 1
        :param numpy.ndarray adj_matrix2: adjacency matrix of annotation 2
        :param numpy.ndarray shortest_path_dist2: adjacency matrix, containing shortest path distance between all pairs of nodes in annotation 2
        :param str mode: {"avg", "f1"} there are two modes proposed in the paper
        :return: float
        """
        n_nodes = len(adj_matrix1)

        n_edges_1 = np.count_nonzero(adj_matrix1)
        n_edges_2 = np.count_nonzero(adj_matrix2)
        sum_of_inverse_1 = 0.0
        sum_of_inverse_2 = 0.0
        for i in range(n_nodes):
            for j in range(n_nodes):
                if adj_matrix1[i][j] != NO_REL_SYMBOL:
                    sum_of_inverse_1 += 1.0 / shortest_path_dist2[i][j]
                if adj_matrix2[i][j] != NO_REL_SYMBOL:
                    sum_of_inverse_2 += 1.0 / shortest_path_dist1[i][j]
        sum_of_inverse_1 /= float(n_edges_1)
        sum_of_inverse_2 /= float(n_edges_2)

        if mode == "avg":
            return (sum_of_inverse_1 + sum_of_inverse_2) / 2.0
        elif mode=="f1":
            return (2.0 * sum_of_inverse_1 * sum_of_inverse_2) / (sum_of_inverse_1 + sum_of_inverse_2)
        else:
            return None

        