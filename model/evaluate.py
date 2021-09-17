"""
Author: Jan Wira Gotama Putra
"""
from typing import *
from tqdm import tqdm
import time
import argparse
import ast
import itertools
import os
import numpy as np
from collections import Counter

from sklearn.metrics import classification_report, confusion_matrix

from preprocessing.treebuilder import TreeBuilder
from preprocessing.discourseunit import Essay
from model.predict import list_directory
from model.predict import convert_linking_prediction_to_heuristic_baseline
from model.metrics import EdgeRecall, SubstructureAgreement


flatten_list = lambda l: [item for sublist in l for item in sublist]


def list_file(path) -> List[str]:
    """
    List directory existing in path
    """
    return [ os.path.join(path, name) for name in os.listdir(path) if os.path.isfile(os.path.join(path, name)) ]


def open_linking_prediction(directory: str) -> (List, List):
    """
    Open STL predictions

    Args:
        directory (str)

    Returns:
        {
            List,
            List
        }
    """
    with open(directory+"/rel_distances_labels_gold.txt", 'r') as f:
        link_golds = ast.literal_eval(f.readline())
    with open(directory+"/rel_distances_labels_pred.txt", 'r') as f:
        link_preds = ast.literal_eval(f.readline())

    return link_golds, link_preds


def structured_output_quality(links) -> (List, float, float, float):
    """
    Infer component labels automatically from the structure
    """
    component_labels = []
    tree_ratio = 0
    avg_depth = 0
    avg_leaf_prop = 0
    all_depths = []

    n_essays = len(links)

    for i in range(len(links)):
        rep = TreeBuilder(links[i])
        component_labels.append(rep.auto_component_labels(AC_breakdown=True))

        if rep.is_tree():
            tree_ratio += 1

            # evaluate this only when the output forms a tree
            depth, leaf_prop = rep.tree_depth_and_leaf_proportion()
            avg_depth += depth
            all_depths.append(depth)
            avg_leaf_prop += leaf_prop
    
    return component_labels, float(tree_ratio)/float(n_essays), float(avg_depth)/float(tree_ratio), float(avg_leaf_prop)/float(tree_ratio), all_depths


def linking_as_pairwise_classification(gold_links, pred_links):
    """
    Evaluate linking result as binary pairwise classification task (like in Stab and Gurevych)
    This is for comparison only
    
    Args:
        gold_links (List): gold answer per essay
        pred_links (List): predicted answer per essay
    """
    gold_ans = []
    pred_ans = []
    dataset_f1_no_link = []
    dataset_f1_link = []

    for i in range(len(gold_links)): # per essay
        gold_rep = TreeBuilder(gold_links[i])
        pred_rep = TreeBuilder(pred_links[i])
        # It is not clear whether it is better to compute it better at paragraph-level or dataset-level
        gold_ans.extend(flatten_list(gold_rep.get_adj_matrix())) 
        pred_ans.extend(flatten_list(pred_rep.get_adj_matrix()))

        # result per essay
        x = flatten_list(gold_rep.get_adj_matrix())
        y = flatten_list(pred_rep.get_adj_matrix())
        report = classification_report(y_true=x, y_pred=y, output_dict=True)

        if '0' in report.keys():
            f1_no_link = report['0']['f1-score']
        else: # all linked
            f1_no_link = 1.0
        f1_link_exist = report['1']['f1-score']
        dataset_f1_no_link.append(f1_no_link)
        dataset_f1_link.append(f1_link_exist)

    # dataset level
    print("=== Linking as Pairwise Classification (like in Stab and Gurevych) ===")
    print(classification_report(y_true = gold_ans, y_pred=pred_ans, digits=3))
    report = classification_report(y_true=gold_ans, y_pred=pred_ans, output_dict=True)
    f1_no_link = report['0']['f1-score']
    f1_link_exist = report['1']['f1-score']
    return f1_no_link, f1_link_exist, (f1_no_link+f1_link_exist)/2.0 # Kuribayashi confirmed this

    # a = np.average(dataset_f1_no_link)
    # b = np.average(dataset_f1_link)
    # return a, b, (a+b)/2.0


def f1_per_depth(dist_gold: List, dist_prediction: List, max_depth: int):
    """
    Find at which depth prediction mismatches happen (when the output forms a tree)

    Args:
        dist_gold (List): gold answer per essay
        dist_prediction (List): predicted answer per essay
        max_depth (int): max structure depth in the dataset

    Returns:
        tuple, i.e., (list, list, list)
    """
    gold_all_depth = []
    pred_all_depth = []

    for i in range(len(dist_gold)):
        rep_gold = TreeBuilder(dist_gold[i])
        rep_pred = TreeBuilder(dist_prediction[i])

        if rep_pred.is_tree():
            g_depths = rep_gold.node_depths()
            p_depths = rep_pred.node_depths()

            gold_all_depth.append(g_depths)
            pred_all_depth.append(p_depths)
    
    gold_all_depth_flat = flatten_list(gold_all_depth)
    pred_all_depth_flat = flatten_list(pred_all_depth)

    print("=== Node depth prediction (from the linking results) ===")
    print(classification_report(y_true = gold_all_depth_flat, y_pred=pred_all_depth_flat, digits=3))
    report = classification_report(y_true = gold_all_depth_flat, y_pred=pred_all_depth_flat, output_dict=True)
    f1s = []
    for i in range(max_depth):
        try:
            f1s.append(report[str(i)]['f1-score'])
        except:
            f1s.append(0.0)

    return f1s


def depth_distribution(gold_all_depth: List, pred_all_depth: List, MAX_DEPTH: int) -> (List, List):
    """
    Calculate the depth distribution of output

    Args:
        gold_all_depth (List): depths in the gold standard
        pred_all_depth (List): depths in prediction output
        MAX_DEPTH (int)

    Returns:
        (list, list)
    """

    gold_depth_distribution = [0.0] * MAX_DEPTH # from depth 0 to MAX_DEPTH-1
    pred_depth_distribution = [0.0] * MAX_DEPTH 
    for i in range(len(gold_all_depth)):
        gold_depth_distribution[gold_all_depth[i]] += 1
    for i in range(len(pred_all_depth)):
        pred_depth_distribution[pred_all_depth[i]] += 1
    for i in range(len(gold_depth_distribution)):
        gold_depth_distribution[i] /= float(len(gold_all_depth))
        pred_depth_distribution[i] /= float(len(pred_all_depth))

    print("=== Tree Depth distribution ===")
    print("Depth \tGold \tPred")
    for i in range(MAX_DEPTH):
        print("%d \t%.3f \t%.3f" % (i, gold_depth_distribution[i], pred_depth_distribution[i]))
    print()
    
    return gold_depth_distribution, pred_depth_distribution


def MAR_scores(link_golds, link_preds):
    """
    Calculate MAR scores at dataset-level

    Args:
        link_golds (List): gold answer per essay
        link_preds (List): pred answer per essay
    
    Returns:
        (float, float, float, float)

    """
    dataset_MAR_edge = 0
    dataset_MAR_path = 0
    dataset_MAR_dset_exact = 0
    dataset_MAR_dset_partial = 0
    for i in range(len(link_golds)):
        gold_rep = TreeBuilder(link_golds[i])
        pred_rep = TreeBuilder(link_preds[i])
        gold_adj_mat = gold_rep.get_adj_matrix_discourse_unit()
        pred_adj_mat = pred_rep.get_adj_matrix_discourse_unit()

        # MAR calculations per essay
        essay_MAR_edge = EdgeRecall.calc_normal(gold_adj_mat, pred_adj_mat, consider_label=False)
        essay_MAR_path = SubstructureAgreement.subpath_sim(Essay.list_subpaths(gold_adj_mat), Essay.list_subpaths(pred_adj_mat))
        essay_MAR_dset_exact = SubstructureAgreement.substructure_sim_exact(Essay.list_substructures(gold_adj_mat), Essay.list_substructures(pred_adj_mat))
        essay_MAR_dset_partial = SubstructureAgreement.substructure_sim_partial(Essay.list_substructures(gold_adj_mat), Essay.list_substructures(pred_adj_mat))
        
        # dataset level
        dataset_MAR_edge += essay_MAR_edge
        dataset_MAR_path += essay_MAR_path
        dataset_MAR_dset_exact += essay_MAR_dset_exact
        dataset_MAR_dset_partial += essay_MAR_dset_partial

    # average score
    dataset_MAR_edge /= len(link_golds)
    dataset_MAR_path /= len(link_golds)
    dataset_MAR_dset_exact /= len(link_golds)
    dataset_MAR_dset_partial /= len(link_golds)

    return dataset_MAR_edge, dataset_MAR_path, dataset_MAR_dset_exact, dataset_MAR_dset_partial



def get_model_run(model_dir: str):
    """
    Get model run order
    """
    subdir = str(model_dir.split("/")[-1].split("-")[-1])
    return subdir


if __name__ == "__main__":
    # user arguments
    parser = argparse.ArgumentParser(description='Evaluation: linking experiment')
    parser.add_argument(
        '-pred_dir', '--pred_dir', type=str, help='directory saving the prediction results', required=True)
    args = parser.parse_args()

    # model
    model_dirs = list_directory(args.pred_dir)
    print("N models to test %d" % (len(model_dirs)))
    model_dirs.sort()

    # CONSTANT
    FAR_FORWARD = 14
    FAR_BACKWARD = -19
    MAX_DEPTH = 29

    # performance metrics for linking
    f1_macro_link = []
    f1_weighted_link = []
    acc_link = []
    f1_per_distance = []
    for i in range(FAR_FORWARD - FAR_BACKWARD):
        f1_per_distance.append([])

    # performance metrics for node labelling inferred from the structure
    f1_mc = []
    f1_ac_non_leaf = []
    f1_ac_leaf = []
    f1_non_ac = []
    f1_macro_node_label = []

    # performance metrics structured output quality
    tree_ratio = []
    avg_depth = []
    avg_leaf_prop = []
    f1_depths = []
    gold_depth_distribution_all = []
    pred_depth_distribution_all = []

    # performance metrics, structured output, MAR metrics
    MAR_edge = []
    MAR_path = []
    MAR_dset_exact = []
    MAR_dset_partial = []

    # linking as pairwise classification
    f1_no_link = []
    f1_link_exist = []
    f1_macro_pairwise = []

    # iterate over models
    for model_dir in model_dirs:
        print("Opening", model_dir)

        # open linking prediction
        link_golds, link_preds = open_linking_prediction(model_dir)
        link_golds_flat = flatten_list(link_golds)
        link_preds_flat = flatten_list(link_preds)

        # # counter, to confirm the tendency to connect to dist=-1
        # freq = Counter(link_preds_flat).most_common(5)
        # print(freq)
        # for x in freq:
        #     print("Counter %d: %.2lf" % (x[0], x[1]/len(link_preds_flat)*100)) 

        # sequence tagging result
        print("=== Sequence tagging evaluation ===")
        print(classification_report(y_true=link_golds_flat, y_pred=link_preds_flat, digits=3))
        report = classification_report(y_true=link_golds_flat, y_pred=link_preds_flat, output_dict=True)
        f1_macro_link.append(report['macro avg']['f1-score'])
        acc_link.append(report['accuracy'])
        f1_weighted_link.append(report['weighted avg']['f1-score'])

        # f1 per target distance
        for i in range(FAR_FORWARD - FAR_BACKWARD):
            try:
                f1_per_distance[i].append(report[str(FAR_BACKWARD+i)]['f1-score'])
            except: # not available
                f1_per_distance[i].append(0.0)

        # structured output quality
        gold_component_labels, gold_tree_ratio, gold_avg_depth, gold_avg_leaf_prop, gold_all_depth = structured_output_quality(link_golds)
        pred_component_labels, pred_tree_ratio, pred_avg_depth, pred_avg_leaf_prop, pred_all_depth = structured_output_quality(link_preds)
        gold_component_labels_flat = flatten_list(gold_component_labels)
        pred_component_labels_flat = flatten_list(pred_component_labels)

        # depth distribution
        gold_depth_distrib, pred_depth_distrib = depth_distribution(gold_all_depth, pred_all_depth, MAX_DEPTH)
        gold_depth_distribution_all.append(gold_depth_distrib)
        pred_depth_distribution_all.append(pred_depth_distrib)

        # automatic argumentative component identification from linking
        print("=== Automatic argumentative component identification (from link structure) ===")
        print(classification_report(y_true=gold_component_labels_flat, y_pred=pred_component_labels_flat, digits=3))
        report = classification_report(y_true=gold_component_labels_flat, y_pred=pred_component_labels_flat, output_dict=True)
        f1_mc.append(report['major claim']['f1-score'])
        f1_ac_non_leaf.append(report['AC (non-leaf)']['f1-score'])
        f1_ac_leaf.append(report['AC (leaf)']['f1-score'])
        f1_non_ac.append(report['non-AC']['f1-score'])
        f1_macro_node_label.append(report['macro avg']['f1-score']) 

        # tree quality of the predicted structure
        tree_ratio.append(pred_tree_ratio)
        avg_depth.append(pred_avg_depth)
        avg_leaf_prop.append(pred_avg_leaf_prop)

        # f1 per depth
        f1_depths.append(f1_per_depth(link_golds, link_preds, MAX_DEPTH))


        # MAR scores
        dataset_MAR_edge, dataset_MAR_path, dataset_MAR_dset_exact, dataset_MAR_dset_partial = MAR_scores(link_golds, link_preds)
        print("=== MAR scores ===")
        print("MAR edge %.3lf" % (dataset_MAR_edge))
        print("MAR path %.3lf" % (dataset_MAR_path))
        print("MAR dset (exact) %.3lf" % (dataset_MAR_dset_exact))
        print("MAR dset (partial) %.3lf\n" % (dataset_MAR_dset_partial))
        MAR_edge.append(dataset_MAR_edge)
        MAR_path.append(dataset_MAR_path)
        MAR_dset_exact.append(dataset_MAR_dset_exact)
        MAR_dset_partial.append(dataset_MAR_dset_partial)

        # Linking as pairwise classification
        f1_nl, f1_l, f1_lm = linking_as_pairwise_classification(link_golds, link_preds)
        f1_no_link.append(f1_nl)
        f1_link_exist.append(f1_l)
        f1_macro_pairwise.append(f1_lm)


    print("==================================================")
    print("=================                =================")
    print("================= GENERAL RESULT =================")
    print("=================                =================")
    print("==================================================")
    print()

    print("=== Sequence tagging evaluation ===")
    print("Run \tAccuracy \tF1-macro \tF1-weighted")
    for i in range(len(model_dirs)):
        subdir = get_model_run(model_dirs[i])
        print("%s \t%.3lf \t%.3lf \t%.3lf" % (subdir, acc_link[i], f1_macro_link[i], f1_weighted_link[i]))
    print("Average \t%.3lf (%.3lf) \t%.3lf (%.3lf) \t%.3lf (%.3lf)" % (np.average(acc_link), np.std(acc_link), 
                                                                        np.average(f1_macro_link), np.std(f1_macro_link), 
                                                                        np.average(f1_weighted_link), np.std(f1_weighted_link)
                                                                        ))

    print()
    print("=== F1 performance (avg.) on each target distance ===")
    print("Dist \tF1 (avg) \tstdev")
    for i in range(FAR_FORWARD - FAR_BACKWARD):
        print("%d \t%.3lf \t%.3lf" % (FAR_BACKWARD+i, np.average(f1_per_distance[i]), np.std(f1_per_distance[i]) ))
    
    print()
    print("=== Automatic component identification ===")
    print("F1 MC \tF1 AC (non-leaf) \tF1 AC (leaf) \tF1 Non-AC \tF1-Macro")
    for i in range(len(model_dirs)):
        subdir = get_model_run(model_dirs[i])
        print("%s \t%.3lf \t%.3lf \t%.3lf \t%.3lf \t%.3lf" % (subdir, f1_mc[i], f1_ac_non_leaf[i], f1_ac_leaf[i], f1_non_ac[i], f1_macro_node_label[i]))
    print("Average \t%.3lf (%.3lf) \t%.3lf (%.3lf) \t%.3lf (%.3lf) \t%.3lf (%.3lf) \t%.3lf (%.3lf)" % (np.average(f1_mc), np.std(f1_mc), 
                                                                        np.average(f1_ac_non_leaf), np.std(f1_ac_non_leaf), 
                                                                        np.average(f1_ac_leaf), np.std(f1_ac_leaf),
                                                                        np.average(f1_non_ac), np.std(f1_non_ac),
                                                                        np.average(f1_macro_node_label), np.std(f1_macro_node_label)
                                                                        ))

    print()
    print("=== Structured Output Quality ===")
    print("Run \tTree Ratio \tDepth \tLeaf Proportion")
    for i in range(len(model_dirs)):
        subdir = get_model_run(model_dirs[i])
        print("%s \t%.3lf \t%.1lf \t%.3lf" % (subdir, tree_ratio[i], avg_depth[i], avg_leaf_prop[i]))
    print("Average \t%.3lf (%.3lf) \t%.1lf (%.3lf) \t%.3lf (%.3lf)" % (np.average(tree_ratio), np.std(tree_ratio), 
                                                                    np.average(avg_depth), np.std(avg_depth), 
                                                                    np.average(avg_leaf_prop), np.std(avg_leaf_prop)
                                                                    ))

    print()
    print("=== Tree Depth distribution ===")
    depth_d_avg = np.average(np.array(pred_depth_distribution_all), axis=0)
    depth_d_stdev = np.std(np.array(pred_depth_distribution_all), axis=0)
    print("Depth \tAverage % \tstdev")
    for i in range(len(depth_d_avg)):
        print("%d \t%.3f \t%.3f" % (i, depth_d_avg[i], depth_d_stdev[i]))
    print()


    print()
    print("=== F1 per node depth ===")
    depth_performance = np.average(np.array(f1_depths), axis=0)
    depth_stdev = np.std(np.array(f1_depths), axis=0)
    print("Depth \tF1 \tstdev")
    for i in range(len(depth_performance)):
        print("%d \t%.3f \t%.3f" % (i, depth_performance[i], depth_stdev[i]))
    print() 

    print()
    print("=== MAR scores ===")
    print("Run \tMAR_edge \tMAR_path \tMAR_dset_exact \tMAR_dset_partial")
    for i in range(len(model_dirs)):
        subdir = get_model_run(model_dirs[i])
        print("%s \t%.3lf \t%.3lf \t%.3lf \t%.3lf" % (subdir, MAR_edge[i], MAR_path[i], MAR_dset_exact[i], MAR_dset_partial[i]))
    print("Average \t%.3lf (%.3lf) \t%.3lf (%.3lf) \t%.3lf (%.3lf) \t%.3lf (%.3lf)" % (np.average(MAR_edge), np.std(MAR_edge), 
                                                                np.average(MAR_path), np.std(MAR_path), 
                                                                np.average(MAR_dset_exact), np.std(MAR_dset_exact),
                                                                np.average(MAR_dset_partial), np.std(MAR_dset_partial)
                                                                ))
    print()


    print()
    print("=== Linking as Pairwise Classification ===")
    print("Run \tF1 no Link \tF1 link Exist\t F1 Macro")
    for i in range(len(model_dirs)):
        subdir = get_model_run(model_dirs[i])
        print("%s \t%.3lf \t%.3lf \t%.3lf" % (subdir, f1_no_link[i], f1_link_exist[i], f1_macro_pairwise[i]))
    print("Average \t%.3lf (%.3lf) \t%.3lf (%.3lf) \t%.3lf (%.3lf)" % ( np.average(f1_no_link), np.std(f1_no_link), 
                                                                        np.average(f1_link_exist), np.std(f1_link_exist),
                                                                        np.average(f1_macro_pairwise), np.std(f1_macro_pairwise)
                                                                        ))
    print()
