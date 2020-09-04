import itertools as it
from operator import itemgetter

import networkx as nx
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from os.path import join

from components.PSPNet.model import load_color_label_dict
from components.path import WEIGHTS_DIR
from components.anp_matcher import generate_gram_matrices, filter_content_masks
from components.utilities import get_unique_colors_from_image, extract_segmentation_masks


def color_tuples_to_label_list_tuples(color_tuples, color_label_dict):
    return it.chain.from_iterable(
        it.product(color_label_dict[first], color_label_dict[second]) for (first, second) in color_tuples)


# Replace colors in dictionary of color -> mask.
def replace_colors_in_dict(color_mask_dict, replacement_colors):
    new_color_mask_dict = {}
    for color, mask in color_mask_dict.items():
        new_color = replacement_colors[color] if color in replacement_colors else color
        # Merge masks if color already exists
        new_color_mask_dict[new_color] = np.logical_or(mask, new_color_mask_dict[
            new_color]) if new_color in new_color_mask_dict else mask

    return new_color_mask_dict


def merge_anps(content_segmentation, style_text, adj_thres, noun_thres, result_dir):
    print('Finding gram matrices for each resolution according to ANPs matching each noun in image')
    # load color - label mapping
    color_label_dict = load_color_label_dict()
    label_color_dict = {label: color for color, labels in color_label_dict.items() for label in labels}
    colors = color_label_dict.keys()

    #Extract the boolean mask for every color
    content_masks = extract_segmentation_masks(content_segmentation, colors)
    content_colors = content_masks.keys()

    #Generate gram matrices for each label according to ANP
    style_gram_matrices, anp_results = generate_gram_matrices(content_colors, style_text, color_label_dict, label_color_dict, adj_thres, noun_thres, result_dir)

    #Discard nouns in image not obtained from ANP list even after semantic matching
    style_colors = style_gram_matrices.keys()
    content_masks = filter_content_masks(content_masks, style_colors)
    
    assert(frozenset(style_gram_matrices.keys()) == frozenset(content_masks.keys()))
    
    return content_masks, style_gram_matrices, anp_results

def reduce_dict(dict, image):
    _, h, w, _ = image.shape
    arr = np.zeros((h, w, 3), int)
    for k, v in dict.items():
        I, J = np.where(v)
        arr[I, J] = k[::-1]
    return arr


def annotate_label_similarity(labels_to_compare, similarity_metric):
    return [(wns.word_similarity(l1, l2, similarity_metric), (l1, l2)) for (l1, l2) in labels_to_compare]


def get_labels_to_compare(label_lists_to_compare):
    return it.chain.from_iterable(it.product(l1, l2) for (l1, l2) in label_lists_to_compare)

