from components.segmentation import compute_segmentation
from components.semantic_merge import extract_segmentation_masks
import itertools as it
from operator import itemgetter

import networkx as nx
import nltk
import numpy as np
import tensorflow as tf
from os.path import join

from components.PSPNet.model import load_color_label_dict
from components.path import WEIGHTS_DIR

content_segmentation, style_segmentation = compute_segmentation("/home/skhan22/PoeticStyleTransfer/data2/summer_lake/2108832-1366x768-[DesktopNexus.com].jpg", "/home/skhan22/PoeticStyleTransfer/data2/summer_lake/2504-1271617411m0fL.jpg")

print(content_segmentation.shape)
print(style_segmentation.shape)


# load color - label mapping
color_label_dict = load_color_label_dict()
label_color_dict = {label: color for color, labels in color_label_dict.items() for label in labels}
colors = color_label_dict.keys()

# Extract the boolean mask for every color
content_masks = extract_segmentation_masks(content_segmentation, colors)
style_masks = extract_segmentation_masks(style_segmentation, colors)

