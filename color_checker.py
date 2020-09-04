from components.PSPNet.model import load_color_label_dict
import sys

color_label_dict = load_color_label_dict()
label_color_dict = {label: color for color, labels in color_label_dict.items() for label in labels}

print(label_color_dict[sys.argv[1]])

