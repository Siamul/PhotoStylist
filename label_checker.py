from components.PSPNet.model import load_color_label_dict
import sys

color_label_dict = load_color_label_dict()

print(color_label_dict[eval(sys.argv[1])])

