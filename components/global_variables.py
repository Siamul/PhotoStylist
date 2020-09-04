import tensorflow as tf
import components.VGG19.model
from components.PSPNet.model import PSPNet50, load_color_label_dict

vgg = components.VGG19.model

weight_restorer = vgg.load_weights()

image_placeholder = tf.compat.v1.placeholder(tf.float32, shape=[1, None, None, 3])

#vgg19 = vgg.VGG19ConvSub(image_placeholder)



