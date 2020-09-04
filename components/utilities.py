import numpy as np
import tensorflow as tf
from PIL import Image

def mask_for_tf_with_color(segmentation_mask, color):
   return tf.expand_dims(tf.expand_dims(tf.constant(segmentation_mask[color].astype(np.float32)), 0), -1)


def get_unique_colors_from_image(image):
    h, w, c = image.shape
    assert (c == 3)
    vec = np.reshape(image, (h * w, c))
    unique_colors = np.unique(vec, axis=0)
    return [tuple(color) for color in unique_colors]

def extract_segmentation_masks(segmentation, colors=None):
    if colors is None:
        # extract distinct colors from segmentation image
        colors = get_unique_colors_from_image(segmentation)
        colors = [color[::-1] for color in colors]

    return {color: mask for (color, mask) in
            ((color, np.all(segmentation.astype(np.int32) == color[::-1], axis=-1)) for color in colors) if
            mask.max()}

def calculate_gram_matrix_with_mask(convolution_layer, mask):
    layer_size = tf.TensorShape(convolution_layer.shape[1:3])
    #mask = np.array(Image.fromarray(mask).resize(layer_size, resample=Image.BILINEAR))
    mask = tf.image.resize(mask, size = layer_size, method=tf.image.ResizeMethod.BILINEAR)
    matrix = tf.reshape(convolution_layer, shape=[-1, convolution_layer.shape[3]])
    mask_reshaped = tf.reshape(mask, shape=[matrix.shape[0], 1])
    matrix_masked = matrix * mask_reshaped
    return tf.matmul(matrix_masked, matrix_masked, transpose_a=True)


def load_image(filename):
    if(filename.endswith('.jpg')):
        image = np.array(Image.open(filename).convert('RGB'), dtype=np.float32)
        image = np.expand_dims(image, axis=0)
    else:
        image = None
    return image


def save_image(image, filename):
    image = image[0, :, :, :]
    image = np.clip(image, 0, 255.0)
    image = np.uint8(image)

    result = Image.fromarray(image)
    result.save(filename)

def load_text(filename):
    with open(filename, 'r') as file:
        str = file.read().replace('\n', ' ')
    return str

def save_text(text, filename):
    with open(filename, 'w') as file:
        return file.write(text)
