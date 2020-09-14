import argparse
import json
import os
from datetime import datetime

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

import components.NIMA.model as nima
import components.VGG19.model as vgg
from components.matting import compute_matting_laplacian
from components.segmentation import compute_segmentation
from components.global_variables import vgg, weight_restorer, image_placeholder
from components.semantic_merge import merge_anps, reduce_dict
from components.utilities import mask_for_tf_with_color, extract_segmentation_masks, load_text, save_text, calculate_gram_matrix_with_mask, load_image, save_image, load_text, save_text



def style_transfer(content_image, color_to_gram_dict, content_masks, init_image, result_dir, timestamp, args):
    print("Style transfer started")
    style_conv_grams = []
    for i in range(5):
        style_gram = {}
        for color in color_to_gram_dict.keys():
            style_gram[color] = color_to_gram_dict[color][i]
        style_conv_grams.append(style_gram)

    
    content_image = vgg.preprocess(content_image)

    global weight_restorer

    image_placeholder = tf.compat.v1.placeholder(tf.float32, shape=[1, None, None, 3])

    with tf.compat.v1.variable_scope("", reuse=True):
        vgg19 = vgg.VGG19ConvSub(image_placeholder)


    with tf.compat.v1.Session() as sess:
        transfer_image = tf.Variable(init_image)
        transfer_image_vgg = vgg.preprocess(transfer_image)
        transfer_image_nima = nima.preprocess(transfer_image)

        sess.run(tf.compat.v1.global_variables_initializer())
        weight_restorer.init(sess)
        content_conv4_2 = sess.run(fetches=vgg19.conv4_2, feed_dict={image_placeholder: content_image})

        with tf.compat.v1.variable_scope("", reuse=True):
            vgg19 = vgg.VGG19ConvSub(transfer_image_vgg)

        content_loss = calculate_layer_content_loss(content_conv4_2, vgg19.conv4_2)
        style_conv1_1_gram, style_conv2_1_gram, style_conv3_1_gram, style_conv4_1_gram, style_conv5_1_gram = style_conv_grams

        style_loss = (1. / 5.) * calculate_layer_style_loss(style_conv1_1_gram, vgg19.conv1_1, content_masks)
        style_loss += (1. / 5.) * calculate_layer_style_loss(style_conv2_1_gram, vgg19.conv2_1, content_masks)
        style_loss += (1. / 5.) * calculate_layer_style_loss(style_conv3_1_gram, vgg19.conv3_1, content_masks)
        style_loss += (1. / 5.) * calculate_layer_style_loss(style_conv4_1_gram, vgg19.conv4_1, content_masks)
        style_loss += (1. / 5.) * calculate_layer_style_loss(style_conv5_1_gram, vgg19.conv5_1, content_masks)

        photorealism_regularization = calculate_photorealism_regularization(transfer_image_vgg, content_image)

        nima_loss = compute_nima_loss(transfer_image_nima)

        content_loss = args.content_weight * content_loss
        style_loss = args.style_weight * style_loss
        photorealism_regularization = args.regularization_weight * photorealism_regularization
        nima_loss = args.nima_weight * nima_loss

        total_loss = content_loss + style_loss + photorealism_regularization + nima_loss

        tf.compat.v1.summary.scalar('Content loss', content_loss)
        tf.compat.v1.summary.scalar('Style loss', style_loss)
        tf.compat.v1.summary.scalar('Photorealism Regularization', photorealism_regularization)
        tf.compat.v1.summary.scalar('NIMA loss', nima_loss)
        tf.compat.v1.summary.scalar('Total loss', total_loss)

        summary_op = tf.compat.v1.summary.merge_all()
        summary_writer = tf.compat.v1.summary.FileWriter(os.path.join(os.path.dirname(__file__), 'logs/{}'.format(timestamp)),
                                               sess.graph)

        iterations_dir = os.path.join(result_dir, "iterations")
        os.mkdir(iterations_dir)

        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=args.adam_learning_rate, beta1=args.adam_beta1,
                                           beta2=args.adam_beta2, epsilon=args.adam_epsilon)

        train_op = optimizer.minimize(total_loss, var_list=[transfer_image])
        sess.run(adam_variables_initializer(optimizer, [transfer_image]))

        min_loss, best_image = float("inf"), None
        for i in range(args.iterations + 1):
            _, result_image, loss, c_loss, s_loss, p_loss, n_loss, summary = sess.run(
                fetches=[train_op, transfer_image, total_loss, content_loss, style_loss, photorealism_regularization,
                         nima_loss, summary_op])

            summary_writer.add_summary(summary, i)

            if i % args.print_loss_interval == 0:
                print(
                    "Iteration: {0:5} \t "
                    "Total loss: {1:15.2f} \t "
                    "Content loss: {2:15.2f} \t "
                    "Style loss: {3:15.2f} \t "
                    "Photorealism Regularization: {4:15.2f} \t "
                    "NIMA loss: {5:15.2f} \t".format(i, loss, c_loss, s_loss, p_loss, n_loss))

            if loss < min_loss:
                min_loss, best_image = loss, result_image

            #if i % args.intermediate_result_interval == 0:
               # save_image(best_image, os.path.join(iterations_dir, "iter_{}.png".format(i)))

        return best_image


def adam_variables_initializer(adam_opt, var_list):
    adam_vars = [adam_opt.get_slot(var, name)
                 for name in adam_opt.get_slot_names()
                 for var in var_list if var is not None]
    adam_vars.extend(list(adam_opt._get_beta_accumulators()))
    return tf.compat.v1.variables_initializer(adam_vars)


def compute_nima_loss(image):
    model = nima.get_nima_model(image)

    def mean_score(scores):
        scores = tf.squeeze(scores)
        si = tf.range(1, 11, dtype=tf.float32)
        return tf.reduce_sum(input_tensor=tf.multiply(si, scores), name='nima_score')

    nima_score = mean_score(model.output)

    nima_loss = tf.identity(10.0 - nima_score, name='nima_loss')
    return nima_loss


def calculate_layer_content_loss(content_layer, transfer_layer):
    return tf.reduce_mean(input_tensor=tf.math.squared_difference(content_layer, transfer_layer))


def calculate_layer_style_loss(color_style_grams, transfer_layer, content_masks):
    # scale masks to current layer
    content_size = tf.TensorShape(transfer_layer.shape[1:3])

    feature_map_count = np.float32(transfer_layer.shape[3])
    feature_map_size = np.float32(transfer_layer.shape[1]) * np.float32(transfer_layer.shape[2])

    means_per_channel = []
    for color in content_masks.keys():
        transfer_gram_matrix = calculate_gram_matrix_with_mask(transfer_layer, mask_for_tf_with_color(content_masks, color))
        style_gram_matrix = color_style_grams[color]

        mean = tf.reduce_mean(input_tensor=tf.math.squared_difference(style_gram_matrix, transfer_gram_matrix))
        means_per_channel.append(mean / (2 * tf.square(feature_map_count) * tf.square(feature_map_size)))

    style_loss = tf.reduce_sum(input_tensor=means_per_channel)

    return style_loss


def calculate_photorealism_regularization(output, content_image):
    # normalize content image and out for matting and regularization computation
    content_image = content_image / 255.0
    output = output / 255.0

    # compute matting laplacian
    matting = compute_matting_laplacian(content_image[0, ...])

    # compute photorealism regularization loss
    regularization_channels = []
    for output_channel in tf.unstack(output, axis=-1):
        channel_vector = tf.reshape(tf.transpose(a=output_channel), shape=[-1])
        matmul_right = tf.sparse.sparse_dense_matmul(matting, tf.expand_dims(channel_vector, -1))
        matmul_left = tf.matmul(tf.expand_dims(channel_vector, 0), matmul_right)
        regularization_channels.append(matmul_left)

    regularization = tf.reduce_sum(input_tensor=regularization_channels)
    return regularization


def change_filename(dir_name, filename, suffix, extension=None):
    path, ext = os.path.splitext(filename)
    if extension is None:
        extension = ext
    return os.path.join(dir_name, path + suffix + extension)


def write_metadata(dir, args, load_segmentation):
    # collect metadata and write to transfer dir
    meta = {
        "init": args.init,
        "iterations": args.iterations,
        "content": args.content_image,
        "style": args.style_text,
        "content_weight": args.content_weight,
        "style_weight": args.style_weight,
        "regularization_weight": args.regularization_weight,
        "nima_weight": args.nima_weight,
        "adjective_threshold": args.adjective_threshold,
        "noun_threshold": args.noun_threshold,
        "load_segmentation": load_segmentation,
        "adam": {
            "learning_rate": args.adam_learning_rate,
            "beta1": args.adam_beta1,
            "beta2": args.adam_beta2,
            "epsilon": args.adam_epsilon
        }
    }
    filename = os.path.join(dir, "meta.json")
    with open(filename, "w+") as file:
        file.write(json.dumps(meta, indent=4))


if __name__ == "__main__":
    """Parse program arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--content_image", type=str, help="content image path", default="content.jpg")
    parser.add_argument("--style_text", type=str, help="style text file path", default="style.txt")
    parser.add_argument("--output_image", type=str, help="Output image path, default: result.jpg",
                        default="result.jpg")
    parser.add_argument("--iterations", type=int, help="Number of iterations, default: 500",
                        default=500)
    parser.add_argument("--intermediate_result_interval", type=int,
                        help="Interval of iterations until a intermediate result is saved., default: 100",
                        default=100)
    parser.add_argument("--print_loss_interval", type=int,
                        help="Interval of iterations until the current loss is printed to console., default: 1",
                        default=1)
    parser.add_argument("--content_weight", type=float,
                        help="Weight of the content loss., default: 1",
                        default=1)
    parser.add_argument("--style_weight", type=float,
                        help="Weight of the style loss., default: 100",
                        default=100)
    parser.add_argument("--regularization_weight", type=float,
                        help="Weight of the photorealism regularization.",
                        default=10 ** 4)
    parser.add_argument("--nima_weight", type=float,
                        help="Weight for nima loss.",
                        default=0)
    parser.add_argument("--adam_learning_rate", type=float,
                        help="Learning rate for the adam optimizer., default: 1.0",
                        default=1.0)
    parser.add_argument("--adam_beta1", type=float,
                        help="Beta1 for the adam optimizer., default: 0.9",
                        default=0.9)
    parser.add_argument("--adam_beta2", type=float,
                        help="Beta2 for the adam optimizer., default: 0.999",
                        default=0.999)
    parser.add_argument("--adam_epsilon", type=float,
                        help="Epsilon for the adam optimizer., default: 1e-08",
                        default=1e-08)
    parser.add_argument("--adjective_threshold", type=float, help="Threshold for adjective matching, default: 0.2",
                        default=0.2)
    parser.add_argument("--noun_threshold", type=float, help="Threshold for noun matching, default: 0.4",
                        default=0.4)
    parser.add_argument("--evaluation", type=bool, help="Script activation for evaluation, default: False",
                        default=False)
    init_image_options = ["noise", "content", "style"]
    parser.add_argument("--init", type=str, help="Initialization image (%s).", default="content")
    parser.add_argument("--gpu", help="comma separated list of GPU(s) to use.", default="0")

    args = parser.parse_args()
    assert (args.init in init_image_options)
    vgg19 = vgg.VGG19ConvSub(image_placeholder)
    # For more information on the similarity metrics: http://gsi-upm.github.io/sematch/similarity/#word-similarity

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.evaluation == False:

        timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M')
        style_text = load_text(args.style_text)

        result_dir = 'result_'+args.content_image.split("/")[-1].split('.')[0]+'_'+style_text
        os.mkdir(result_dir)

        # check if manual segmentation masks are available
        content_segmentation_filename = change_filename('', args.content_image, '_seg', '.png')
        load_segmentation = os.path.exists(content_segmentation_filename)

        write_metadata(result_dir, args, load_segmentation)  

        """Check if image files exist"""
        for path in [args.content_image, args.style_text]:
            if path is None or not os.path.isfile(path):
                print("File {} does not exist.".format(path))
                exit(0)

        content_image = load_image(args.content_image)

        # use existing if available
        if (load_segmentation):
            print("Load segmentation from files.")
            content_segmentation_image = cv2.imread(content_segmentation_filename)
            content_segmentation_masks = extract_segmentation_masks(content_segmentation_image)
        else:
            print("Create segmentation.")
            content_segmentation = compute_segmentation(args.content_image)
            cv2.imwrite(change_filename(result_dir, args.content_image, '_seg_raw', '.png'), content_segmentation)
   

        content_segmentation_masks, color_to_gram_dict, anp_results = merge_anps(content_segmentation, style_text, 
                                                                       args.adjective_threshold, args.noun_threshold, result_dir)

        cv2.imwrite(change_filename(result_dir, args.content_image, '_seg', '.png'),
                    reduce_dict(content_segmentation_masks, content_image))

        if args.init == "noise":
            random_noise_scaling_factor = 0.0001
            random_noise = np.random.randn(*content_image.shape).astype(np.float32)
            init_image = vgg.postprocess(random_noise * random_noise_scaling_factor).astype(np.float32)
        elif args.init == "content":
            init_image = content_image
        elif args.init == "style":
            init_image = style_image
        else:
            print("Init image parameter {} unknown.".format(args.init))
            exit(0)

        result = style_transfer(content_image, color_to_gram_dict,  content_segmentation_masks, init_image, result_dir, timestamp, args)
        save_image(result, os.path.join(result_dir, "pst_result.png"))

    else:
        for i in range(51):

            j = i+1

            content_image_name = './eval/content'+str(j)+'.jpg'
            content_image = load_image(content_image_name)
            content_segmentation = compute_segmentation(content_image_name)

            style_file = './eval/style'+str(j)+'.txt'

            with open(style_file, 'r') as sfile:
                for line in sfile:
                    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M')

                    style_text = line.strip()

                    result_dir = 'result_'+'content'+str(j)+'_'+style_text
                    os.mkdir(result_dir)

                    content_segmentation_masks, color_to_gram_dict, anp_results = merge_anps(content_segmentation, style_text,
                                                                       args.adjective_threshold, args.noun_threshold, result_dir)

                    init_image = content_image
                    
                    result = style_transfer(content_image, color_to_gram_dict,  content_segmentation_masks, init_image, result_dir, timestamp, args)
                    save_image(result, os.path.join(result_dir, "pst_result.png"))


