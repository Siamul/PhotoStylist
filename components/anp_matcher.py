from flair.models import SequenceTagger
from flair.data import Sentence
from gensim.parsing.preprocessing import remove_stopwords
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
import numpy as np
from numpy.linalg import norm
import os
import wget
import zipfile
import sys
from tqdm import tqdm
from operator import itemgetter
import random
from components.global_variables import vgg, weight_restorer, image_placeholder
from PIL import Image
from components.segmentation import compute_segmentation
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from components.utilities import mask_for_tf_with_color, get_unique_colors_from_image, extract_segmentation_masks, calculate_gram_matrix_with_mask, load_image, save_image
import cv2
#from color_transfer import color_transfer

dataset_location = './ToyDataset'

class tag:
    def __init__(self, text, pos):
        self.text = text
        self.pos_ = pos

def adjnounfinder(text, results):
    tagger = SequenceTagger.load('upos')
    sentence = Sentence(remove_stopwords(text.lower()))
    tagger.predict(sentence)
    string = sentence.to_tagged_string()
    print("Result of Parts-of-Speech tagging with flair: ", end = "")
    print(string)
    results += string + '\n'
    taglist = []
    stringparts = string.split(' ')
    for i in range(len(stringparts)):
        if stringparts[i] == '<ADJ>':
            taglist.append(tag(stringparts[i-1], 'ADJ'))
        elif stringparts[i] == '<PROPN>':
            taglist.append(tag(stringparts[i-1], 'PROPN'))
        elif stringparts[i] == '<NOUN>':
            taglist.append(tag(stringparts[i-1], 'NOUN'))
    return taglist, results

def text_anp_extract(input, results):
    #print(remove_stopwords(input.lower()))
    string, results = adjnounfinder(remove_stopwords(input.lower()), results)
    anps = set()
    adjectives = set()
    nouns = set()
    for i in range(len(string)):
        token = string[i]
        print(token.text + ' ' + token.pos_)
        if token.pos_ == 'ADJ':
            adjectives.add(token.text)
        if token.pos_ == 'PROPN' or token.pos_ == 'NOUN':
            nouns.add(token.text)
            j = i
            k = i
            anp = ''
            while j > 0:
                j -= 1
                adjtoken = string[j]
                if(adjtoken.pos_ == 'ADJ'):
                    anp = str(string[j])+'_'+str(string[i])
                    break
                if(k < len(string)-1):
                    k += 1
                    adjtoken = string[k]
                    if(adjtoken.pos_ == 'ADJ'):
                        anp = str(string[k])+'_'+str(string[i])
                        break
            if j == 0:
                while(k < len(string)-1):
                    k += 1
                    adjtoken = string[k]
                    if(adjtoken.pos_ == 'ADJ'):
                        anp = str(string[k])+'_'+str(string[i])
            anps.add(anp)
    return anps, adjectives, results


def bar_progress(current, total, width=80):
  progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
  # Don't use print() as it will print in new line every time.
  sys.stdout.write("\r" + progress_message)
  sys.stdout.flush()

def load_Word2Vec_model():

    if(not os.path.isfile('./glove.6B.zip')):
        print("Acquiring the GloVe model:")
        url = 'http://nlp.stanford.edu/data/glove.6B.zip'
        wget.download(url, './', bar = bar_progress)

    if(not os.path.isfile('./glove.6B.300d.txt')):
        with zipfile.ZipFile('./glove.6B.zip') as zf:
            print('extracting glove.6B.300d.txt from glove.6B.zip')
            try:
                zf.extract('glove.6B.300d.txt')
                print('extraction complete')
            except Exception as e:
                print(str(e))
                pass

    if(not os.path.isfile('glove.6B.300d.txt.word2vec')):
        print('Converting glove.6B.300d.txt to gensim word2vec model')
        glove2word2vec('glove.6B.300d.txt', 'glove.6B.300d.txt.word2vec')

    return KeyedVectors.load_word2vec_format('./glove.6B.300d.txt.word2vec')

def cosine_similarity(a, b):
    return np.inner(a, b)/(norm(a)*norm(b))

def word_similarity(word1, word2, model):
    if word1 in model.vocab and word2 in model.vocab:
        vec1 = model[word1]
        vec2 = model[word2]
        return cosine_similarity(vec1, vec2)
    else:
        return -1  #ignoring words not in vocab

def load_noun_adj_dict():
    if(not os.path.exists('./hf_ademvsocommonANPs.txt')):
        url = 'https://www.dropbox.com/s/fwh3na3knxddyw1/hf_ademvsocommonANPs.txt?dl=1'
        wget.download(url, './')
    f = open('./hf_ademvsocommonANPs.txt')
    datasetANPs = []
    for anp in f:
        datasetANPs.append(anp.rstrip())
    f.close()
    nounadj = {}
    for anp in datasetANPs:
        anpparts = anp.split('_')
        if anpparts[1] in nounadj.keys():
            nounadj[anpparts[1]].append(anpparts[0])
        else:
            nounadj[anpparts[1]] = [anpparts[0]]
    return nounadj

def match_anp(anps, nounadj, model, adj_thres, noun_thres, results):
    matchedanps = set()
    for anp in anps:
        anpparts = anp.split('_')
        textadj = anpparts[0]
        textnoun = anpparts[1]
        textgen_noun_pairs = [(textnoun, gennoun, word_similarity(textnoun, gennoun, model)) for gennoun in nounadj.keys()]
        bestnounmatch = max(textgen_noun_pairs, key = itemgetter(2))
        textgen_adj_pairs = [(textadj, genadj, word_similarity(textadj, genadj, model)) for genadj in nounadj[bestnounmatch[1]]]
        bestadjmatch = max(textgen_adj_pairs, key = itemgetter(2))
        if(bestnounmatch[2]>noun_thres and bestadjmatch[2]>adj_thres):
            matchedanps.add(bestadjmatch[1]+'_'+bestnounmatch[1])
    print("Resultant ANPs from matching text with mvso dataset: ", end ="")
    print(matchedanps)
    results+='text anps: '+str(matchedanps)+'\n'
    return matchedanps, results


def image_anp_extract(inounlist, text, nounadj, model, adj_thres, noun_thres, results):
    tanps, tadjs, results = text_anp_extract(text, results)
    matchedanps, results = match_anp(tanps, nounadj, model, adj_thres, noun_thres, results)
    matchedanpsimg = [anp for anp in matchedanps if anp.split('_')[1] in inounlist]
    matchednounimg = [anp.split('_')[1] for anp in matchedanps if anp.split('_')[1] in inounlist]
    inounset = set(inounlist) - set(matchednounimg)
    filt_tadjs = [tadj for tadj in tadjs if tadj in model.vocab]
    tadjvec = sum([model[tadj] for tadj in filt_tadjs])/len(filt_tadjs)
    gnounset = set(nounadj.keys())
    cnounset = inounset.intersection(gnounset)
    imageanps = set()
    for cnoun in cnounset:
        cnounadjs = nounadj[cnoun]
        distadjlist = [(cnounadj, cosine_similarity(tadjvec, model[cnounadj])) for cnounadj in cnounadjs if cosine_similarity(tadjvec, model[cnounadj])>adj_thres]
        if len(distadjlist) != 0:
            bestadj = max(distadjlist, key = itemgetter(1))
            imageanps.add(bestadj[0]+'_'+cnoun)
    print("Resultant ANPs from finding adjective for nouns in image according to text: ", end="")
    print(list(imageanps))
    results+='only image anps: '+str(imageanps)+'\n'
    return list(imageanps) + matchedanpsimg, results



def generate_gram_matrices(content_colors, style_text, color_label_dict, label_color_dict, adj_thres, noun_thres, result_dir):

    content_labels = []
    for color in content_colors:
        content_labels.append(color_label_dict[color][0])

    print(content_labels)
    
    #load noun to adjective mappings dictionary
    noun_adj_dict = load_noun_adj_dict()

    #load the Word2Vec model with GloVe embeddings
    w2v_model = load_Word2Vec_model()

    results = ''

    #cross-match the anps from text and nouns from images to find the list of anps for style transfer
    anp_list, results = image_anp_extract(content_labels, style_text, noun_adj_dict, w2v_model, adj_thres, noun_thres, results)

    print('Common ANPs found from cross-matching image and text: ', end ='')
    print(anp_list)
    results+='cross-matched anps: '+str(anp_list)+'\n'
    results+='-------------------------------------\n'
    #create the folder to save selected style images
    save_directory = os.path.join(result_dir, 'style images')
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)
    
    #get color to gram matrices dictionary from generator according to anp_list
    return generate_color_gram_dict(anp_list, label_color_dict, save_directory, results)

def generate_color_gram_dict(anp_list, label_color_dict, save_dir, results):
    
    color_gram_dict = {}
    with tf.compat.v1.variable_scope("", reuse=True):
        vgg19 = vgg.VGG19ConvSub(image_placeholder)

    for anp in anp_list:
        label = anp.split('_')[1]
        req_color = label_color_dict[label]
        gram_matrices_for_anp, results = get_gram_from_generator(anp, req_color, save_dir, results, vgg19)
        color_gram_dict[req_color] = gram_matrices_for_anp
    
    return color_gram_dict, results

#first = True
#first_selected_image = np.zeros((3,3,3), np.uint8)

def get_gram_from_generator(anp, color, save_dir, results, vgg19):   #fake generator implementation
                                    #just randomly picks an image from the ANP dataset and gets the gram matrices for it
    
    image_names = []
    for image_name in os.listdir(os.path.join(dataset_location, anp)):
        if(image_name.endswith('.jpg')):
            image_names.append(image_name)

    rand_indices = list(range(len(image_names)))
    random.shuffle(rand_indices)

    selected_image = os.path.join(dataset_location.rstrip(), anp.rstrip(), image_names[rand_indices[0]].rstrip())

    print('loading '+selected_image)

    image_segmentation = compute_segmentation(selected_image)
    image_masks = extract_segmentation_masks(image_segmentation)
    
    i = 1
    while not color in image_masks.keys():

        selected_image = os.path.join(dataset_location, anp, image_names[rand_indices[i]].rstrip())
        print('previous image didnt work loading' + selected_image)

        image_segmentation = compute_segmentation(selected_image)
        image_masks = extract_segmentation_masks(image_segmentation)
        i += 1

  #  global first
  #  global first_selected_image

  #  if(first):
  #      first_selected_image = cv2.imread(selected_image)
  #      first = False

 #   colored_image = color_transfer(first_selected_image, cv2.imread(selected_image))
 #   cv2.imwrite(os.path.join(save_dir, anp+'.jpg'), colored_image)
 #   selected_image = os.path.join(save_dir, anp+'.jpg')

    required_mask = mask_for_tf_with_color(image_masks, color)
    
    results+=anp+' image: '+selected_image+'\n'
    style_anp_image = load_image(selected_image)
    #cv2.imwrite(os.path.join(save_dir, anp+'_seg_raw.png'), image_segmentation)

    style_anp_image = vgg.preprocess(style_anp_image)

    anp_gram_matrices = []
    
    global weight_restorer
    

    with tf.compat.v1.Session() as sess:

        sess.run(tf.compat.v1.global_variables_initializer())
        weight_restorer.init(sess)
        style_convs = sess.run(
                fetches=[vgg19.conv1_1, vgg19.conv2_1, vgg19.conv3_1, vgg19.conv4_1, vgg19.conv5_1],
                feed_dict={image_placeholder: style_anp_image})

    for style_conv in style_convs:
        anp_gram_matrices.append(tf.compat.v1.identity(calculate_gram_matrix_with_mask(style_conv, required_mask)))
    
    return anp_gram_matrices, results


def filter_content_masks(content_masks, style_colors):
    new_content_masks = {}
    for color in content_masks.keys():
        if color in style_colors:
            new_content_masks[color] = content_masks[color]
    return new_content_masks
