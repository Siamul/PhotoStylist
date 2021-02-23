# PhotoStylist
This repository contains the code for the paper "PhotoStylist: Altering the Style of Photos based on the Connotations of Texts". 
The implementation is done using Tensorflow. 

You have to download the weights in the link: https://drive.google.com/file/d/1ZdS6fe11aaVzhbrc5fyLOvuyAeOLG0pd/view?usp=sharing and extract it to "weights/" folder.

You have to download the MVSO dataset and update the location in the code. We have provided a toy dataset that is able to run the code for the content.jpg and style.txt provided.

The instructions below show you how to run the code.

For running the test example:

>> python3 photostylist.py

For running using your own content image and style text (remember to download and use MVSO dataset):

>> python3 photostylist.py --content_image <path_to_content_image_file> --style_text <path_to_style_text_file>

To view the data from the evaluation survey, use the "Evaluation_Graphs.ipynb" IPython Notebook. It contains codes to generate the graphs presented in the paper as well.
