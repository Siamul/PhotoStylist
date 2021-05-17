# PhotoStylist
This repository contains the code for the paper "PhotoStylist: Altering the Style of Photos based on the Connotations of Texts" published in PAKDD 2021. The paper can be found here: https://doi.org/10.1007/978-3-030-75762-5_51

The implementation is done using Tensorflow. 

You have to download the weights in the link: https://drive.google.com/file/d/1ZdS6fe11aaVzhbrc5fyLOvuyAeOLG0pd/view?usp=sharing and extract it to "weights/" folder.

You have to download the MVSO dataset and update the location in the code. Or, you could make a toy dataset with images according to different ANPs and supply it to the model.

The instructions below show you how to run the code.

For running using your own content image and style text (remember to download and use MVSO dataset):

>> python3 photostylist.py --content_image <path_to_content_image_file> --style_text <path_to_style_text_file>

To view the data from the evaluation survey, use the "Evaluation_Graphs.ipynb" IPython Notebook. It contains codes to generate the graphs presented in the paper as well.
