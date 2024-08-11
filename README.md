## Overview

This repository contains a Python application for training and predicting image classifications. It leverages a pre-trained model (likely VGG16) and fine-tunes it on a given dataset. Once trained, the model can be used to predict the class of an image.

### Prerequisites
* Python 3.6+
* PyTorch
* torchvision
* numpy
* PIL
* argparse
* json
  
### Usage
#### Training
To train a model:

Bash
python train.py --data_dir <path_to_your_data> --save_dir <path_to_save_model>

--data_dir: Path to the directory containing your image dataset. The dataset should be organized into subdirectories, each representing a class.  
--save_dir: Path to the directory where the trained model will be saved.
#### Predicting
To predict the class of an image:

Bash
python predict.py --image_path <path_to_image> --checkpoint <path_to_checkpoint> --top_k <number_of_predictions> --category_names <path_to_category_names> --gpu <use_gpu_if_available>


--image_path: Path to the image to be classified.  
--checkpoint: Path to the trained model checkpoint.  
--top_k: Number of top predictions to return (default: 5).  
--category_names: Path to a JSON file mapping class indices to class names. Replace cat_to_name.json with an appropriate file for your dataset.  
#### Additional Notes
The provided cat_to_name.json file is for the dataset : 'https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz'.  
You should replace it with a JSON file that maps class indices to class names for your specific dataset.  
The script assumes a specific data organization and model architecture. You may need to modify the code to accommodate different datasets or models.  
Consider adding hyperparameter tuning and data augmentation for improved performance.  
