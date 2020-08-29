# Covid-19-Mask-Detection
## Demo ##

![gif](demo/demo.gif)

## About ##
This is a small project I worked on for a day to detect if a person is wearing a mask. It uses OpenCV's DNN module to detect the faces which are then passed into a convolutional neural network trained on 440 images to predicting the output.
## Dataset ##
The dataset was downloaded from Kaggle and converted into a NumPy array with labels [1, 0] for faces with mask and [0, 1] for faces without a mask. Additionally, the training images are resized into 100 by 100 grayscale. 
The dataset is then uploaded on google drive and trained on google colab with additional image augmentation for better generalization.

Dataset: https://www.kaggle.com/dhruvmak/face-mask-detection

## Training ##
1. Preprocess the training data and save into numpy array with png_to_npy.py
2. Upload the training data on Google Drive
3. Create a Google Colab project, ensure GPU is enabled
4. Upload the .ipynb file under training_colab to Google Colab and train the model 

## Usage ## 
1. Download or clone this repo
2. Run detect.py

## References ## 
More on face detection: https://towardsdatascience.com/extracting-faces-using-opencv-face-detection-neural-network-475c5cd0c260
