# Fish-Segmentation
![img13.png](images/img13.PNG)

This repository contains the implementation of the NN model for fish segmentation and classification. The Large Scale Fish Dataset (https://ieeexplore.ieee.org/abstract/document/9259867) was used for training and testing. It contains about 9000 images belonging to 9 classes. Feature of the current approach is the CNN architecture with two independent outputs for segmentation and classification. Such an architecture was implemented since each image in the dataset contained only an object of one class. 

Large Scale Fish Dataset has this configuration:
```
E:.
├───Black Sea Sprat
│   ├───Black Sea Sprat
│   └───Black Sea Sprat GT
├───Gilt-Head Bream
│   ├───Gilt-Head Bream
│   └───Gilt-Head Bream GT
├───Hourse Mackerel
│   ├───Hourse Mackerel
│   └───Hourse Mackerel GT
├───Red Mullet
│   ├───Red Mullet
│   └───Red Mullet GT
├───Red Sea Bream
│   ├───Red Sea Bream
│   └───Red Sea Bream GT
├───Sea Bass
│   ├───Sea Bass
│   └───Sea Bass GT
├───Shrimp
│   ├───Shrimp
│   └───Shrimp GT
├───Striped Red Mullet
│   ├───Striped Red Mullet
│   └───Striped Red Mullet GT
└───Trout
    ├───Trout
    └───Trout GT
```

Before use, the dataset is split into training, validation and test image sets. To do this, first, a list of all system paths for images and their corresponding masks is read. Then this list is divided into training, validation and test lists and saved to a file. Albumintations library (https://albumentations.ai/docs/api_reference/augmentations/) was used for image augmentation. A custom metric IoU implemented in the library segmentation models (https://segmentation-models.readthedocs.io/en/latest/index.html) was also used. As can be seen from the learning curves presented below, the classification and segmentation quality metrics for the validation set were mostly superior to those for the training set, which indicates that the model is not overfitting.

![img2.png](images/img2.PNG)

## Testing
This section presents the results of the trained model obtained on the test set: 

*Loss: 0.022;*

*Segmentation loss: 0.022;*

*Segmentation IoU: 91.2%;*

*Classification loss: 0.000;*

*Classification F1 score: 100%;*

Examples of image segmentation and classification by model:

![img1.png](images/img1.PNG)

![img3.png](images/img3.PNG)

## Using script

If you want to train/evaluate or save the model by yourself, then use the code presented in **main.py**. The main parameters used in the code are specified in the file **config.py**. For image classification and segmentation by pretrained model use **test.py**. The trained model is stored in the folder **custom_DeeplabV3Plus (trained_model)**. Below are examples of image classification and segmentation by the trained model:

![img4.png](images/img4.PNG)

![img5.png](images/img5.PNG)

![img6.png](images/img6.PNG)

![img7.png](images/img7.PNG)

![img8.png](images/img8.PNG)

![img9.png](images/img9.PNG)

![img10.png](images/img10.PNG)

![img11.png](images/img11.PNG)

![img12.png](images/img12.PNG)
