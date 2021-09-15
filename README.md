# Fish-Segmentation
![img13.png](images/img13.PNG)

This repository contains the implementation of the Segmentation and Classification road sign classifier. The German Traffic Sign Recognition Benchmark dataset (https://benchmark.ini.rub.de) was used for training and testing. It contains about 39000 training and 12500 test images belonging to 43 classes. The complexity of this dataset lies in the rather strong imbalance of the classes (see the figure below) so the class weights are applied during training.

## Testing
This section presents the results of the trained model obtained on the test set: 

*Loss: 0.022;*

*Segmentation loss: 0.022;*

*Segmentation IoU: 91.2%;*

*Classification loss: 0.000;*

*Classification F1 score: 100%;*

Examples of image classification by model:

![img1.png](images/img1.PNG)

![img3.png](images/img3.PNG)

## Using script

If you want to train/evaluate or save the model by yourself, then use the code presented in **main.py**. The main parameters used in the code are specified in the file **config.py**. For image classification by pretrained model use **test.py**. The trained model is stored in the folder **custom_DeeplabV3Plus (trained_model)**. Below are examples of image classification by the trained model:

![img4.png](images/img4.PNG)

![img5.png](images/img5.PNG)

![img6.png](images/img6.PNG)

![img7.png](images/img7.PNG)

![img8.png](images/img8.PNG)

![img9.png](images/img9.PNG)

![img10.png](images/img10.PNG)

![img11.png](images/img11.PNG)

![img12.png](images/img12.PNG)
