import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from typing import Tuple
import cv2
from tensorflow_addons.metrics import F1Score
import json
from config import MODEL_PATH, CLASS_NAMES, WORK_DATA_PATH, INPUT_SHAPE
from segmentation_models.metrics import IOUScore
from json.decoder import JSONDecodeError


class DataGen(tf.keras.utils.Sequence):

    def __init__(self, data_path: str, batch_size: int, target_size: Tuple[int, int, int]) -> None:
        """
        Data generator for the task of fish segmentation and classification.
        :param data_path: path to folder containing image data.
        :param batch_size: size of the batch of images fed to the input of the neural network.
        :param target_size: the size to which all images in the dataset are reduced.
        """
        self.data_path = data_path
        self.batch_size = batch_size
        self.target_size = target_size
        self.file_pathes = self.__get_images_path()
        self.image_size = None

    def __getitem__(self, index: int) -> Tuple[np.ndarray, list]:
        """
        Getting batch.
        :param index: batch number.
        :return: image tensor and list of image file paths.
        """
        data_batch = self.file_pathes[index * self.batch_size:(index + 1) * self.batch_size]
        images = self.__get_data(data_batch)
        return images, data_batch

    def __len__(self):
        return len(self.file_pathes) // self.batch_size

    def __get_images_path(self) -> list:
        """
        Getting pathes to images.
        :return: list containing the pathes to the files in the directory and subdirectories given by self.path.
        """
        file_pathes = []
        for path, subdirs, files in os.walk(self.data_path):
            for name in files:
                file_pathes.append(os.path.join(path, name))
        return file_pathes

    def __get_data(self, images_path: list) -> np.ndarray:
        """
        Making batch.
        :param images_path: list of pathes for images included in the batch.
        :return: image tensor.
        """
        images_batch = np.asarray([self.__get_image(path) for path in images_path])
        return images_batch

    def __get_image(self, path: str) -> np.ndarray:
        """
        Reads an image from a folder .
        :param path: path to the folder with images.
        :return: normalized image array.
        """
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        self.image_size = image.shape[:-1]
        image = cv2.resize(image, self.target_size[:-1], interpolation=cv2.INTER_AREA)
        return image / 255


class FSCWork:

    def __init__(self, path_to_model: str, path_to_data: str, target_size: Tuple[int, int, int],) -> None:
        """
        Fish segmentation and classification class.
        :param path_to_model: folder where the model is stored.
        :param path_to_data: path to folder containing data.
        :param target_size: the size to which all images in the dataset are reduced.
        """
        try:
            self.nn = tf.keras.models.load_model(path_to_model,
                                                 custom_objects={"F1Score": F1Score, 'IoU': IOUScore})
        except FileNotFoundError:
            raise ValueError('There is no trained model! Try to train the model first.')
        self.path_to_data = path_to_data
        self.target_size = target_size

    def predict(self, batch_size: int, path_to_save: str) -> None:
        """
        Method for calculating predictions for a large number of images. it will also output the image classification
         speed as frames per second.
        :param batch_size: size of the batch of images fed to the input of the neural network.
        :param path_to_save: path to folder where file with predictions will be saved.
        """
        computation_time = 0.
        predicted_labels = []
        predicted_masks = []
        data_gen = DataGen(data_path=self.path_to_data, batch_size=batch_size, target_size=self.target_size)
        print('[INFO] predicting labels...')
        for i in range(len(data_gen)):
            img_batch, img_names = data_gen[i]
            start_time = time.time()
            predict_masks, predict_labels = self.nn.predict(img_batch, use_multiprocessing=True)
            finish_time = time.time()
            computation_time += finish_time - start_time

            predicted_labels += (np.argmax(predict_labels, axis=-1)).tolist()
            predicted_masks += predict_masks.tolist()

            with open(os.path.join(path_to_save, "predictions.json"), 'w+') as file:
                try:
                    data = json.load(file)
                except JSONDecodeError:
                    data = dict()
                for image_name, mask, label in zip(img_names, predict_masks, predict_labels.tolist()):
                    data[image_name] = [mask.tolist(), label]
                json.dump(data, file, indent=4)

        print('[INFO] Mean FPS: {:.04f}.'.format(len(predicted_labels) / computation_time))

    def predict_and_show(self, class_names: list = []) -> None:
        """
        Method for showing predictions for single image.
        :param class_names: class names.
        """
        data_gen = DataGen(data_path=self.path_to_data, batch_size=1, target_size=self.target_size)

        for i in range(len(data_gen)):
            img, _ = data_gen[i]
            predict_mask, predict_label = self.nn(img, training=False)
            prediction = np.max(predict_label)
            if class_names:
                label = class_names[int(np.argmax(predict_label))]
            else:
                label = int((np.argmax(predict_label)))
            mask = predict_mask[0, :, :, 0].numpy()
            mask = ((mask - np.min(mask))
                    * (1 / (np.max(mask) - np.min(mask)) * 255)).astype('uint8')
            img = ((img[0, :, :, :] - img[0, :, :, :].min())
                   * (1 / (img[0, :, :, :].max() - img[0, :, :, :].min()) * 255)).astype('uint8')
            mask[mask < 0.7 * mask.max()] = 0
            mask[mask >= 0.7 * mask.max()] = 255
            predict_contour, _ = cv2.findContours(image=mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(image=img, contours=predict_contour, contourIdx=-1, color=(0, 255, 0), thickness=1,
                             lineType=cv2.LINE_AA)
            img = cv2.resize(img, data_gen.image_size[::-1], interpolation=cv2.INTER_AREA)
            plt.axis('off')
            plt.imshow(img)
            plt.title('Predicted Label: {}, \n confidence of prediction: {:.02f}%.'.format(label, prediction * 100))
            plt.waitforbuttonpress(0)
            plt.close('all')


if __name__ == '__main__':
    #  Creating the road signs classifier
    classifier = FSCWork(
        path_to_model=MODEL_PATH,
        path_to_data=r'E:\DATASETS\Fish\Fish_Dataset\Red Mullet\Red Mullet',
        target_size=INPUT_SHAPE
    )
    #  Getting predictions for images (use this method if you want to quickly classify a large number of images)
    classifier.predict(batch_size=16, path_to_save=WORK_DATA_PATH)
    #  Getting predictions for images (use this method if you want to see examples of image classification)
    classifier.predict_and_show(class_names=CLASS_NAMES)
