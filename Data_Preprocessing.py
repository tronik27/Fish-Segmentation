import tensorflow as tf
import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt
from Augmentation import image_augmentation
from random import shuffle
import cv2
import os
import glob
from sklearn.model_selection import train_test_split
import json
import random


class CustomDataGen(tf.keras.utils.Sequence):

    def __init__(self, labels: list, image_path: list, mask_path: list, batch_size: int,
                 target_size: Tuple[int, int, int], aug_config: list = []) -> None:
        """
        Data generator for the image classification and segmentation task.
        :param labels: array containing labels for images included in the batch.
        :param image_path: list containing paths for images included in the batch.
        :param mask_path: list containing paths for masks for images included in the batch.
        :param batch_size: size of the batch of images fed to the input of the neural network.
        :param target_size: the size to which all images in the dataset are reduced.
        :param aug_config: a dictionary containing the parameter values for augmentation.
        """
        self.labels = labels
        self.image_path = image_path
        self.mask_path = mask_path
        self.__shuffle_data()
        self.batch_size = batch_size
        self.target_size = target_size
        self.aug_config = aug_config
        self.number_of_images = len(self.image_path)
        self.num_classes = len(set(self.labels))

    def on_epoch_end(self):
        """
        Random shuffling of training data at the end of each epoch during training.
        """
        if self.augmentation:
            self.__shuffle_data()

    def __getitem__(self, index: int) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Getting batch.
        :param index: batch number.
        :return: image,masks and labels tensors.
        """
        image_batch = self.image_path[index * self.batch_size:(index + 1) * self.batch_size]
        mask_batch = self.mask_path[index * self.batch_size:(index + 1) * self.batch_size]
        labels_batch = self.labels[index * self.batch_size:(index + 1) * self.batch_size]
        images, masks, labels = self.__get_data(image_batch, mask_batch, labels_batch)
        return images, [masks, labels]

    def __len__(self):
        return self.number_of_images // self.batch_size

    def __get_data(self, images_path: list, masks_path: list, labels: np.array, original: bool = False) ->\
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Making batch.
        :param images_path: list containing paths for images included in the batch.
        :param masks_path: list containing paths for masks for images included in the batch.
        :param labels: array containing labels for images included in the batch.
        :return: images, masks and labels tensors.
        """
        images_batch = list()
        masks_batch = list()
        for image_path, mask_path, label in zip(images_path, masks_path, labels):
            image, mask = self.__get_image_and_mask(image_path, mask_path, original)
            images_batch.append(image)
            masks_batch.append(mask)
        labels_batch = self.__to_categorical(labels)
        masks_batch = np.expand_dims(np.asarray(masks_batch), axis=-1)
        return np.asarray(images_batch), masks_batch, labels_batch

    def __get_image_and_mask(self, image_path: str, mask_path: str, original: bool = False)\
            -> Tuple[np.ndarray, np.ndarray]:
        """
        Reads image and mask from a folder .
        :param image_path: path to the image.
        :param mask_path: path to the mask corresponding to the image.
        :return: normalized image and mask arrays.
        """
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, self.target_size[:-1], interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, self.target_size[:-1], interpolation=cv2.INTER_AREA)
        if self.aug_config and not original:
            image, mask = self.augmentation(image, mask)
        return image / 255, mask / 255

    def __to_categorical(self, labels: np.ndarray) -> np.ndarray:
        """
        Converts a class labels to binary class matrix.
        :param labels: class labels of images included in the batch.
        :return: ground truth .
        """
        return tf.keras.utils.to_categorical(labels, num_classes=self.num_classes)

    def augmentation(self, image: np.array, mask: np.array) -> np.array:
        """
        Apply augmentation to the image and mask.
        :param image: image array.
        :param mask: mask array.
        :return: augmented image and mask.
        """
        augmentation = image_augmentation(config=self.aug_config, target_shape=self.target_size)
        transform = augmentation(image=image, mask=mask)
        return transform['image'], transform['mask']

    def __shuffle_data(self):
        """
        Random shuffling of data.
        """
        temp = list(zip(self.image_path, self.mask_path, self.labels))
        shuffle(temp)
        self.image_path, self.mask_path, self.labels = zip(*temp)

    def show_image_data(self, class_names: list, num_of_examples: int = 3) -> None:
        """
        Method for showing original and augmented image with labels.
        :param num_of_examples: number of images to display.
        :param class_names: class names.
        """
        for _ in range(num_of_examples):
            j = random.randint(0, self.number_of_images)
            augmented_image, masks, _ = self.__get_data(
                images_path=[self.image_path[j]],
                masks_path=[self.mask_path[j]],
                labels=[self.labels[j]]
            )
            image, _, _ = self.__get_data(
                images_path=[self.image_path[j]],
                masks_path=[self.mask_path[j]],
                labels=[self.labels[j]],
                original=True)
            mask = masks[0, :, :, 0]
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9, 4))
            fig.suptitle('Original, augmented image and mask', fontsize=16)
            axes[0].imshow(image[0, :, :, :])
            axes[0].set_title('Original,\n {}'.format(class_names[self.labels[j]]), size=12)
            axes[0].axis('off')
            axes[1].imshow(augmented_image[0, :, :, :])
            axes[1].set_title('Augmented, class: "{}"'.format(self.labels[j]), size=12)
            axes[1].axis('off')
            axes[2].imshow(mask, cmap="gray")
            axes[2].set_title('Mask', size=12)
            axes[2].axis('off')
            plt.show()


def prepare_dataset(path_to_data: str, valid_size: float, test_size: float):
    """
    Method of splitting a dataset into datasets for training, testing and validation.
    :param path_to_data: path to dataframe.
    :param test_size: part of the dataframe that will be allocated to the test set.
    :param valid_size: part of the dataframe that will be allocated to the validation set.
    :return: train, test and validation datasets.
    """
    train_img_path = []
    test_img_path = []
    validation_img_path = []
    train_mask_path = []
    test_mask_path = []
    validation_mask_path = []
    train_labels = []
    test_labels = []
    validation_labels = []
    for i, class_name in enumerate(os.listdir(path_to_data)):
        subdirs = os.listdir(os.path.join(path_to_data, class_name))
        for subdir in subdirs:
            if subdir.endswith('GT'):
                mask_path = glob.glob(os.path.join(path_to_data, class_name, subdir, '*.png'), recursive=True)
            else:
                image_path = glob.glob(os.path.join(path_to_data, class_name, subdir, '*.png'), recursive=True)

        image_train, image_test, mask_train, mask_test = train_test_split(image_path, mask_path,
                                                                          test_size=valid_size + test_size,
                                                                          shuffle=True,
                                                                          random_state=42)
        image_valid, image_test, mask_valid, mask_test = train_test_split(image_test, mask_test,
                                                                          test_size=test_size/(valid_size + test_size),
                                                                          random_state=42)
        train_img_path += image_train
        test_img_path += image_test
        validation_img_path += image_valid
        train_mask_path += mask_train
        test_mask_path += mask_test
        validation_mask_path += mask_valid
        train_labels += (i * np.ones(len(image_train), dtype=int)).tolist()
        test_labels += (i * np.ones(len(image_test), dtype=int)).tolist()
        validation_labels += (i * np.ones(len(image_valid), dtype=int)).tolist()
    data = {
        'train_img_path': train_img_path,
        'test_img_path': test_img_path,
        'validation_img_path': validation_img_path,
        'train_mask_path': train_mask_path,
        'test_mask_path': test_mask_path,
        'validation_mask_path': validation_mask_path,
        'train_labels': train_labels,
        'test_labels': test_labels,
        'validation_labels': validation_labels
    }
    with open("data.json", "w") as f:
        json.dump(data, f, indent=4)
