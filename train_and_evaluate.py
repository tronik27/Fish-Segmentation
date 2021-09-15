import tensorflow as tf
from tensorflow_addons.metrics import F1Score
import matplotlib.pyplot as plt
from Data_Preprocessing import CustomDataGen
from cnn_model import DenseDeepLabV3Plus
import numpy as np
from multiprocessing import cpu_count
import random
from typing import Tuple
import json
import cv2 as cv
from segmentation_models.metrics import IOUScore


class FishSegmentationClassification:
    def __init__(self,
                 batch_size: int,
                 target_size: Tuple[int, int, int],
                 num_classes: int,
                 num_filters: int,
                 learning_rate: float,
                 model_name: str,
                 class_names: list,
                 input_name: str,
                 output_name: str,
                 path_to_model_weights: str) -> None:
        """
        Fish segmenter and classifier class.
        :param batch_size: size of the batch of images fed to the input of the neural network.
        :param target_size: the size to which all images in the dataset are reduced.
        :param num_classes: number of classes of images in dataset.
        :param num_filters: network expansion factor, determines the number of filters in start layer.
        :param learning_rate: learning rate when training the model.
        :param model_name: name of model.
        :param class_names: class names.
        :param input_name: name of the input tensor.
        :param output_name: name of the output tensor.
        :param path_to_model_weights: folder where the weights of the model will be saved after the epoch at which it
         showed the best result.
        """
        self.batch_size = batch_size
        self.target_size = target_size
        self.num_classes = num_classes
        self.class_names = class_names
        self.path_to_model_weights = path_to_model_weights
        self.nn = DenseDeepLabV3Plus(input_shape=self.target_size, num_classes=num_classes, num_filters=num_filters,
                                     input_name=input_name, output_name=output_name).build()
        self.nn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                        loss={'masks': tf.keras.losses.BinaryCrossentropy(),
                              'labels': tf.keras.losses.CategoricalCrossentropy()},
                        metrics={'masks': IOUScore(name='IoU', per_image=True),
                                 'labels': F1Score(num_classes=self.num_classes, threshold=0.5, average='micro')})
        self.model_name = model_name
        self.model_summary = self.nn.summary()

    def train(self,
              path: str,
              augmentation: list = [],
              epochs: int = 100,
              show_learning_curves: bool = False,
              show_image_data: bool = False,
              num_of_examples: int = 3
              ) -> None:
        """
        Method for training the model.
        :param path: path to folder containing data.
        :param augmentation: list of transforms to be applied to the training image.
        :param epochs: number of epochs to train the model.
        :param show_learning_curves: indicates whether to show show learning curves or not.
        :param show_image_data: indicates whether to show original and augmented image with labels or not.
        :param num_of_examples: number of original and augmented image examples to display.
        """
        with open(path, "r") as file:
            data = json.load(file)
        train_datagen = CustomDataGen(
            labels=data['train_labels'],
            image_path=data['train_img_path'],
            mask_path=data['train_mask_path'],
            batch_size=self.batch_size,
            target_size=self.target_size,
            aug_config=augmentation
        )
        validation_datagen = CustomDataGen(
            labels=data['validation_labels'],
            image_path=data['validation_img_path'],
            mask_path=data['validation_mask_path'],
            batch_size=self.batch_size,
            target_size=self.target_size
        )

        if show_image_data:
            print('[INFO] displaying images from dataset. Close the window to continue...')
            train_datagen.show_image_data(class_names=self.class_names, num_of_examples=num_of_examples)

        print('[INFO] training network...')
        history = self.nn.fit(
            train_datagen,
            validation_data=validation_datagen,
            steps_per_epoch=train_datagen.number_of_images // self.batch_size,
            callbacks=self.__get_callbacks(),
            epochs=epochs,
            workers=cpu_count(),
        )

        if show_learning_curves:
            print('[INFO] displaying information about learning process. Close the window to continue...')
            self.__plot_learning_curves(history)

    def evaluate(self, path: str, show_image_data: bool, num_examples: int) -> None:
        """
        Method for evaluating a model on a test set.
        :param path: path to test data file.
        :param show_image_data: indicates whether to show predictions for set images or not.
        :param num_examples: number of predictions examples to display
        """
        with open(path, "r") as file:
            data = json.load(file)

        test_datagen = CustomDataGen(
            labels=data['test_labels'],
            image_path=data['test_img_path'],
            mask_path=data['test_mask_path'],
            batch_size=self.batch_size,
            target_size=self.target_size
        )

        try:
            self.nn.load_weights(self.path_to_model_weights)
        except FileNotFoundError:
            raise ValueError('There are no weights to evaluate the trained model! Try to train the model first.')

        print('[INFO] evaluating network...')
        results = self.nn.evaluate(test_datagen, batch_size=self.batch_size, verbose=0, use_multiprocessing=True)
        for i, metric in enumerate(self.nn.metrics_names):
            print('{}: {:.03f}'.format(metric, results[i]))
        if show_image_data:
            self.__show_image_data(test_datagen, num_examples=num_examples)

    def save_model(self, path_to_save: str) -> None:
        """
        Method for saving the whole model.
        :param path_to_save: folder where the model will be stored.
        """
        print('[INFO] saving network model...')
        try:
            self.nn.load_weights(self.path_to_model_weights)
        except FileNotFoundError:
            raise ValueError('There are no weights to save the trained model! Try to train the model first.')

        self.nn.save(path_to_save, save_format='h5')

    def __show_image_data(self, generator: tf.keras.utils.Sequence, num_examples: int = 10) -> None:
        """
        Method for showing predictions.
        :param generator: data generator.
        :param num_examples: number of predictions examples to display
        """
        for _ in range(num_examples):
            j = random.randint(0, len(generator))
            images, [_, labels] = generator[j]
            if images.shape[0] > 3:
                images = images[:3, :, :, :]
            predict_masks, predict_labels = self.nn.predict(images)
            predictions = np.max(predict_labels, axis=-1)
            predict_labels = np.argmax(predict_labels, axis=-1)
            labels = np.argmax(labels, axis=-1)
            fig, axes = plt.subplots(nrows=1, ncols=images.shape[0], figsize=(12, 6))
            fig.suptitle('Network prediction results:', fontsize=14, fontweight="bold")
            for i in range(images.shape[0]):
                text = 'True label:\n {},\n Predicted label:\n {},\n Confidence of prediction:\n {:.02f}%.'.format(
                    self.class_names[labels[i]],
                    self.class_names[predict_labels[i]],
                    predictions[i] * 100
                )
                mask = ((predict_masks[i, :, :, 0] - predict_masks[i, :, :, 0].min())
                        * (1 / (predict_masks[i, :, :, 0].max() - predict_masks[i, :, :, 0].min()) * 255)).astype('uint8')
                mask[mask < 0.7 * mask.max()] = 0
                mask[mask >= 0.7 * mask.max()] = 255
                predict_contour, _ = cv.findContours(image=mask, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)
                image = ((images[i, :, :, :] - images[i, :, :, :].min())
                         * (1 / (images[i, :, :, :].max() - images[i, :, :, :].min()) * 255)).astype('uint8')
                cv.drawContours(image=image, contours=predict_contour, contourIdx=-1, color=(0, 255, 0), thickness=1,
                                lineType=cv.LINE_AA)
                axes[i].imshow(image)
                axes[i].set_title(text, size=12)
                axes[i].axis('off')
            plt.show()

    def __get_callbacks(self) -> list:
        """
        Method for creating a list of callbacks.
        :return: list containing callbacks.
        """

        def scheduler(epoch, lr):
            if epoch < 1:
                return lr
            else:
                return lr * tf.math.exp(-0.1)

        callbacks = list()
        callbacks.append(tf.keras.callbacks.LearningRateScheduler(scheduler))
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', min_delta=0.005,
                                                         factor=0.5, patience=3, min_lr=0.00001)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(self.path_to_model_weights, save_weights_only=True,
                                                        save_best_only=True, monitor='val_loss', mode='min')
        stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, min_delta=0.0001)
        callbacks += [reduce_lr, checkpoint, stop]
        return callbacks

    def __plot_learning_curves(self, metric_data) -> None:
        """
        Method for plotting learning curves.
        :param metric_data: dictionary containing metric an loss logs.
        """
        print(metric_data)
        figure, axes = plt.subplots(len(metric_data.history) // 2, 1, figsize=(5, 10))
        for axe, metric in zip(axes, self.nn.metrics_names):
            name = metric.replace("_", " ").capitalize()
            axe.plot(metric_data.epoch, metric_data.history[metric], label='Train')
            axe.plot(metric_data.epoch, metric_data.history['val_' + metric], linestyle="--",
                     label='Validation')
            axe.set_xlabel('Epoch')
            axe.set_ylabel(name)
            axe.grid(color='coral', linestyle='--', linewidth=0.5)
            axe.legend()
        plt.show()
