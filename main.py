from train_and_evaluate import FishSegmentationClassification
from Data_Preprocessing import prepare_dataset
from config import BATCH_SIZE, INPUT_SHAPE, NUM_CLASSES, PATH, NUM_FILTERS, LEARNING_RATE, MODEL_NAME,\
    NUM_EPOCHS, AUG_CONFIG, MODEL_PATH, SHOW_IMAGE_DATA, PATH_TO_DATA, VALID_SIZE, TEST_SIZE,\
    SHOW_LEARNING_CURVES, CLASS_NAMES, NUM_OF_EXAMPLES, INPUT_NAME, OUTPUT_NAME, WEIGHTS_PATH


def main(prepare_data: bool, train: bool, evaluate: bool, save: bool):
    if prepare_data:
        prepare_dataset(path_to_data=PATH, valid_size=VALID_SIZE, test_size=TEST_SIZE)
    #  Creating the fish segmentation and classification neural network
    classifier = FishSegmentationClassification(
        batch_size=BATCH_SIZE,
        target_size=INPUT_SHAPE,
        num_classes=NUM_CLASSES,
        num_filters=NUM_FILTERS,
        learning_rate=LEARNING_RATE,
        model_name=MODEL_NAME,
        class_names=CLASS_NAMES,
        input_name=INPUT_NAME,
        output_name=OUTPUT_NAME,
        path_to_model_weights=WEIGHTS_PATH
    )
    if train:
        # Training neural network
        classifier.train(
            path=PATH_TO_DATA,
            epochs=NUM_EPOCHS,
            augmentation=AUG_CONFIG,
            show_learning_curves=SHOW_LEARNING_CURVES,
            show_image_data=SHOW_IMAGE_DATA,
            num_of_examples=NUM_OF_EXAMPLES
        )
    if evaluate:
        #  Testing neural network
        classifier.evaluate(
            path=PATH_TO_DATA,
            show_image_data=SHOW_IMAGE_DATA,
            num_examples=NUM_OF_EXAMPLES
        )
    if save:
        #  Saving trained neural network
        classifier.save_model(path_to_save=MODEL_PATH)


if __name__ == '__main__':
    main(
        prepare_data=False,
        train=False,
        evaluate=False,
        save=True
    )
