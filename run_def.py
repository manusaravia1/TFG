import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import glob
import tensorflow as tf
from PIL import Image
import typer

from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from keras import Model
from tensorflow.keras.applications import InceptionV3, MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import *

# Nombres de las siete clases
class_names = [
    "other",
    "chassis_engine_number",
    "speedometer_odometer",
    "document",
    "car_internal",
    "car_outer",
    "car_dashboard",
]

# directorios para train y test
dir_train = "/home/manusaravia/Insurmapp/images_big/"
dir_test = "/home/manusaravia/Insurmapp/images_big_test/"

# Creacion de las subcarpetas necesarias
def create_train_directories(df):

    for clase in class_names:
        os.mkdir(dir_train + clase)

    img_list = [image[-26:] for image in glob.glob(dir_train + "*")]

    for index, row in df.iterrows():
        if row["image"] in img_list:
            print("Entro")
            img = Image.open(dir_train + row["image"])
            img = img.save(dir_train + row["label"] + "/" + row["image"])


def create_test_directories(df):
    for clase in class_names:
        os.mkdir(dir_test + clase)

    img_list = [image[-26:] for image in glob.glob(dir_test + "*")]

    for index, row in df.iterrows():
        if row["image"] in img_list:
            print("Entro")
            img = Image.open(dir_test + row["image"])
            img = img.save(dir_test + row["label"] + "/" + row["image"])


def create_inception_model(img_height, img_width):
    input_tensor = tf.keras.layers.Input(shape=(img_height, img_width, 3))

    inception_base = InceptionV3(
        include_top=False, weights="imagenet", input_tensor=input_tensor
    )

    for layer in inception_base.layers:
        layer.trainable = True
        if isinstance(layer, BatchNormalization):
            layer.momentum = 0.9

        # Make all layers upto -25 non-trainable except BatchNorm layers
        for layer in inception_base.layers[:-25]:
            if not isinstance(layer, BatchNormalization):
                layer.trainable = False

    x = inception_base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.1)(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.2)(x)
    output_tensor = Dense(len(class_names), activation="softmax")(x)

    model = Model(inputs=input_tensor, outputs=output_tensor)

    return model


def create_mobilenet_model(img_height, img_width):
    input_tensor = tf.keras.layers.Input(shape=(img_height, img_width, 3))

    mobilenet_base = MobileNetV2(input_tensor=input_tensor, include_top=False)

    for layer in mobilenet_base.layers:
        layer.trainable = True
        if isinstance(layer, BatchNormalization):
            layer.momentum = 0.9

        # Make all layers upto -25 non-trainable except BatchNorm layers
        for layer in mobilenet_base.layers[:-25]:
            if not isinstance(layer, BatchNormalization):
                layer.trainable = False

    x = mobilenet_base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.1)(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.2)(x)
    output_tensor = Dense(len(class_names), activation="softmax")(x)

    model = Model(inputs=input_tensor, outputs=output_tensor)

    return model


def create_generator(preprocess, train_dir, val_dir, img_height, img_width, batch_size):
    # Data generators with Image augmentations
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )
    validation_datagen = ImageDataGenerator(preprocessing_function=preprocess)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode="categorical",
    )

    validation_generator = validation_datagen.flow_from_directory(
        val_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode="categorical",
    )

    return train_generator, validation_generator


def scheduler(epoch, lr):
    return lr if epoch < 10 else lr * 0.9


def fit_model(model, path_model, train_generator, validation_generator):
    # definimos varios callbacks: ModelCheckpoint, EarlyStopping y LearningRateScheduler
    my_callbacks = [
        ModelCheckpoint(
            filepath="/home/manusaravia/Insurmapp/" + path_model + "_ckpt.h5",
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
        ),
        EarlyStopping(monitor="val_accuracy", patience=5),
        LearningRateScheduler(scheduler),
    ]

    # Entrenamiento
    history_inception = model.fit(
        train_generator,
        epochs=10,
        validation_data=validation_generator,
        callbacks=my_callbacks,
    )

    return history_inception


def show_training_graph(data):
    acc = data.history["accuracy"]
    val_acc = data.history["val_accuracy"]
    loss = data.history["loss"]
    val_loss = data.history["val_loss"]

    epochs = range(len(acc))
    plt.figure(figsize=(11, 8))
    plt.subplot(1, 2, 1)
    plt.grid(True)
    plt.plot(epochs, acc, label="Training Accuracy")
    plt.plot(epochs, val_acc, label="Validation Accuracy")
    plt.title("Evolución de la precisión en entrenamiento y validación")
    plt.legend(loc="lower right")

    plt.subplot(1, 2, 2)
    plt.grid(True)
    plt.plot(epochs, loss, label="Training Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.title("Evolución de loss en entrenamiento y validación")
    plt.legend(loc="upper right")

    plt.show()
    plt.savefig("/home/manusaravia/Insurmapp/model_graph.png")


# Main del problema que llama a varias funciones
def main(
    input_folder: str = "/home/manusaravia/Insurmapp",
    batch_size: int = 32,
    isCPU: bool = typer.Option(False, " /--isCPU", " /-d"),
    isInception: bool = typer.Option(True, " /--isInception", " /-d"),
    isCreatedDirectories: bool = typer.Option(True, " /--isCreatedDirectories", " /-d"),
):

    df = pd.read_pickle(input_folder + "/annotations.p")

    if not isCreatedDirectories:
        create_train_directories(df)
        create_test_directories(df)

    if isCPU:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    config = tf.compat.v1.ConfigProto(
        gpu_options=tf.compat.v1.GPUOptions(allow_growth=True)
    )
    gpus = tf.config.experimental.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(gpus[0], True)

    train_dir, val_dir = input_folder + "/images_big", input_folder + "/images_big_test"

    if isInception:
        print("<<<< Modelo InceptionV3 >>>>")
        img_height, img_width = 299, 299
        model = create_inception_model(img_height, img_width)
        model.compile(
            optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"]
        )

        preprocess = tf.keras.applications.inception_v3.preprocess_input

        train_generator, validation_generator = create_generator(
            preprocess, train_dir, val_dir, img_height, img_width, batch_size
        )

        path_model = "model_inceptionv3"

        history_inception = fit_model(
            model, path_model, train_generator, validation_generator
        )

        show_training_graph(history_inception)
        # model.save('/home/manusaravia/Insurmapp/history_inception.h5')

    else:
        print("<<<< Modelo MobileNet >>>>")
        img_height, img_width = 224, 224
        model = create_mobilenet_model(img_height, img_width)
        model.compile(
            optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"]
        )

        preprocess = tf.keras.applications.mobilenet_v2.preprocess_input

        train_generator, validation_generator = create_generator(
            preprocess, train_dir, val_dir, img_height, img_width, batch_size
        )

        path_model = "model_mobilenet"

        history_mobilenet = fit_model(
            model, path_model, train_generator, validation_generator
        )

        show_training_graph(history_mobilenet)


if __name__ == "__main__":
    typer.run(main)
