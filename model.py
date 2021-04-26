import tensorflow as tf


def make_model(input_shape, n):
    model = tf.keras.models.Sequential()

    model.add(
        tf.keras.layers.Conv2D(
            32, (5, 5), strides=(3, 3), activation="relu", input_shape=input_shape
        )
    )
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model.add(tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(1000, activation="relu"))
    model.add(tf.keras.layers.Dense(500, activation="relu"))
    model.add(tf.keras.layers.Dense(n))

    return model


def make_baseline_model(input_shape, n):
    model = tf.keras.models.Sequential()

    resnet50 = tf.keras.applications.ResNet50(
        include_top=False, weights="imagenet", pooling="max", classes=1000
    )
    resnet50.trainable = False
    model.add(resnet50)

    model.add(tf.keras.layers.Dense(1000, activation="relu"))
    model.add(tf.keras.layers.Dense(500, activation="relu"))
    model.add(tf.keras.layers.Dense(n))

    return model