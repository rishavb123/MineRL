import tensorflow as tf

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(32, (5, 5), strides=(3, 3), activation='relu', input_shape=(800, 600, 3)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
assert model.output_shape == (None, 133, 99, 32)

model.add(tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
assert model.output_shape == (None, 32, 24, 64)

model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
assert model.output_shape == (None, 15, 11, 128)

# tf.keras.applications.ResNet50(
#     include_top=False, weights='imagenet', input_tensor=None,
#     input_shape=None, pooling=None, classes=1000)

model.add(tf.keras.layers.Flatten())
assert model.output_shape == (None, 15*11*128)

model.add(tf.keras.layers.Dense(1000))
model.add(tf.keras.layers.Dense(6))
assert model.output_shape == (None, 6)