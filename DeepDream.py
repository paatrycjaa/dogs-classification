import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from pipeline import set_gpu_enabled
import os
import matplotlib.pyplot as plt
import re


class DeepDream:

    def __init__(self, model, layer_settings=None, in_base=True, step_size=2.0, iterations=20):
        self.step = step_size  # Gradient ascent step size
        self.iterations = iterations  # Number of ascent steps per scale
        self.max_loss = 100000

        # These are the names of the layers
        # for which we try to maximize activation,
        # as well as their weight in the final loss
        # we try to maximize.
        if layer_settings is None:
            self.layer_settings = {
                'expanded_conv_5/depthwise': 1.0,
            }
        else:
            self.layer_settings = layer_settings
        self.model = model

        if in_base:
            self.outputs_dict = dict(
                [
                    (layer.name, layer.output)
                    for layer in [model.layers[0].get_layer(name) for name in self.layer_settings.keys()]
                ]
            )
        else:
            self.outputs_dict = dict(
                [
                    (layer.name, layer.output)
                    for layer in [model.get_layer(name) for name in self.layer_settings.keys()]
                ]
            )

        print("##########################################")
        print("OUTPUT DICT")
        print("##########################################")
        print(self.outputs_dict)

        # Set up a model that returns the activation values for every target layer
        # (as a dict)
        if in_base:
            self.feature_extractor = keras.Model(inputs=self.model.layers[0].inputs, outputs=self.outputs_dict)
        else:
            self.feature_extractor = keras.Model(inputs=self.model.inputs, outputs=self.outputs_dict)

        print("##########################################")
        print("INPUT")
        print("##########################################")
        if in_base:
            print(self.model.layers[0].inputs)
        else:
            print(self.model.inputs)
        print("##########################################")
        print("feature extractor")
        print("##########################################")
        print(self.feature_extractor)

    def _preprocess_image(self, img):
        # Util function to format picture into appropriate array.
        img = keras.preprocessing.image.img_to_array(img)
        img = np.reshape(img, (1, 224, 224, 3))
        img = keras.applications.mobilenet_v3.preprocess_input(img)
        return img

    def _deprocess_image(self, x):
        x = x.reshape((x.shape[1], x.shape[2], 3))
        x = np.clip(x, 0, 255).astype("uint8")
        return x

    def _compute_loss(self, input_image):
        features = self.feature_extractor(input_image)
        # Initialize the loss
        loss = tf.zeros(shape=())
        for name in features.keys():
            coeff = self.layer_settings[name]
            activation = features[name]
            # We avoid border artifacts by only involving non-border pixels in the loss.
            scaling = tf.reduce_prod(tf.cast(tf.shape(activation), "float32"))
            loss += coeff * tf.reduce_sum(tf.square(activation[:, 2:-2, 2:-2, :])) / scaling
        return loss

    def _gradient_ascent_step(self, img, learning_rate):
        with tf.GradientTape() as tape:
            tape.watch(img)
            loss = self._compute_loss(img)
        # Compute gradients.
        grads = tape.gradient(loss, img)
        # Normalize gradients.
        grads /= tf.maximum(tf.reduce_mean(tf.abs(grads)), 1e-6)
        img += learning_rate * grads
        return loss, img

    def _gradient_ascent_loop(self, img, iterations, learning_rate, max_loss=None):
        for i in range(iterations):
            loss, img = self._gradient_ascent_step(img, learning_rate)
            if max_loss is not None and loss > max_loss:
                break
            print("... Loss value at step %d: %.2f" % (i, loss))
        return img

    def deep_dream_filtering(self, img):
        original_img = self._preprocess_image(img)
        img = tf.identity(original_img)  # Make a copy
        print("Processing...")
        img = self._gradient_ascent_loop(
            img, iterations=self.iterations, learning_rate=self.step, max_loss=self.max_loss
        )
        return self._deprocess_image(img.numpy())


def crop_and_resize(img, size=(224, 224)):
    edge = min(img.size[0], img.size[1])
    box = (
        (img.size[0] - edge) / 2,
        (img.size[1] - edge) / 2,
        (img.size[0] - edge) / 2 + edge,
        (img.size[1] - edge) / 2 + edge
    )
    img = img.crop(box)
    img = img.resize(size)
    return img


if __name__ == "__main__":
    set_gpu_enabled(False)
    model_dir = 'model_for_deepdream'
    input_file = 'doggo4.jpg'
    output_dir = 'deepdream'
    step_size = 1  # Gradient ascent step size
    iterations = 100  # Number of ascent steps
    layers = [
        'Conv',
        'expanded_conv/depthwise',
        'expanded_conv/project',
        'expanded_conv_3/project',
        'expanded_conv_5/depthwise',
        'expanded_conv_5/project',
        'expanded_conv_6/expand',
        'expanded_conv_6/depthwise',
        'expanded_conv_7/depthwise',
        'expanded_conv_8/expand',
        'expanded_conv_8/depthwise',
        'expanded_conv_9/project',
        'expanded_conv_10/depthwise',
        'expanded_conv_10/project',
        'Conv_1',
        'Conv_2'
    ]

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    model = keras.models.load_model(model_dir)
    # model.layers[0].summary()
    # ddd
    image = load_img(input_file)
    plt.imshow(image)
    plt.title(input_file)
    plt.show()
    image = crop_and_resize(image)
    plt.imshow(image)
    plt.title(input_file + "[cropped]")
    plt.show()

    for layer in layers:
        layer_settings = {
            layer: 1.0,
        }
        dream = DeepDream(model, layer_settings=layer_settings,
                          step_size=step_size, iterations=iterations)
        result_image = dream.deep_dream_filtering(image)
        result_name = input_file.split('/')[-1].split('.')[0] + "-s" + str(step_size) + "-i" + str(iterations) \
            + "-" + re.sub('[^a-zA-Z0-9_]', '', layer)
        keras.preprocessing.image.save_img(os.path.join(output_dir, result_name) + ".png", result_image)
        plt.imshow(result_image)
        plt.title(layer)
        plt.show()
