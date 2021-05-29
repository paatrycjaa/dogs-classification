import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import inception_v3
from tensorflow.keras.preprocessing.image import load_img
from IPython.display import Image, display
import os
from pipeline import set_gpu_enabled


class DeepDream():

    def __init__(self, model, layer_settings = None):
        self.base_image_path = "sky.jpeg"
        self.result_prefix = "sky_dream"

    # These are the names of the layers
    # for which we try to maximize activation,
    # as well as their weight in the final loss
    # we try to maximize.
    # You can tweak these setting to obtain new visual effects.
        if(layer_settings == None):
            self.layer_settings = {
                "mixed4": 1.0,
                "mixed5": 1.5,
                "mixed6": 2.0,
                "mixed7": 2.5,
            }
        else:
            self.layer_settings = layer_settings
        self.model = model

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
        self.feature_extractor = keras.Model(inputs=self.model.inputs, outputs=self.outputs_dict)
        print("##########################################")
        print("INPUT")
        print("##########################################")
        print(self.model.inputs)
        print("##########################################")
        print("feature extractor")
        print("##########################################")
        print(self.feature_extractor)
        

        # Playing with these hyperparameters will also allow you to achieve new effects
        self.step = 0.01  # Gradient ascent step size
        self.num_octave = 3  # Number of scales at which to run gradient ascent
        self.octave_scale = 1.4  # Size ratio between scales
        self.iterations = 20  # Number of ascent steps per scale
        self.max_loss = 15.0

    def preprocess_image(self, img):
        # Util function to open, resize and format pictures
        # into appropriate arrays.
        img = keras.preprocessing.image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = inception_v3.preprocess_input(img)
        return img


    def deprocess_image(self, x):
        # Util function to convert a NumPy array into a valid image.
        x = x.reshape((x.shape[1], x.shape[2], 3))
        # Undo inception v3 preprocessing
        x /= 2.0
        x += 0.5
        x *= 255.0
        # Convert to uint8 and clip to the valid range [0, 255]
        x = np.clip(x, 0, 255).astype("uint8")
        return x

    def compute_loss(self, input_image):
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

    def gradient_ascent_step(self, img, learning_rate):
        with tf.GradientTape() as tape:
            tape.watch(img)
            loss = self.compute_loss(img)
        # Compute gradients.
        grads = tape.gradient(loss, img)
        # Normalize gradients.
        grads /= tf.maximum(tf.reduce_mean(tf.abs(grads)), 1e-6)
        img += learning_rate * grads
        return loss, img


    def gradient_ascent_loop(self, img, iterations, learning_rate, max_loss=None):
        for i in range(iterations):
            loss, img = self.gradient_ascent_step(img, learning_rate)
            if max_loss is not None and loss > max_loss:
                break
            print("... Loss value at step %d: %.2f" % (i, loss))
        return img

    def deep_dream_filtering(self, img):

        original_img = self.preprocess_image(img)
        original_shape = original_img.shape[1:3]

        successive_shapes = [original_shape]
        for i in range(1, self.num_octave):
            shape = tuple([int(dim / (self.octave_scale ** i)) for dim in original_shape])
            successive_shapes.append(shape)
        successive_shapes = successive_shapes[::-1]
        shrunk_original_img = tf.image.resize(original_img, successive_shapes[0])

        img = tf.identity(original_img)  # Make a copy
        for i, shape in enumerate(successive_shapes):
            print("Processing octave %d with shape %s" % (i, shape))
            img = tf.image.resize(img, shape)
            img = self.gradient_ascent_loop(
                img, iterations=self.iterations, learning_rate=self.step, max_loss=self.max_loss
            )
            upscaled_shrunk_original_img = tf.image.resize(shrunk_original_img, shape)
            same_size_original = tf.image.resize(original_img, shape)
            lost_detail = same_size_original - upscaled_shrunk_original_img

            img += lost_detail
            shrunk_original_img = tf.image.resize(original_img, shape)

        keras.preprocessing.image.save_img(self.result_prefix + ".png", self.deprocess_image(img.numpy()))

        display(Image(self.result_prefix + ".png"))

if __name__ == "__main__":
    set_gpu_enabled(True)
    model = inception_v3.InceptionV3(weights="imagenet", include_top=False)
    image = load_img("doggo.jpg")
    dream = DeepDream(model)

    dream.deep_dream_filtering(image)