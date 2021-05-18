import tensorflow as tf
from tensorflow import keras
from keras import backend as back
from models import MobileNetV3
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from IPython.display import Image, display
import math

ALPHA_VAL = 0.003

def make_heatmap_from_test_img(model, img_generator, layer_name, img_index, ifBase = False):
    numberOfBatches = math.ceil(img_index/img_generator.batch_size)
    x, y = img_generator.test_array(numberOfBatches)
    if (len(x)<img_index-1):
        print("Image index too large")
        return

    img_array = keras.preprocessing.image.img_to_array(x[img_index])
    img_array = np.expand_dims(img_array, axis=0)
    if(ifBase):
        heatmap = make_gradcam_heatmap(img_array, model.base_model, layer_name)
    else:
        heatmap = make_gradcam_heatmap(img_array, model.model, layer_name)
    plt.matshow(heatmap)
    plt.show()
    save_and_display_gradcam(x[img_index], heatmap, alpha = ALPHA_VAL)

def make_gradcam_heatmap(img_array, model, layer_name, pred_index=None):
    #mapping input and output of choosed layer
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )
    #print("MODEL.INPUTS")
    #print(model.inputs)
    #print("")
    #print("LAYER OUTPUT")
    #print(model.get_layer(layer_name).output)
    #print("")
    #print("MODEL OUTPUT")
    #print(model.output)
    #print("")
    #compute gradient
    with tf.GradientTape() as tape:
        layer_output, preds = grad_model(img_array)
        #print("LAYER OUTPUT")
        #print(layer_output)
        #print("")
        #print("PREDS")
        #print(preds)
        #print("")
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, layer_output)
    #print("GRADS")
    #print(grads)
    #print("")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    #print("Pooled GRADS")
    #print(pooled_grads)
    #print("")
    layer_output = layer_output[0]
    heatmap = layer_output @ pooled_grads[..., tf.newaxis]
    #print("HEATMAP")
    #print(heatmap)
    #print("")

    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img, heatmap, cam_path="cam.jpg", alpha=0.4):
    #preprocessing original image
    img = keras.preprocessing.image.img_to_array(img)
    #rescale heatmap to img size
    heatmap = np.uint8(255 * heatmap)
    #choose jet colormap
    jet = cm.get_cmap("jet")
    #Create image with based on RGB colorized heatmap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)
    #combine image with heatmap
    # correct heatmap scale to match image
    superimposed_img = 255 * jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    superimposed_img.save(cam_path)
    display(Image(cam_path))
    return

