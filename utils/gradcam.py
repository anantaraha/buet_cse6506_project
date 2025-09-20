
import cv2 as cv
import numpy as np
from matplotlib import colormaps

import tensorflow as tf
from tensorflow.keras.models import Model

class GradcamModel:

    def __init__(self, model, layer_name) -> None:
        # Remove last layer's softmax
        #model.layers[-1].activation = None

        # First, we create a model that maps the input image to the activations
        # of the last conv layer as well as the output predictions
        #self.grad_model = Model(
        #    model.inputs, [model.get_layer(layer_name).output, model.layers[-1].output]
        #)
        self.layer_name = layer_name
        
        # Create a logits model without modifying the original
        output_layer = model.layers[-1]
        logits_model = Model(
            model.inputs, 
            [model.get_layer(self.layer_name).output, output_layer.input]
        )
        self.grad_model = logits_model
    
    def _make_gradcam_heatmap(self, img_array, pred_index=None):
        # We compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            #print(self.grad_model.input_shape)
            #print(img_array.shape)
            #print(f"Model outputs: {[output.shape for output in self.grad_model.outputs]}")
            last_conv_layer_output, preds = self.grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]
        
        #print('Prediction:', pred_index.numpy())
        # This is the gradient of the output neuron (top predicted or chosen)
        # with regard to the output feature map of the last conv layer
        grads = tape.gradient(class_channel, last_conv_layer_output)

        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        # then sum all the channels to obtain the heatmap class activation
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return pred_index.numpy(), heatmap.numpy()

    def generate_heatmaps(self, img_path, alpha=0.5):
        # Load the original image
        #img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        img = np.array(cv.cvtColor(cv.imread(img_path, cv.IMREAD_UNCHANGED), cv.COLOR_BGR2RGB), dtype=np.uint8)
        #img = tf.keras.utils.img_to_array(img)
        
        # Make gradcam
        pred_index, heatmap = self._make_gradcam_heatmap(np.expand_dims(img, axis=0) / 255.)

        # Rescale heatmap to a range 0-255
        heatmap = np.uint8(255 * heatmap)

        # Use jet colormap to colorize heatmap
        jet = colormaps["jet"]

        # Use RGB values of the colormap
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]

        # Create an image with RGB colorized heatmap
        jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
        jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap)

        # Superimpose the heatmap on original image
        superimposed_img = jet_heatmap * alpha + img
        superimposed_img = tf.keras.utils.array_to_img(superimposed_img)

        return pred_index, img, heatmap, superimposed_img
