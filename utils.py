# import the necessary packages
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.framework import convert_to_constants


def check_paths():
    if not os.path.exists('models'):
        os.mkdir('models')
    if not os.path.exists('models/quantized_resnet50'):
        os.mkdir('models/quantized_resnet50')


def load_imagenet_model():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    model = ResNet50(weights="imagenet")
    return model


def load_tf_model(model_dir):
    # load model in TF Saved Model format
    model = tf.saved_model.load(model_dir)
    return model


def load_saved_trt_model(model_dir):
    # Load Resnet50 model, compiled with TensorRT
    saved_model_loaded = tf.saved_model.load(
        model_dir, tags=[tag_constants.SERVING])
    graph_func = saved_model_loaded.signatures[
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    frozen_func = convert_to_constants.convert_variables_to_constants_v2(
        graph_func)
    return frozen_func


def load_saved_quantized_model(path_to_file):
    # Load Resnet50 model, quantized with Tensorflow
    interpreter = tf.lite.Interpreter(path_to_file)
    interpreter.allocate_tensors()
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details


def make_prediction_using_trt_model(model, image):
    image = tf.convert_to_tensor(image)
    preds = model(image)[0].numpy()
    return preds


def make_prediction_using_tf_model(model, image):
    image = tf.convert_to_tensor(image)
    preds = model(image).numpy()
    return preds


def make_prediction_using_quantized_model(interpreter, input_details, output_details, image):
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])
    return preds


def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    # return the processed image
    return image
