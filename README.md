This project allow you to run Resnet50 and compiled in TensorRT the same model to make comparison

Usage:

Note: you should be able to install Tensorflow with GPU support and have installed TensorRT!

Install dependencies:
`pip3 -r requirements.txt`

Run `python3 model_converter.py` to create compile model using TensorRT

Run `flask-keras-prediction.py`:

Without any flag - server will use model from tensorflow.keras

`--use-tf-model` = server will use the same model, converted to TF Saved Model format

`--use-trt-model` = server will use the same model, compiled with TensorRT

`--use-quantized-model` = server will use the same model, quantized with Tensorflow
