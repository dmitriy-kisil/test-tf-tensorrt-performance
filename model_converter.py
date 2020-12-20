# import the necessary packages
import numpy as np
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt

from utils import load_imagenet_model


def save_trt_model(input_saved_model_dir, output_saved_model_dir):
    conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
    conversion_params = conversion_params._replace(
        max_workspace_size_bytes=(1 << 32))
    # If you have GPU with FP16 support, then you may uncomment line below
    # conversion_params = conversion_params._replace(precision_mode="FP16")
    conversion_params = conversion_params._replace(
        maximum_cached_engines=100)

    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=input_saved_model_dir,
        conversion_params=conversion_params)
    converter.convert()

    converter.build(input_fn=my_input_fn)
    converter.save(output_saved_model_dir)


def my_input_fn():
    input_shapes = model.input_shape
    # For Resnet50 input.shape will be (?, 224, 224, 3)
    input_shapes = list(input_shapes)
    if input_shapes[0] is None:
        # ? would be recognized as None and TensorRT could not handle this.
        # So replace ? with 1 will help
        input_shapes[0] = 1
    input_shapes = tuple(input_shapes)
    # Even if your model have one input, function should return a tuple
    input = np.random.normal(size=input_shapes).astype(np.float32)
    yield (input, )


if __name__ == "__main__":
    input_saved_model_dir, output_saved_model_dir = 'models/resnet50', 'models/trt_resnet50'
    # Load Resnet50 directly from tensorflow.keras
    model = load_imagenet_model()
    # Saved model as Tensorflow Saved Model format
    tf.saved_model.save(model, input_saved_model_dir)
    # Compile Tensorflow Saved Model model with TensorRT
    save_trt_model(input_saved_model_dir, output_saved_model_dir)
