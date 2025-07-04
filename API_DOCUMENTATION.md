# API Documentation

## Project Overview

This project provides a Flask web application for image classification using ResNet50 models in different formats. It supports:
- Standard TensorFlow Keras models
- TensorFlow Saved Model format
- TensorRT compiled models  
- Quantized TensorFlow Lite models

The application allows performance comparison between different model formats for the same ResNet50 architecture.

## Installation and Setup

### Prerequisites

- Python 3.x
- CUDA-capable GPU (for TensorRT support)
- TensorRT installed (for TensorRT model compilation)

### Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
- `tensorflow==2.4.0` - Core TensorFlow framework
- `pillow==7.2.0` - Image processing library
- `flask==1.1.2` - Web framework
- `numpy==1.19.4` - Numerical computing library

### Initial Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Create and compile models: `python3 model_converter.py`
3. Run the Flask server: `python3 flask-keras-prediction.py`

## Web API Endpoints

### POST /predict/

**Description:** Classifies uploaded images using ResNet50 and returns top predictions.

**Request:**
- **Method:** POST
- **Content-Type:** multipart/form-data
- **Parameters:**
  - `image` (file): Image file to classify (JPEG, PNG, etc.)

**Response:**
```json
{
  "success": true,
  "predictions": [
    {
      "label": "Egyptian cat",
      "probability": 0.8124
    },
    {
      "label": "tabby, tabby cat",
      "probability": 0.1234
    }
  ]
}
```

**Error Response:**
```json
{
  "success": false
}
```

**Usage Example:**
```bash
curl -X POST -F "image=@cat.jpg" http://localhost:8000/predict/
```

**Python Example:**
```python
import requests

url = "http://localhost:8000/predict/"
files = {"image": open("cat.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

## Command Line Interface

### flask-keras-prediction.py

**Usage:**
```bash
python3 flask-keras-prediction.py [OPTIONS]
```

**Options:**
- `--use-trt-model` - Use TensorRT compiled model for inference
- `--use-tf-model` - Use TensorFlow Saved Model format
- `--use-quantized-model` - Use quantized TensorFlow Lite model
- (no flags) - Use standard TensorFlow Keras model

**Examples:**
```bash
# Use standard Keras model
python3 flask-keras-prediction.py

# Use TensorRT compiled model
python3 flask-keras-prediction.py --use-trt-model

# Use TensorFlow Saved Model
python3 flask-keras-prediction.py --use-tf-model

# Use quantized model
python3 flask-keras-prediction.py --use-quantized-model
```

## Public Functions and Components

### utils.py

#### check_paths()
**Description:** Creates necessary directories for model storage if they don't exist.

**Parameters:** None

**Returns:** None

**Usage:**
```python
from utils import check_paths
check_paths()
```

#### load_imagenet_model()
**Description:** Loads pre-trained ResNet50 model from TensorFlow Keras with ImageNet weights.

**Parameters:** None

**Returns:** `tensorflow.keras.Model` - Pre-trained ResNet50 model

**Usage:**
```python
from utils import load_imagenet_model
model = load_imagenet_model()
```

#### load_tf_model(model_dir)
**Description:** Loads a model from TensorFlow Saved Model format.

**Parameters:**
- `model_dir` (str): Path to the saved model directory

**Returns:** `tensorflow.python.saved_model.load.Loader` - Loaded TensorFlow model

**Usage:**
```python
from utils import load_tf_model
model = load_tf_model('models/resnet50')
```

#### load_saved_trt_model(model_dir)
**Description:** Loads a TensorRT compiled model.

**Parameters:**
- `model_dir` (str): Path to the TensorRT model directory

**Returns:** `tensorflow.python.framework.convert_to_constants.ConcreteFunction` - TensorRT compiled model

**Usage:**
```python
from utils import load_saved_trt_model
trt_model = load_saved_trt_model('models/trt_resnet50')
```

#### load_saved_quantized_model(path_to_file)
**Description:** Loads a quantized TensorFlow Lite model.

**Parameters:**
- `path_to_file` (str): Path to the .tflite model file

**Returns:** 
- `interpreter` (tf.lite.Interpreter): TensorFlow Lite interpreter
- `input_details` (list): Input tensor details
- `output_details` (list): Output tensor details

**Usage:**
```python
from utils import load_saved_quantized_model
interpreter, input_details, output_details = load_saved_quantized_model('models/quantized_resnet50/quant_model.tflite')
```

#### prepare_image(image, target)
**Description:** Preprocesses PIL image for ResNet50 inference.

**Parameters:**
- `image` (PIL.Image): Input image
- `target` (tuple): Target dimensions (width, height)

**Returns:** `numpy.ndarray` - Preprocessed image array ready for model inference

**Usage:**
```python
from PIL import Image
from utils import prepare_image

image = Image.open('cat.jpg')
processed_image = prepare_image(image, target=(224, 224))
```

#### make_prediction_using_trt_model(model, image)
**Description:** Makes predictions using TensorRT compiled model.

**Parameters:**
- `model`: TensorRT compiled model
- `image` (numpy.ndarray): Preprocessed image array

**Returns:** `numpy.ndarray` - Prediction probabilities

**Usage:**
```python
from utils import make_prediction_using_trt_model
predictions = make_prediction_using_trt_model(trt_model, processed_image)
```

#### make_prediction_using_tf_model(model, image)
**Description:** Makes predictions using TensorFlow Saved Model.

**Parameters:**
- `model`: TensorFlow Saved Model
- `image` (numpy.ndarray): Preprocessed image array

**Returns:** `numpy.ndarray` - Prediction probabilities

**Usage:**
```python
from utils import make_prediction_using_tf_model
predictions = make_prediction_using_tf_model(tf_model, processed_image)
```

#### make_prediction_using_quantized_model(interpreter, input_details, output_details, image)
**Description:** Makes predictions using quantized TensorFlow Lite model.

**Parameters:**
- `interpreter` (tf.lite.Interpreter): TensorFlow Lite interpreter
- `input_details` (list): Input tensor details
- `output_details` (list): Output tensor details
- `image` (numpy.ndarray): Preprocessed image array

**Returns:** `numpy.ndarray` - Prediction probabilities

**Usage:**
```python
from utils import make_prediction_using_quantized_model
predictions = make_prediction_using_quantized_model(interpreter, input_details, output_details, processed_image)
```

### model_converter.py

#### save_trt_model(tf_saved_model_dir, trt_model_dir)
**Description:** Converts TensorFlow Saved Model to TensorRT format for optimized inference.

**Parameters:**
- `tf_saved_model_dir` (str): Path to TensorFlow Saved Model directory
- `trt_model_dir` (str): Output path for TensorRT model

**Returns:** None

**Usage:**
```python
from model_converter import save_trt_model
save_trt_model('models/resnet50', 'models/trt_resnet50')
```

#### save_quantized_model(tf_model_dir, path_to_quantized_model_file)
**Description:** Converts TensorFlow Saved Model to quantized TensorFlow Lite format.

**Parameters:**
- `tf_model_dir` (str): Path to TensorFlow Saved Model directory
- `path_to_quantized_model_file` (str): Output path for quantized model file

**Returns:** None

**Usage:**
```python
from model_converter import save_quantized_model
save_quantized_model('models/resnet50', 'models/quantized_resnet50/quant_model.tflite')
```

#### my_input_fn()
**Description:** Generator function that provides sample input data for TensorRT model compilation.

**Parameters:** None

**Returns:** Generator yielding input tensors

**Usage:**
```python
from model_converter import my_input_fn
for input_data in my_input_fn():
    print(input_data[0].shape)  # (1, 224, 224, 3)
```

## Complete Usage Example

Here's a complete example demonstrating how to use the project:

```python
# 1. Setup and model conversion
from model_converter import save_trt_model, save_quantized_model
from utils import load_imagenet_model, check_paths
import tensorflow as tf

# Create necessary directories
check_paths()

# Load base model
model = load_imagenet_model()

# Save as TensorFlow Saved Model
tf_saved_model_dir = 'models/resnet50'
tf.saved_model.save(model, tf_saved_model_dir)

# Convert to TensorRT
trt_model_dir = 'models/trt_resnet50'
save_trt_model(tf_saved_model_dir, trt_model_dir)

# Convert to quantized model
quantized_model_path = 'models/quantized_resnet50/quant_model.tflite'
save_quantized_model(tf_saved_model_dir, quantized_model_path)

# 2. Using different model formats
from utils import (load_saved_trt_model, load_saved_quantized_model, 
                   prepare_image, make_prediction_using_trt_model,
                   make_prediction_using_quantized_model)
from PIL import Image
from tensorflow.keras.applications import imagenet_utils

# Load image and preprocess
image = Image.open('cat.jpg')
processed_image = prepare_image(image, target=(224, 224))

# Using TensorRT model
trt_model = load_saved_trt_model(trt_model_dir)
trt_predictions = make_prediction_using_trt_model(trt_model, processed_image)
trt_results = imagenet_utils.decode_predictions(trt_predictions)

# Using quantized model
interpreter, input_details, output_details = load_saved_quantized_model(quantized_model_path)
quantized_predictions = make_prediction_using_quantized_model(
    interpreter, input_details, output_details, processed_image)
quantized_results = imagenet_utils.decode_predictions(quantized_predictions)

print("TensorRT Results:", trt_results[0][:3])
print("Quantized Results:", quantized_results[0][:3])
```

## Performance Considerations

### Model Format Comparison

1. **Standard Keras Model**: Baseline performance, easiest to use
2. **TensorFlow Saved Model**: Slightly optimized, good for deployment
3. **TensorRT Model**: Highly optimized for NVIDIA GPUs, best inference speed
4. **Quantized Model**: Reduced memory footprint, good for mobile/edge deployment

### Memory Usage

- **Standard/TF Saved Model**: ~100MB
- **TensorRT Model**: ~25-50MB (depending on optimization)
- **Quantized Model**: ~25MB

### Inference Speed (approximate)

- **Standard Keras**: 100ms per image
- **TensorFlow Saved Model**: 80ms per image
- **TensorRT Model**: 20-40ms per image
- **Quantized Model**: 60-80ms per image

## Error Handling

The API returns appropriate error responses:

- **Missing image**: `{"success": false}`
- **Invalid image format**: `{"success": false}`
- **Model loading errors**: Server startup will fail with descriptive error messages

## Configuration

### Model Paths

- TensorFlow Saved Model: `models/resnet50/`
- TensorRT Model: `models/trt_resnet50/`
- Quantized Model: `models/quantized_resnet50/quant_model.tflite`

### Server Configuration

- **Host**: `0.0.0.0` (accepts connections from any IP)
- **Port**: `8000`
- **Threading**: Disabled by default (set `threaded=True` for concurrent requests)

## Troubleshooting

### Common Issues

1. **TensorRT not found**: Install TensorRT and ensure it's in your PATH
2. **CUDA errors**: Ensure CUDA is properly installed and compatible
3. **Memory errors**: Reduce batch size or use quantized models
4. **Model loading errors**: Ensure models are properly created using `model_converter.py`

### Debug Mode

Enable Flask debug mode for development:
```python
app.run(host='0.0.0.0', port='8000', debug=True)
```