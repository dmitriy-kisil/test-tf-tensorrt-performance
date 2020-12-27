# import the necessary packages
import io
import argparse

import flask
from PIL import Image
from tensorflow.keras.applications import imagenet_utils

from utils import check_paths
from utils import prepare_image, make_prediction_using_trt_model, make_prediction_using_tf_model
from utils import make_prediction_using_quantized_model
from utils import load_saved_trt_model, load_tf_model, load_imagenet_model, load_saved_quantized_model

# initialize our Flask application and the Keras model

app = flask.Flask(__name__)


@app.route("/predict/", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # preprocess the image and prepare it for classification
            image = prepare_image(image, target=(224, 224))

            # classify the input image and then initialize the list
            # of predictions to return to the client
            if args.use_trt_model:
                preds = make_prediction_using_trt_model(model, image)
            elif args.use_tf_model:
                preds = make_prediction_using_tf_model(model, image)
            elif args.use_quantized_model:
                preds = make_prediction_using_quantized_model(interpreter, input_details, output_details, image)
            else:
                preds = model.predict(image)
            results = imagenet_utils.decode_predictions(preds)
            data["predictions"] = []

            # loop over the results and add them to the list of
            # returned predictions
            for (imagenetID, label, prob) in results[0]:
                r = {"label": label, "probability": float(prob)}
                data["predictions"].append(r)

            # indicate that the request was a success
            data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    check_paths()
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-trt-model", action="store_true", help="use compiled with TensorRT model")
    parser.add_argument("--use-tf-model", action="store_true", help="use model, converted to TF Saved Model format")
    parser.add_argument("--use-quantized-model", action="store_true", help="use model, quantized with Tensorflow")
    args = parser.parse_args()
    if args.use_trt_model:
        # Load model, compiled with TensorRT
        trt_model_dir = 'models/trt_resnet50'
        model = load_saved_trt_model(trt_model_dir)
    elif args.use_tf_model:
        # Load model from TF Saved Model format
        tf_saved_model_dir = 'models/resnet50'
        model, input_details, output_details = load_tf_model(tf_saved_model_dir)
    elif args.use_quantized_model:
        # Load model from TF Saved Model format
        path_to_quantized_model_file = 'models/quantized_resnet50/quant_model.tflite'
        interpreter, input_details, output_details = load_saved_quantized_model(path_to_quantized_model_file)
    else:
        # Otherwise load model directly from tensorflow.keras
        model = load_imagenet_model()

    # Add threaded=False if you want to use keras instead of tensorflow.keras
    app.run(host='0.0.0.0', port='8000', threaded=False)
